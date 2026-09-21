import sys
import unittest
from unittest.mock import patch
import torch
from torch import nn
from models.resnet_torch import TorchResNet18, TorchResNet50, TorchResNet101
from models.resnet_ttnn import TTNNResNet18, TTNNResNet50, TTNNResNet101
from models.tt_layers import TTConv2d, TTGlobalAvgPool2d, TTFlatten, TTLinear, tt_add
from models.tt_tensor import input_value, result_value
from compat.model_compare import compare_models
from compat.weight_mapper import prepare_torch_weights, load_torch_weights_into_ttnn, convert_linear_weight, _fuse_conv_bn
from .cpu_graph_reference import evaluate


class TorchTiny(nn.Module):
    def __init__(self, bias=False, groups=1):
        super().__init__()
        self.features = nn.Sequential(nn.Conv2d(3,6,3,padding=1,bias=bias,groups=groups),
                                      nn.BatchNorm2d(6),nn.ReLU())
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(6,5)
    def forward(self,x):
        return self.classifier(torch.flatten(self.pool(self.features(x)),1))


class TTTiny:
    image_size = 32
    device = None
    def __init__(self,bias=False,groups=1):
        self.conv1 = TTConv2d(3,6,None,3,1,1,bias=bias,groups=groups,
                             batch_norm=True,relu=True,name='conv1')
        self.pool = TTGlobalAvgPool2d()
        self.flatten = TTFlatten()
        self.fc = TTLinear(6,5,name='fc')
    def __call__(self,x):
        x=input_value(x,self.image_size)
        x=self.conv1(x)
        x=self.pool(x)
        x=self.flatten(x)
        x=self.fc(x)
        return result_value(x)


class CompatibilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
    def setUp(self):
        torch.manual_seed(123)
        self.s,self.t = TorchTiny().eval(),TTTiny()

    def test_all_resnet_depths_and_linear(self):
        for source,target in ((TorchResNet18,TTNNResNet18),(TorchResNet50,TTNNResNet50),(TorchResNet101,TTNNResNet101)):
            with self.subTest(source=source.__name__):
                r=compare_models(source(num_classes=37).eval(),target(image_size=33,num_classes=37))
                self.assertTrue(r.compatible,str(r))
                self.assertEqual(r.weight_mapping['fc'],'fc')

    def test_names_differ_and_explicit_mapping(self):
        r=compare_models(self.s,self.t,mapping={'features.0':'conv1','classifier':'fc'})
        self.assertTrue(r.compatible,str(r))
        self.assertEqual(r.weight_mapping['features.0'],'conv1')

    def test_wrong_explicit_mapping_and_duplicate_targets(self):
        for m in ({'features.0':'fc'},{'features.0':'conv1','classifier':'conv1'},{'missing':'fc'}):
            self.assertFalse(compare_models(self.s,self.t,mapping=m).compatible)

    def test_parameter_attributes_mismatch(self):
        for name,value in [('has_bias',True),('dilation',(2,2)),('padding',(0,0)),
                           ('stride',(2,2)),('groups',3),('bn_eps',0.001),('relu',False)]:
            with self.subTest(field=name):
                target=TTTiny()
                setattr(target.conv1,name,value)
                r=compare_models(self.s,target)
                self.assertFalse(r.compatible,str(r))

    def test_channel_mismatch_diagnostic(self):
        self.t.conv1.out_channels=12
        r=compare_models(self.s,self.t)
        self.assertFalse(r.compatible)
        self.assertIn('mismatch',str(r))

    def test_grouped_conv(self):
        r,converted=prepare_torch_weights(TorchTiny(groups=3).eval(),TTTiny(groups=3))
        self.assertTrue(r.compatible)
        self.assertEqual(converted[0].weight.shape,(6,1,3,3))

    def test_bn_fusion_and_linear_layout(self):
        bn=self.s.features[1]
        with torch.no_grad():
            bn.running_mean.copy_(torch.randn(6))
            bn.running_var.copy_(torch.rand(6)+0.5)
            bn.weight.copy_(torch.randn(6))
            bn.bias.copy_(torch.randn(6))
        original={k:v.clone() for k,v in self.s.state_dict().items()}
        r,converted=prepare_torch_weights(self.s,self.t)
        x=torch.randn(2,3,32,32)
        y=evaluate(r.ttnn_graph,converted,x)
        with torch.no_grad():
            expected=self.s(x)
        self.assertTrue(torch.allclose(y,expected,atol=0.01,rtol=0.02),(y-expected).abs().max())
        self.assertEqual(tuple(converted[-1].weight.shape),(6,5))
        self.assertIsNone(self.t.conv1.weight)
        for key,value in original.items():
            self.assertTrue(torch.equal(value,self.s.state_dict()[key]))

    def test_resnet18_numeric_graph_and_partial_batch(self):
        s,t=TorchResNet18().eval(),TTNNResNet18(image_size=33,batch_size=8)
        r,converted=prepare_torch_weights(s,t,input_shape=(2,3,33,33))
        x=torch.randn(2,3,33,33)
        with torch.no_grad():
            a=s(x)
            b=evaluate(r.ttnn_graph,converted,x)
        self.assertEqual(a.shape,b.shape)
        self.assertTrue(torch.allclose(a,b,atol=0.06,rtol=0.06),(a-b).abs().max())

    def test_bn_fusion_matches_pytorch(self):
        conv=nn.Conv2d(4,6,3,bias=True).eval()
        bn=nn.BatchNorm2d(6).eval()
        with torch.no_grad():
            bn.running_mean.copy_(torch.randn(6))
            bn.running_var.copy_(torch.rand(6)+0.1)
        w,b=_fuse_conv_bn(conv,bn)
        ref=torch.nn.utils.fusion.fuse_conv_bn_eval(conv,bn)
        torch.testing.assert_close(w,ref.weight)
        torch.testing.assert_close(b,ref.bias)

    def test_linear_shape_checks(self):
        with self.assertRaisesRegex(ValueError,'expected shape'):
            convert_linear_weight(torch.zeros(3,4),3,4)

    def test_training_and_untracked_bn_fail(self):
        self.s.train()
        self.assertFalse(compare_models(self.s,self.t).compatible)
        self.s.eval()
        self.s.features[1]=nn.BatchNorm2d(6,track_running_stats=False).eval()
        self.assertFalse(compare_models(self.s,self.t).compatible)

    def test_unsupported_operation_fails_closed(self):
        self.s.features[2]=nn.Sigmoid().eval()
        r=compare_models(self.s,self.t)
        self.assertFalse(r.compatible)
        self.assertIn('unsupported',str(r))

    def test_same_layers_different_residual_connections_fail(self):
        class Source(nn.Module):
            def __init__(self):
                super().__init__()
                self.a=nn.Conv2d(3,3,3,padding=1,bias=False)
                self.b=nn.Conv2d(3,3,3,padding=1,bias=False)
            def forward(self,x):
                a=self.a(x)
                return self.b(a)+x
        class Target:
            image_size=32
            def __init__(self):
                self.a=TTConv2d(3,3,name='a')
                self.b=TTConv2d(3,3,name='b')
            def __call__(self,x):
                a=self.a(x)
                return tt_add(self.b(a),a)
        r=compare_models(Source().eval(),Target())
        self.assertFalse(r.compatible)
        self.assertIn('topology',str(r))

    def test_nonfinite_weight_stops_before_ttnn_import(self):
        with torch.no_grad():
            self.s.classifier.weight[0,0]=float('nan')
        with patch.dict(sys.modules,{'ttnn':None}):
            with self.assertRaisesRegex(ValueError,'finite'):
                load_torch_weights_into_ttnn(self.s,self.t)
        self.assertIsNone(self.t.conv1.weight)

    def test_stale_report_does_not_authorize_loading(self):
        self.assertTrue(compare_models(self.s,self.t).compatible)
        self.t.conv1.has_bias=True
        with patch.dict(sys.modules,{'ttnn':None}):
            with self.assertRaisesRegex(RuntimeError,'bias mismatch'):
                load_torch_weights_into_ttnn(self.s,self.t)
        self.assertIsNone(self.t.conv1.weight)

    def test_invalid_override_is_not_silently_ignored(self):
        with self.assertRaisesRegex(ValueError,'Unknown layer overrides'):
            TTNNResNet18(layer_overrides={'conv_typo':{'act_block_h_override':64}})

    def test_inplace_mutation_of_residual_input_is_rejected(self):
        class Source(nn.Module):
            def __init__(self):
                super().__init__()
                self.relu=nn.ReLU(inplace=True)
            def forward(self,x):
                y=self.relu(x)
                return y+x
        self.assertIn('inplace',str(compare_models(Source().eval(),self.t)))

    def test_declared_input_size_is_checked(self):
        r=compare_models(self.s,self.t,input_shape=(1,3,33,33))
        self.assertFalse(r.compatible)
        self.assertIn('Input shape mismatch',str(r))

    def test_ttnn_constructor_no_torch_model_and_no_device_tensors(self):
        target=TTNNResNet18(image_size=33)
        self.assertIsNone(target.conv1.weight)
        self.assertIsNone(target.layer4_1.conv2.weight)


if __name__ == '__main__':
    unittest.main()
