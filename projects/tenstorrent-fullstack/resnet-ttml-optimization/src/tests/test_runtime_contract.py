import sys
import unittest
from unittest.mock import patch
import torch
from .fake_ttnn import FakeTTNN
from .test_compatibility import TorchTiny, TTTiny
from models.resnet_torch import TorchResNet18
from models.resnet_ttnn import TTNNResNet18
from compat.weight_mapper import load_torch_weights_into_ttnn
from compat.forward_validate import validate_forward


class RuntimeContractTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        torch.set_num_threads(2)
        self.fake=FakeTTNN()
        self.patch=patch.dict(sys.modules,{'ttnn':self.fake})
        self.patch.start()
        for key in ('layer_factory','optimization_config'):
            sys.modules.pop(key,None)
    def tearDown(self):
        self.patch.stop()

    def test_wrapper_forward_validation_including_linear(self):
        s,t=TorchTiny().eval(),TTTiny()
        t.device=t.conv1.device=t.fc.device=object()
        load_torch_weights_into_ttnn(s,t)
        r=validate_forward(s,t,torch.randn(2,3,32,32),atol=0.02,rtol=0.03)
        self.assertTrue(r.shape_compatible,str(r))
        self.assertTrue(r.numerical_compatible,str(r))
        self.assertTrue(all(row.numeric_match for row in r.rows))

    def test_weight_upload_failure_preserves_every_previous_weight(self):
        s,t=TorchTiny().eval(),TTTiny()
        t.conv1.device=t.fc.device=object()
        load_torch_weights_into_ttnn(s,t)
        old_conv,old_fc=t.conv1.weight,t.fc.weight
        self.fake.fail_upload=self.fake.uploads+3
        with self.assertRaisesRegex(RuntimeError,'Injected'):
            load_torch_weights_into_ttnn(s,t)
        self.assertIs(t.conv1.weight,old_conv)
        self.assertIs(t.fc.weight,old_fc)

    def test_resnet_runtime_cache_partial_batch_and_reload(self):
        s,t=TorchResNet18().eval(),TTNNResNet18(device=object(),image_size=33,batch_size=8)
        load_torch_weights_into_ttnn(s,t)
        for batch in (2,1,2):
            x=self.fake.from_torch(torch.randn(batch,33,33,3))
            before=self.fake.uploads
            y=t(x)
            self.assertEqual(tuple(y.shape),(1,1,batch,512))
            self.assertEqual(self.fake.uploads,before)
        backend=t.conv1._backend
        self.assertEqual(set(backend.prepared_weights),{(2,33,33),(1,33,33)})
        self.assertTrue(self.fake.conv_calls[0]['return_weights_and_bias'])
        self.assertFalse(self.fake.conv_calls[-20]['return_weights_and_bias'])
        load_torch_weights_into_ttnn(s,t)
        self.assertIsNone(t.conv1._backend)
        t(self.fake.from_torch(torch.randn(1,33,33,3)))
        self.assertIsNot(t.conv1._backend,backend)

    def test_original_rules_and_manual_override_priority(self):
        from layer_factory import MakeCNN
        a=MakeCNN(3,64,device=object(),batch_size=1,input_height=224,input_width=224)
        self.assertEqual(a.resolved_options['act_block_h_override'],64)
        self.assertEqual(a.resolved_options['shard_layout'],'height')
        self.assertTrue(a.resolved_options['config_tensors_in_dram'])
        self.assertFalse(a.resolved_options['enable_act_double_buffer'])
        self.assertFalse(a.resolved_options['deallocate_activation'])
        b=MakeCNN(3,64,defaults={'act_block_h_override':128},
                  overrides={'act_block_h_override':32},act_block_h_override=0)
        self.assertEqual(b.resolved_options['act_block_h_override'],0)
        self.assertEqual(b.option_sources['act_block_h_override'],'OVERRIDE')


if __name__ == '__main__':
    unittest.main()
