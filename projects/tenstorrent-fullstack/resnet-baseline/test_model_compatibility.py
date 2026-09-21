import argparse
import json
from pathlib import Path
import torch
from models.resnet_torch import TorchResNet18, TorchResNet50, TorchResNet101, load_reference_weights
from models.resnet_ttnn import TTNNResNet18, TTNNResNet50, TTNNResNet101
from compat.model_compare import compare_models
from compat.weight_mapper import prepare_torch_weights, load_torch_weights_into_ttnn
from compat.forward_validate import validate_forward


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--depth', type=int, choices=(18,50,101), default=18)
    parser.add_argument('--image-size', type=int, default=224)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--num-classes', type=int, default=None, help='미지정: backbone; 지정: 독립 TTLinear 포함')
    parser.add_argument('--weights', type=str, help='local state_dict or backbone.pt')
    parser.add_argument('--pretrained', action='store_true', help='ImageNet weights 다운로드')
    parser.add_argument('--mapping', type=Path, help='명시적 Torch 이름 -> TTNN 이름 JSON')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--device', action='store_true', help='실제 TTNN weight 적재 및 forward 검사')
    mode.add_argument('--cpu-weights', action='store_true', help='CPU에서 fusion/format 변환까지 검사')
    parser.add_argument('--device-id', type=int, default=0)
    parser.add_argument('--l1-small-size', type=int, default=0, help='bytes; 기존 TTML 설정에 맞게 지정')
    parser.add_argument('--atol', type=float, default=0.1)
    parser.add_argument('--rtol', type=float, default=0.1)
    args = parser.parse_args()
    torch.manual_seed(42)
    torch.set_num_threads(4)
    mapping = json.loads(args.mapping.read_text()) if args.mapping else None
    torch_type = {18:TorchResNet18,50:TorchResNet50,101:TorchResNet101}[args.depth]
    ttnn_type = {18:TTNNResNet18,50:TTNNResNet50,101:TTNNResNet101}[args.depth]
    torch_model = torch_type(num_classes=args.num_classes).eval().requires_grad_(False)
    load_reference_weights(torch_model,args.depth,args.weights,pretrained=args.pretrained)
    context = None
    try:
        device = None
        if args.device:
            import ttml
            context = ttml.autograd.AutoContext.get_instance()
            context.open_device(device_ids=[args.device_id],l1_small_size=args.l1_small_size)
            device = context.get_device()
        ttnn_model = ttnn_type(device=device,image_size=args.image_size,
                              batch_size=args.batch_size,num_classes=args.num_classes)
        report = compare_models(torch_model,ttnn_model,mapping=mapping)
        print(report)
        if not report.compatible:
            return 1
        if args.cpu_weights:
            _, weights = prepare_torch_weights(torch_model,ttnn_model,mapping=mapping)
            for item in weights:
                print(f'{item.name}: {item.kind} weight={tuple(item.weight.shape)} '
                      f'bias={None if item.bias is None else tuple(item.bias.shape)}')
            print(f'CPU format conversion: PASS ({len(weights)} weighted layers); device upload: SKIPPED')
        elif args.device:
            load_torch_weights_into_ttnn(torch_model,ttnn_model,mapping=mapping)
            print('Weight load: PASS')
            dummy = torch.randn(args.batch_size,3,args.image_size,args.image_size)
            validation = validate_forward(torch_model,ttnn_model,dummy,atol=args.atol,rtol=args.rtol,mapping=mapping)
            print(validation)
            return 0 if validation.shape_compatible and validation.numerical_compatible else 2
        else:
            print('Device upload / forward validation: SKIPPED (use --device)')
        return 0
    finally:
        if context is not None:
            context.reset_graph()
            context.close_device()


if __name__ == '__main__':
    raise SystemExit(main())
