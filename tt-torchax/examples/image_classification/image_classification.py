
import sys
import argparse

import tt_xla.utils as ttxla
from tt_xla.tt_torchax import TorchAXSingleInference, TorchAXDDPInference

import torch
from imagenet_data import imagenet_utils

def parse_args(parser):
    parser.add_argument("-ddp", help="ddp", 
                        action="store_true", dest="ddp", default=False)
    parser.add_argument("-torch", help="Use torch instead of torchAX", 
                        action="store_true", dest="torch", default=False)
    parser.add_argument("-list", help="list models", 
                        action="store_true", dest="list", default=False)
    parser.add_argument("-target", help="test target", default=0)
    args = parser.parse_args()
    return args

def image_classification_torch(m, input):
    # move the input and model to GPU for speed if available
    if torch.cuda.is_available():
        input = input.to('cuda')
        m.to('cuda')

    return m(input)

def image_classification_single(m, input):
    m = TorchAXSingleInference(m)
    return m(input)

def image_classification_ddp(m, input): 
    m_scripted = torch.jit.script(m)
    ddp_model = TorchAXDDPInference(m_scripted)
    sharded_input = ddp_model.shard_input(input)
    return ddp_model(sharded_input)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='TorchAX image classification run script')
    args = parse_args(parser)
    models = imagenet_utils.get_models()
    if args.list:
        for idx, m in enumerate(models):
            print("[", idx, ":", models[idx], "]")
        exit()

    target = int(args.target)
    if target >= len(models):
        print("target model does not exists: ", target)
        exit()

    print("[", target, ":", models[target], "]")
    batch_size = 1
    if not args.torch:
        ttxla.initialize()
        if args.ddp:
            batch_size = ttxla.get_num_devices()

    input = imagenet_utils.load_input(batch_size)
    for idx, model_name in enumerate(models):
        if idx == target:
            model = torch.hub.load('pytorch/vision:v0.10.0', model_name, pretrained=True)
            model.eval()
            if args.torch:
                output = image_classification_torch(model, input)
            elif args.ddp:
                output = image_classification_ddp(model, input)
            else:
                output = image_classification_single(model, input)
            
            print("[", idx, ":", model_name, "]")
            imagenet_utils.print_output(output)

# 6 resnet50 ddp memory allocation issue
# 7 resnet101 ddp memory allocation issue
# 8 resnet152 ddp memory allocation issue
# 9 resnext50_32x4d ddp memory allocation issue
# 10 resnext101_32x8d ddp memory allocation issue
# 11 wide_resnet50_2 ddp memory allocation issue
# 12 wide_resnet50_2 ddp memory allocation issue
