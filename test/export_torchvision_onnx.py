from email.policy import default
import os
from random import choices
import sys
import argparse

import torch
import torchvision.models as models


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))


def main(args):
    if args.batch_size <= 0:
        print('Error: --batch-size must be >= 0.')
        return
    onnx_dir = os.path.join(args.output_dir, 'batch_{}'.format(args.batch_size))
    os.makedirs(onnx_dir, exist_ok = True)
    onnx_file = os.path.join(onnx_dir, 'onnx_{}_imagenet_batch_{}.onnx'.format(args.model, args.batch_size))
    model = models.__dict__[args.model](pretrained = True)
    x = torch.randn((args.batch_size, 3, 224, 224))
    torch.onnx.export(model, x, onnx_file)
    print('onnx model saved to {}'.format(onnx_file))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Export models in torchvision to onnx format.')
    parser.add_argument('model', type = str, choices = model_names + ['all'])
    parser.add_argument('--batch-size', type = int, default = 1)
    parser.add_argument('--output-dir', type = str, default = './experiment_data/onnx_models/')
    args = parser.parse_args()
    print(args)
    if args.model == 'all':
        for model in model_names:
            args.model = model
            try:
                main(args)
            except Exception as e:
                print('Error:', e)
    else:
        main(args)
