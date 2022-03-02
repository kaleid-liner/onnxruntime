from fileinput import filename
import os
import sys
import json
import argparse
from unicodedata import name

import numpy as np
import onnx
from onnx import helper, shape_inference, numpy_helper
from onnx import AttributeProto, TensorProto, GraphProto, NodeProto, ValueInfoProto


def make_attribute_dict(cfg):
    attribute = {
        'kernel_shape' : cfg['kernel_shape'],
        'pads' : cfg['pads'] + cfg['pads'],
        'dilations' : cfg['dilations'],
        'strides' : cfg['strides'],
        'group' : cfg['group'],
    }
    # attribute = {
    #     'kernel_shape' : [7, 7],
    #     'pads' : [3, 3, 3, 3],
    #     'dilations' : [1, 1],
    #     'strides' : [2, 2],
    #     'group' : 1,
    # }
    return attribute


def make_input_shape(cfg):
    return [cfg['batch_size'], cfg['input_channels']] + cfg['image_shape']


def make_weight_shapes(cfg):
    shape_w = [cfg['output_channels'], cfg['input_channels']] + cfg['kernel_shape']
    shape_b = [cfg['output_channels']]
    return shape_w, shape_b


def calculate_output_shape(cfg):
    # const int output_h = (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
    # const int output_w = (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
    height, width = cfg['image_shape']
    kernel_h, kernel_w = cfg['kernel_shape']
    pad_h, pad_w = cfg['pads']
    dilation_h, dilation_w = cfg['dilations']
    stride_h, stride_w = cfg['strides']
    output_h = (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) // stride_h + 1
    output_w = (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) // stride_w + 1
    output_shape = [cfg['batch_size'], cfg['output_channels'], output_h, output_w]
    return output_shape


def construct_model(cfg):
    # make node
    op_type = 'Conv'
    node_input = ['X', 'W', 'B']
    node_output = ['Y']
    node_name = 'conv_node'
    attribute = make_attribute_dict(cfg)
    node = helper.make_node(op_type, node_input, node_output, node_name, **attribute)
    
    # prepare input/weights/output
    input_shape = make_input_shape(cfg)
    shape_w, shape_b = make_weight_shapes(cfg)
    output_shape = calculate_output_shape(cfg)

    X = helper.make_tensor_value_info('X', TensorProto.FLOAT, input_shape)
    W = np.random.randn(*shape_w).astype(np.float32)
    B = np.random.randn(*shape_b).astype(np.float32)
    Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, output_shape)
    W_tensor = numpy_helper.from_array(W, 'W')
    B_tensor = numpy_helper.from_array(B, 'B')

    # make graph
    graph_inputs = [X]
    graph_outputs = [Y]
    graph_initializers = [W_tensor, B_tensor]
    graph = helper.make_graph([node], 'graph with single conv node.', graph_inputs, graph_outputs, graph_initializers)

    # make model
    model = helper.make_model(graph)

    print('input_shape:', input_shape)
    print('shape_w:', shape_w)
    print('shape_b:', shape_b)
    print('output_shape:', output_shape)

    return model


def make_model_save_name(cfg):
    save_name = 'onnx_conv'
    # input shape NCHW
    save_name += '_in_{}_{}_{}_{}'.format(cfg['batch_size'], cfg['input_channels'], cfg['image_shape'][0], cfg['image_shape'][1])
    # kernel shape NHW
    save_name += '_kernel_{}_{}_{}'.format(cfg['output_channels'], cfg['kernel_shape'][0], cfg['kernel_shape'][1])
    # attributes (assume isotropic)
    save_name += '_attr_{}_{}_{}'.format(cfg['pads'][0], cfg['dilations'][0], cfg['strides'][0])
    # finished
    save_name += '.onnx'
    return save_name


def main(args):
    os.makedirs(args.save_dir, exist_ok = True)
    # if config file is specified, read conv configs from file (higher priority)
    # config file must be a list of dict with all needed keys, and saved in json format
    if args.config_file:
        with open(args.config_file) as f:
            configs = json.load(f)
        print('loaded configs from file, number of configs:', len(configs))
    else:
        print('using specified single config.')
        configs = [args.__dict__]
    # construct models
    for cfg in configs:
        try:
            model = construct_model(cfg)
            model_save_name = make_model_save_name(cfg)
            model_save_path = os.path.join(args.save_dir, model_save_name)
            onnx.save(model, model_save_path)
            onnx.checker.check_model(model_save_path)
        except Exception as e:
            print('Error:', e)


if __name__ == '__main__':

    parser = argparse.ArgumentParser('Generate ONNX Conv Kernels.')
    # config
    parser.add_argument('--image-shape', type = int, nargs = 2, default = [224, 224])
    parser.add_argument('--input-channels', type = int, default = 3)
    parser.add_argument('--output-channels', type = int, default = 32)
    parser.add_argument('--batch-size', type = int, default = 1)
    parser.add_argument('--kernel-shape', type = int, nargs = 2, default = [7, 7])
    parser.add_argument('--pads', type = int, nargs = 2, default = [3, 3])
    parser.add_argument('--dilations', type = int, nargs = 2, default = [1, 1])
    parser.add_argument('--strides', type = int, nargs = 2, default = [2, 2])
    parser.add_argument('--group', type = int, default = 1)
    parser.add_argument('--config-file', type = str, default = None)
    # save path
    parser.add_argument('--save-dir', type = str, default = './experiment_data/conv_kernels/')

    args = parser.parse_args()
    print(args)
    main(args)
