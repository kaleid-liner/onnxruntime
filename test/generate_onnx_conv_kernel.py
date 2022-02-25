import os
import sys
import argparse

import numpy as np
import onnx
from onnx import helper, shape_inference, numpy_helper
from onnx import AttributeProto, TensorProto, GraphProto, NodeProto, ValueInfoProto


def make_attribute_dict(args):
    attribute = {
        'kernel_shape' : args.kernel_shape,
        'pads' : args.pads + args.pads,
        'dilations' : args.dilations,
        'strides' : args.strides,
        'group' : args.group,
    }
    # attribute = {
    #     'kernel_shape' : [7, 7],
    #     'pads' : [3, 3, 3, 3],
    #     'dilations' : [1, 1],
    #     'strides' : [2, 2],
    #     'group' : 1,
    # }
    return attribute


def make_input_shape(args):
    return [args.batch_size, args.input_channels] + args.image_shape


def make_weight_shapes(args):
    shape_w = [args.output_channels, args.input_channels] + args.kernel_shape
    shape_b = [args.output_channels]
    return shape_w, shape_b


def calculate_output_shape(args):
    # const int output_h = (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
    # const int output_w = (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
    height, width = args.image_shape
    kernel_h, kernel_w = args.kernel_shape
    pad_h, pad_w = args.pads
    dilation_h, dilation_w = args.dilations
    stride_h, stride_w = args.strides
    output_h = (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) // stride_h + 1
    output_w = (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) // stride_w + 1
    output_shape = [args.batch_size, args.output_channels, output_h, output_w]
    return output_shape


def construct_model(args):
    # make node
    op_type = 'Conv'
    node_input = ['X', 'W', 'B']
    node_output = ['Y']
    node_name = 'conv_node'
    attribute = make_attribute_dict(args)
    node = helper.make_node(op_type, node_input, node_output, node_name, **attribute)
    
    # prepare input/weights/output
    input_shape = make_input_shape(args)
    shape_w, shape_b = make_weight_shapes(args)
    output_shape = calculate_output_shape(args)

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


def main(args):
    model = construct_model(args)
    # check and save
    model_save_path = args.save_path
    onnx.save(model, model_save_path)
    onnx.checker.check_model(model_save_path)


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
    # save path
    parser.add_argument('--save_path', type = str, default = './model.onnx')

    args = parser.parse_args()
    print(args)
    main(args)
