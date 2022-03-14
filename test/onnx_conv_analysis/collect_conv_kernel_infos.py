import os
import argparse
import pickle
import json
from textwrap import indent
from typing import Dict
import numpy as np
import onnx
from onnx import helper, shape_inference
from onnx import AttributeProto, TensorProto, GraphProto, NodeProto, ValueInfoProto


def file_base_name(file_name):
    if '.' in file_name:
        separator_index = file_name.index('.')
        base_name = file_name[:separator_index]
        return base_name
    else:
        return file_name


def path_base_name(path):
    file_name = os.path.basename(path)
    return file_base_name(file_name)


def extract_value_info_shape(value_info : ValueInfoProto):
    shape = value_info.type.tensor_type.shape
    dims = [dim.dim_value for dim in shape.dim]
    return dims


def extract_tensor_shape(tensor : TensorProto):
    dims = list(tensor.dims)
    return dims


def extract_node_input_output_shapes(node : NodeProto, value_info_map : Dict[str, ValueInfoProto], initializer_map : Dict[str, TensorProto]):
    input_names = node.input
    output_names = node.output
    input_shapes = []
    output_shapes = []
    for x in input_names:
        if x in value_info_map:
            shape = extract_value_info_shape(value_info_map[x])
        else:
            shape = extract_tensor_shape(initializer_map[x])
        input_shapes.append(shape)
    for x in output_names:
        shape = extract_value_info_shape(value_info_map[x])
        output_shapes.append(shape)
    return input_shapes, output_shapes


def extract_conv_attribute(node : NodeProto):
    assert(node.op_type == 'Conv')
    attributes = node.attribute
    attribute_dict = {
        'auto_pad' : 'NOTSET',
        'dilations' : [1, 1],
        'group' : 1,
        'kernel_shape' : [3, 3],
        'pads' : [1, 1, 1, 1],
        'strides' : [1, 1],
    }
    for attr in attributes:
        if attr.name in ['dilations', 'kernel_shape', 'pads', 'strides']:
            attribute_dict[attr.name] = list(attr.ints)
        elif attr.name == 'group':
            attribute_dict[attr.name] = attr.i
        elif attr.name == 'auto_pad':
            attribute_dict[attr.name] = attr.s
        else:
            print('Warning: unrecognized attribute name:', attr.name)
    return attribute_dict


def main(args):
    # load model
    base_name = path_base_name(args.model)
    model = onnx.load(args.model)
    model = shape_inference.infer_shapes(model)
    graph = model.graph
    value_infos = graph.value_info
    graph_initializers = graph.initializer

    # construct name-to-tensor map
    value_info_map = {}
    for value in value_infos:
        value_info_map[value.name] = value
    for item in graph.input:
        value_info_map[item.name] = item
    for item in graph.output:
        value_info_map[item.name] = item
    # print(sorted(value_info_map.keys()))
    
    # construct name-to-weight map
    initializer_map = {}
    for init in graph_initializers:
        initializer_map[init.name] = init
    # print(sorted(initializer_map.keys()))

    conv_nodes = [node for node in graph.node if node.op_type == 'Conv']
    conv_infos = []
    for node in conv_nodes:
        input_shape, output_shape = extract_node_input_output_shapes(node, value_info_map, initializer_map)
        attribute = extract_conv_attribute(node)
        conv_infos.append({
            'model' : base_name,
            'node' : node.name,
            'input' : input_shape,
            'output' : output_shape,
            'attribute' : attribute,
        })
    return conv_infos


def reduce_list(x):
    from functools import reduce
    return reduce(lambda a, b : a + b, x)


if __name__ == '__main__':

    parser = argparse.ArgumentParser('Collect Conv Kernel infos from ONNX models.')
    parser.add_argument('--model', type = str, default = None)
    parser.add_argument('--model-dir', type = str, default = None)
    parser.add_argument('--output-dir', type = str, default = './experiment_data/conv_kernel_infos/')
    args = parser.parse_args()
    print(args)

    conv_infos = []

    if args.model:
        print('extracting Conv info from model:', args.model)
        info = main(args)
        conv_infos.append(info)
    elif args.model_dir:
        model_list = os.listdir(args.model_dir)
        model_list = [i for i in model_list if i.split('.')[-1] == 'onnx']
        for model in model_list:
            args.model = os.path.join(args.model_dir, model)
            print('extracting Conv info from model:', args.model)
            try:
                info = main(args)
                conv_infos.append(info)
            except Exception as e:
                print('Error:', e)
    else:
        print('Error: either model or model-dir must be specified.')
    
    merge_conv_infos = True
    if merge_conv_infos:
        conv_infos = reduce_list(conv_infos)

    os.makedirs(args.output_dir, exist_ok = True)
    save_path = os.path.join(args.output_dir, 'onnx_model_conv_infos.json')
    with open(save_path, 'w') as f:
        json.dump(conv_infos, f, indent = 4)
    print('conv info saved to file:', save_path)
    print(conv_infos)
