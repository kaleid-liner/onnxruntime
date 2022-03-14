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


def extract_gemm_attribute(node : NodeProto):
    assert(node.op_type == 'Gemm')
    attributes = node.attribute
    attribute_dict = {
        'alpha' : 1.0,
        'beta' : 1.0,
        'transA' : 0,
        'transB' : 0,
    }
    for attr in attributes:
        if attr.name in ['alpha', 'beta']:
            attribute_dict[attr.name] = attr.f
        elif attr.name in ['transA', 'transB']:
            attribute_dict[attr.name] = attr.i
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

    gemm_nodes = [node for node in graph.node if node.op_type == 'Gemm']
    gemm_infos = []
    for node in gemm_nodes:
        input_shape, output_shape = extract_node_input_output_shapes(node, value_info_map, initializer_map)
        attribute = extract_gemm_attribute(node)
        gemm_infos.append({
            'model' : base_name,
            'node' : node.name,
            'input' : input_shape,
            'output' : output_shape,
            'attribute' : attribute,
        })
    return gemm_infos


def reduce_list(x):
    from functools import reduce
    return reduce(lambda a, b : a + b, x)


def extract_MKN(gemm_item):
    input_list = gemm_item['input']
    output_list = gemm_item['output']
    attribute = gemm_item['attribute']
    # param check
    if len(input_list) != 3:
        return None
    if len(input_list[0]) != 2 or len(input_list[1]) != 2 or len(input_list[2]) != 1:
        return None
    if len(output_list) != 1 or len(output_list[0]) != 2:
        return None
    # process
    M, K1 = 0, 0
    if attribute['transA']:
        K1, M = input_list[0]
    else:
        M, K1 = input_list[0]
    K2, N = 0, 0
    if attribute['transB']:
        N, K2 = input_list[1]
    else:
        K2, N = input_list[1]
    if K1 != K2 or N != input_list[2][0]:
        print('Error: incompatible dimentions in Gemm.')
        return None
    if M != output_list[0][0] or N != output_list[0][1]:
        print('Warning: error output shape.')
    return (M, K1, N)


def post_process_gemm_infos(gemm_infos):
    for item in gemm_infos:
        item['MKN'] = extract_MKN(item)


if __name__ == '__main__':

    parser = argparse.ArgumentParser('Collect Gemm Kernel infos from ONNX models.')
    parser.add_argument('--model', type = str, default = None)
    parser.add_argument('--model-dir', type = str, default = None)
    parser.add_argument('--output-dir', type = str, default = './experiment_data/gemm_kernel_infos/')
    args = parser.parse_args()
    print(args)

    gemm_infos = []

    if args.model:
        print('extracting Gemm info from model:', args.model)
        info = main(args)
        gemm_infos.append(info)
    elif args.model_dir:
        model_list = os.listdir(args.model_dir)
        model_list = [i for i in model_list if i.split('.')[-1] == 'onnx']
        for model in model_list:
            args.model = os.path.join(args.model_dir, model)
            print('extracting Gemm info from model:', args.model)
            try:
                info = main(args)
                gemm_infos.append(info)
            except Exception as e:
                print('Error:', e)
    else:
        print('Error: either model or model-dir must be specified.')
    
    merge_gemm_infos = True
    if merge_gemm_infos:
        gemm_infos = reduce_list(gemm_infos)
        post_process_gemm_infos(gemm_infos)

    os.makedirs(args.output_dir, exist_ok = True)
    save_path = os.path.join(args.output_dir, 'onnx_model_gemm_infos.json')
    with open(save_path, 'w') as f:
        json.dump(gemm_infos, f, indent = 4)
    print('gemm info saved to file:', save_path)
    print(gemm_infos)
