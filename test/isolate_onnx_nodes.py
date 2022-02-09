import os
import sys
import copy
import argparse
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


def duplicate_node(node : NodeProto, num_repeat : int, subgraph_initializer, subgraph_output):
    op_type = node.op_type
    node_attribute = list(node.attribute)
    subgraph_initializer_names = [item.name for item in subgraph_initializer]
    # duplicate weights and outputs and nodes
    node_list = []
    subgraph_initializer_dup = []
    subgraph_output_dup = []
    for i in range(num_repeat):
        i_str = '_' + str(i)
        # duplicate initializer and modify inputs
        node_input = []
        for item in node.input:
            if item in subgraph_initializer_names:
                node_input.append(item + i_str)
            else:
                node_input.append(item)
        for item in subgraph_initializer:
            item_i = copy.deepcopy(item)
            item_i.name += i_str
            subgraph_initializer_dup.append(item_i)
        # duplicate output
        node_output = []
        for item in node.output:
            node_output.append(item + i_str)
        for item in subgraph_output:
            item_i = copy.deepcopy(item)
            item_i.name += i_str
            subgraph_output_dup.append(item_i)
        # construct node
        node_i = helper.make_node(op_type, node_input, node_output, node.name + i_str)
        node_i.attribute.extend(node_attribute)
        node_list.append(node_i)
    return node_list, subgraph_initializer_dup, subgraph_output_dup


def construct_single_node_graph(node : NodeProto, value_info_map : Dict[str, ValueInfoProto], initializer_map : Dict[str, TensorProto], num_repeat : int = 1, duplicate_weights : bool = True):
    # extract input/output info
    subgraph_initializer = []
    subgraph_input = []
    subgraph_output = []
    for i in node.input:
        if i in initializer_map:
            # input is weight/bias etc
            subgraph_initializer.append(initializer_map[i])
        else:
            # input is data tensor
            subgraph_input.append(value_info_map[i])
    for i in node.output:
        # output must be data tensor
        subgraph_output.append(value_info_map[i])
    # construct graph
    if num_repeat <= 1:
        # single node
        isolated_graph = helper.make_graph([node], node.name + '_isolated', subgraph_input, subgraph_output, subgraph_initializer)
    elif duplicate_weights:
        # duplicate weights as well as nodes, nodes share only inputs, but each node has it's own weights and outputs
        node_list, subgraph_initializer_dup, subgraph_output_dup = duplicate_node(node, num_repeat, subgraph_initializer, subgraph_output)
        isolated_graph = helper.make_graph(node_list, node.name + '_isolated', subgraph_input, subgraph_output_dup, subgraph_initializer_dup)
    else:
        # duplicate nodes only, each node has exactly the same inputs and weights,but different output
        node_list = []
        subgraph_output_dup = []
        for i in range(num_repeat):
            i_str = '_' + str(i)
            # duplicate output
            node_output = []
            for item in node.output:
                node_output.append(item + i_str)
            for item in subgraph_output:
                item_i = copy.deepcopy(item)
                item_i.name += i_str
                subgraph_output_dup.append(item_i)
            node_i = helper.make_node(node.op_type, node.input, node_output, node.name + i_str)
            node_i.attribute.extend(node.attribute)
            node_list.append(node_i)
        isolated_graph = helper.make_graph(node_list, node.name + '_isolated', subgraph_input, subgraph_output_dup, subgraph_initializer)
    return isolated_graph


def main(args):
    os.makedirs(args.output_dir, exist_ok = True)
    base_name = path_base_name(args.model)
    model_output_dir = os.path.join(args.output_dir, base_name)
    os.makedirs(model_output_dir, exist_ok = True)

    if args.parallel_nodes < 1:
        args.parallel_nodes = 1

    # load model
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
    print(sorted(value_info_map.keys()))
    
    # construct name-to-weight map
    initializer_map = {}
    for init in graph_initializers:
        initializer_map[init.name] = init
    print(sorted(initializer_map.keys()))

    # process nodes
    duplicate_weights = not args.use_shared_weights
    for node in graph.node:
        print('processing node', node.name)
        num_repeat = args.parallel_nodes
        while num_repeat >= 1:
            try:
                isolated_graph = construct_single_node_graph(node, value_info_map, initializer_map, num_repeat, duplicate_weights)
                isolated_model = helper.make_model(isolated_graph, producer_name = 'donglinbai')
                model_save_path = os.path.join(model_output_dir, base_name + '_' + node.name + '_rep_' + str(num_repeat) + '.onnx')
                onnx.save(isolated_model, model_save_path)
                onnx.checker.check_model(model_save_path)
                print('The model is checked!')
                break
            except Exception as e:
                print('Exception:', e)
                num_repeat = num_repeat // 2


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Extract nodes in onnx model to seperate models.')
    parser.add_argument('model', type = str)
    parser.add_argument('--output-dir', type = str, default = './experiment_data/output_models/')
    parser.add_argument('--parallel-nodes', type = int, default = 1)
    parser.add_argument('--use-shared-weights', action = 'store_true', default = False)
    args = parser.parse_args()
    print(args)
    main(args)
