import os
import sys
import json
import argparse
from typing import Dict

import numpy as np
import onnx
from onnx import helper, shape_inference, numpy_helper
from onnx import AttributeProto, TensorProto, GraphProto, NodeProto, ValueInfoProto


# Compute Y = alpha * A' * B' + beta * C
# A' = transpose(A) if transA else A
# B' = transpose(B) if transB else B
def construct_model(cfg : Dict[str, int]):
    # make node
    op_type = 'Gemm'
    node_input = ['A', 'B', 'C']
    node_output = ['Y']
    node_name = 'gemm_node'
    attribute = {
        'alpha' : 1.0,
        'beta' : 1.0,
        'transA' : 0,
        'transB' : 0,
    }
    node = helper.make_node(op_type, node_input, node_output, node_name, **attribute)

    # prepare input/output
    M = cfg['M']
    K = cfg['K']
    N = cfg['N']
    A = helper.make_tensor_value_info('A', TensorProto.FLOAT, (M, K))
    B = helper.make_tensor_value_info('B', TensorProto.FLOAT, (K, N))
    C = helper.make_tensor_value_info('C', TensorProto.FLOAT, (M, N))
    Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, (M, N))

    # make graph
    graph_inputs = [A, B, C]
    graph_outputs = [Y]
    graph_initializers = []
    graph = helper.make_graph([node], 'graph with single gemm node', graph_inputs, graph_outputs, graph_initializers)

    # make model
    model = helper.make_model(graph)

    return model


def main(args):
    os.makedirs(args.save_dir, exist_ok = True)
    # if config file is specified, read MNK from file (higher priority)
    # config file must be a list of dict with keys (M, K, N), and saved in json format
    if args.config_file:
        with open(args.config_file) as f:
            configs = json.load(f)
        print('loaded configs from file, number of configs:', len(configs))
    else:
        print('using specified single config.')
        configs = [{
            'M' : args.M,
            'K' : args.K,
            'N' : args.N,
        }]
    # construct models
    for cfg in configs:
        try:
            model = construct_model(cfg) 
            model_save_path = os.path.join(args.save_dir, 'onnx_gemm_{}_{}_{}.onnx'.format(cfg['M'], cfg['K'], cfg['N']))
            onnx.save(model, model_save_path)
            onnx.checker.check_model(model_save_path)
        except Exception as e:
            print('Error:', e)


if __name__ == '__main__':

    parser = argparse.ArgumentParser('Generate ONNX Gemm Kernels.')
    # config
    parser.add_argument('--M', type = int, default = 1)
    parser.add_argument('--K', type = int, default = 512)
    parser.add_argument('--N', type = int, default = 1000)
    parser.add_argument('--config-file', type = str, default = None)
    # save path
    parser.add_argument('--save-dir', type = str, default = './experiment_data/gemm_kernels/')

    args = parser.parse_args()
    print(args)
    main(args)
