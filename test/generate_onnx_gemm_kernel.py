import os
import sys
import argparse

import numpy as np
import onnx
from onnx import helper, shape_inference, numpy_helper
from onnx import AttributeProto, TensorProto, GraphProto, NodeProto, ValueInfoProto


# Compute Y = alpha * A' * B' + beta * C
# A' = transpose(A) if transA else A
# B' = transpose(B) if transB else B
def construct_model(args):
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
    M = args.M
    K = args.K
    N = args.N
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
    model = construct_model(args)
    # check and save
    model_save_path = args.save_path
    onnx.save(model, model_save_path)
    onnx.checker.check_model(model_save_path)


if __name__ == '__main__':

    parser = argparse.ArgumentParser('Generate ONNX Gemm Kernels.')
    # config
    parser.add_argument('--M', type = int, default = 1)
    parser.add_argument('--K', type = int, default = 512)
    parser.add_argument('--N', type = int, default = 1000)
    # save path
    parser.add_argument('--save_path', type = str, default = './model.onnx')

    args = parser.parse_args()
    print(args)
    main(args)
