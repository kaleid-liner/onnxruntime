import os
import sys
import csv
import argparse

import numpy as np
import matplotlib.pyplot as plt

ALL_OPS = {'Gather', 'Split', 'Add', 'MaxPool', 'Conv', 'LayerNormalization', 'Concat', 'FusedConv', 'Gemm', 'SkipLayerNormalization', 
            'FusedMatMul', 'Relu', 'ReduceMean', 'AveragePool', 'Transpose', 'Softmax', 'BiasGelu', 'MatMul', 'Pad', 'GlobalAveragePool', 
            'BatchNormalization', 'Div', 'HardSigmoid', 'Reshape', 'Clip', 'Unsqueeze', 'Mul', 'Flatten'}


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


def main(args):

    if args.node_path is None:
        print('Error: --node-path not specified.')
        return
    os.makedirs(args.output_dir, exist_ok = True)

    base_name = path_base_name(args.node_path)
    model = '_'.join(base_name.split('_')[2:-5])

    # read records
    with open(args.node_path) as f:
        f_csv = csv.DictReader(f)
        records = list(f_csv)

    # group by ops
    op_names = [item['op_name'] for item in records]
    op_names = set(op_names)
    print(op_names)
    records_by_ops = {}
    for op in op_names:
        records_by_ops[op] = []
    for item in records:
        op = item['op_name']
        records_by_ops[op].append(item)

    # merge latency and energy
    merge_latency_energy = {}
    for op in op_names:
        op_group = records_by_ops[op]
        node_names = [item['name'] for item in op_group]
        latencies = [float(item['gpu_latency']) for item in op_group]
        energies = [float(item['gpu_energy']) for item in op_group]
        merge_latency_energy[op] = {
            'node_names' : node_names,
            'latency' : sum(latencies),
            'energy' : sum(energies),
        }

    k = 4

    # sort by energy
    sort_by_energy = list(merge_latency_energy.items())
    sort_by_energy.sort(key = lambda x : x[1]['energy'], reverse = True)
    if len(sort_by_energy) > k:
        reserved = sort_by_energy[:k]
        to_be_merged = sort_by_energy[k:]
        merged = ('Others', {'node_names' : [], 'latency' : 0.0, 'energy' : 0.0})
        for item in to_be_merged:
            merged[1]['node_names'] += item[1]['node_names']
            merged[1]['latency'] += item[1]['latency']
            merged[1]['energy'] += item[1]['energy']
        sort_by_energy = reserved + [merged]

    # plot energy
    plt.clf()
    plt.pie([x[1]['energy'] for x in sort_by_energy], labels = [x[0] for x in sort_by_energy], normalize = True)
    plt.title('energy per kernel ({})'.format(model))
    save_file = os.path.join(args.output_dir, base_name + '_energy.png')
    plt.savefig(save_file)

    #sort by latency
    sort_by_latency = list(merge_latency_energy.items())
    sort_by_latency.sort(key = lambda x : x[1]['latency'], reverse = True)
    if len(sort_by_latency) > k:
        reserved = sort_by_latency[:k]
        to_be_merged = sort_by_latency[k:]
        merged = ('Others', {'node_names' : [], 'latency' : 0.0, 'energy' : 0.0})
        for item in to_be_merged:
            merged[1]['node_names'] += item[1]['node_names']
            merged[1]['latency'] += item[1]['latency']
            merged[1]['energy'] += item[1]['energy']
        sort_by_latency = reserved + [merged]
    
    #plot latency
    plt.clf()
    plt.pie([x[1]['latency'] for x in sort_by_latency], labels = [x[0] for x in sort_by_latency], normalize = True)
    plt.title('latency per kernel ({})'.format(model))
    save_file = os.path.join(args.output_dir, base_name + '_latency.png')
    plt.savefig(save_file)


if __name__ == '__main__':

    parser = argparse.ArgumentParser('Plot node profiling results.')
    parser.add_argument('--node-path', type = str, default = None)
    parser.add_argument('--node-dir', type = str, default = None)
    parser.add_argument('--output-dir', type = str, default = './experiment_data/onnx_output/node_plots/')

    args = parser.parse_args()
    print(args)

    if args.node_path:
        main(args)
    elif args.node_dir:
        node_files = os.listdir(args.node_dir)
        node_files = [item for item in node_files if item.split('.')[-1] == 'csv']
        count = 1
        for node_file in node_files:
            args.node_path = os.path.join(args.node_dir, node_file)
            print('[{}/{}] plot'.format(count, len(node_files)), node_file)
            try:
                main(args)
            except Exception as e:
                print('Error:', e)
            count += 1

