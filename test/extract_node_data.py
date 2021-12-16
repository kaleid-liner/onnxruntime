import os
import sys
import csv
import json
import argparse


def main(args):

    if args.input is None:
        print('Error: --input must be specified.')
        return
    
    with open(args.input) as f:
        data = json.load(f)
    
    nodes = [item for item in data if item['cat'] == 'Node']
    nodes_run=[item for item in nodes if item['name'].endswith('kernel_time')]

    print(len(nodes))
    print(len(nodes_run))

    node_infos = []
    for item in nodes_run:
        info = {
            # basic info
            'name' : item['name'],
            'op_name' : item['args']['op_name'],
            'graph_index' : item['args']['graph_index'],
            'exec_plan_index' : item['args']['exec_plan_index'],
            'provider' : item['args']['provider'],
            # performance
            'gpu_energy' : item['args']['gpu_energy'],
            'gpu_latency' : item['args']['gpu_latency'],
            'repeat' : item['args']['loop_repeat'],
            'onnx_latency' : item['dur'],
            # size
            'activation_size' : item['args']['activation_size'],
            'parameter_size' : item['args']['parameter_size'],
            'output_size' : item['args']['output_size'],
        }
        node_infos.append(info)
    
    with open(args.output_json, 'w') as f:
        json.dump(node_infos, f, indent = 2)
    
    # process energy
    for item in node_infos:
        item['gpu_energy'] = item['gpu_energy'].strip('[]').split(',')[args.gpu_id]

    fieldnames = list(node_infos[0].keys())
    with open(args.output_csv, 'w') as f:
        f_csv = csv.DictWriter(f, fieldnames)
        f_csv.writeheader()
        f_csv.writerows(node_infos)


if __name__ == '__main__':

    parser = argparse.ArgumentParser('Extract Node Data from Profiling Results.')
    parser.add_argument('--input', type = str, default = None)
    parser.add_argument('--output_json', type = str, default = 'node_infos.json')
    parser.add_argument('--output_csv', type = str, default = 'node_infos.csv')
    parser.add_argument('--gpu_id', type = int, default = 0)

    args = parser.parse_args()
    print(args)

    main(args)
