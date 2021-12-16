import os
import sys
import csv
import json
import argparse


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

    if args.input_path is None:
        print('Error: --input-path must be specified.')
        return
    
    os.makedirs(args.output_dir, exist_ok = True)
    
    with open(args.input_path) as f:
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
    
    # save in json format
    base_name = path_base_name(args.input_path)
    output_json = os.path.join(args.output_dir, 'nodes_' + base_name + '.json')
    with open(output_json, 'w') as f:
        json.dump(node_infos, f, indent = 2)
    
    # process energy
    for item in node_infos:
        item['gpu_energy'] = item['gpu_energy'].strip('[]').split(',')[args.gpu_id]

    # save in csv format
    output_csv = os.path.join(args.output_dir, 'nodes_' + base_name + '.csv')
    fieldnames = list(node_infos[0].keys())
    with open(output_csv, 'w') as f:
        f_csv = csv.DictWriter(f, fieldnames)
        f_csv.writeheader()
        f_csv.writerows(node_infos)


if __name__ == '__main__':

    parser = argparse.ArgumentParser('Extract Node Data from Profiling Results.')
    parser.add_argument('--input-path', type = str, default = None)
    parser.add_argument('--input-dir', type = str, default = None)
    parser.add_argument('--output-dir', type = str, default = './experiment_data/onnx_output/profile_nodes/')
    parser.add_argument('--gpu_id', type = int, default = 0)

    args = parser.parse_args()
    print(args)

    if args.input_path:
        main(args)
    elif args.input_dir:
        records = os.listdir(args.input_dir)
        records = [item for item in records if item.split('.')[-1] == 'json']
        for record in records:
            args.input_path = os.path.join(args.input_dir, record)
            try:
                main(args)
            except Exception as e:
                print('Error:', e)
    else:
        print('Error: input is not specified.')
        sys.exit(-1)
