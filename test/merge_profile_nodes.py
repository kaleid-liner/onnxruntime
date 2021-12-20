import os
import sys
import csv
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

    if args.profile_dir is None:
        print('Error: --profile-dir must be specified.')
    node_files = os.listdir(args.profile_dir)
    node_files = [item for item in node_files if item.split('.')[-1] == 'csv']

    merged_lines = []
    for node_file in node_files:
        with open(os.path.join(args.profile_dir, node_file)) as f:
            f_csv = csv.DictReader(f)
            nodes = list(f_csv)
            energies = [float(item['gpu_energy']) for item in nodes]
            latencies = [float(item['gpu_latency']) for item in nodes]
            onnx_latencies = [float(item['onnx_latency']) / float(item['repeat']) / 1e6 for item in nodes]
            model = '_'.join(path_base_name(node_file).split('_')[2:-5])
            line = {
                'model' : model,
                'latency' : sum(latencies),
                'energy' : sum(energies),
                'onnx_latency' : sum(onnx_latencies),
            }
            merged_lines.append(line)

    merged_lines.sort(key = lambda x : x['model'])
    
    fieldnames = ['model', 'latency', 'energy', 'onnx_latency']
    with open(args.output, 'w') as f:
        f_csv = csv.DictWriter(f, fieldnames)
        f_csv.writeheader()
        f_csv.writerows(merged_lines)


if __name__ == '__main__':

    parser = argparse.ArgumentParser('Merge profiling nodes from analyze_ort.')
    parser.add_argument('--profile-dir', type = str, default = None)
    parser.add_argument('--output', type = str, default = './experiment_data/onnx_output/merged_profile_nodes.txt')

    args = parser.parse_args()
    print(args)

    main(args)
