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


def process_line(line : str, gpu_id : int = 0):

    items = line.strip('\n').split(',')
    res = []
    # model
    base_name = path_base_name(items[0])
    # MKN
    base_name_parts = base_name.split('_')
    M = base_name_parts[2]
    K = base_name_parts[3]
    N = base_name_parts[4]
    # repeat, latency, energy
    repeat = items[1]
    latency = items[2]
    energy = items[3 + gpu_id]
    res = {
        'model' : base_name,
        'M' : M,
        'K' : K,
        'N' : N,
        'repeat' : repeat,
        'latency' : latency,
        'energy' : energy,
    }
    return res


def main(args):

    if args.logs_dir is None:
        print('Error: --logs-dir must be specified.')
    log_files = os.listdir(args.logs_dir)
    log_files = [item for item in log_files if item.split('.')[-1] == 'txt' and item.startswith('onnx')]

    merged_lines = []
    for log_file in log_files:
        with open(os.path.join(args.logs_dir, log_file)) as f:
            lines = f.readlines()
        for line in lines:
            if line.startswith('#'):
                continue
            else:
                line_processed = process_line(line, args.gpu_id)
                merged_lines.append(line_processed)

    fieldnames = ['model', 'M', 'K', 'N', 'repeat', 'latency', 'energy']
    with open(args.output, 'w') as f:
        f_csv = csv.DictWriter(f, fieldnames)
        f_csv.writeheader()
        f_csv.writerows(merged_lines)


if __name__ == '__main__':

    parser = argparse.ArgumentParser('Merge GPU logs from run_ort.')
    parser.add_argument('--logs-dir', type = str, default = None)
    parser.add_argument('--output', type = str, default = './experiment_data/onnx_output/onnx_gemm_results.csv')
    parser.add_argument('--gpu-id', type = int, default = 0)

    args = parser.parse_args()
    print(args)

    main(args)
