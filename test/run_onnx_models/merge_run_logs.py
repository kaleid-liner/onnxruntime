import os
import sys
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


def process_line(args, line : str, is_single_node_model : bool = False):

    items = line.strip('\n').split(',')
    res = []
    # model
    base_name = path_base_name(items[0])
    base_name_parts = base_name.split('_')
    if is_single_node_model:
        items[0] = '_'.join(base_name_parts[1:-7] + base_name_parts[-4:-2])
    else:
        items[0] = '_'.join(base_name_parts[1:-3])
    # repeat, latency
    res = items[:3]
    # energy
    res.append(items[3 + args.gpu_id])
    if is_single_node_model:
        # parallel nodes
        res.append(base_name_parts[-1])
    return ','.join(res) + '\n'



def main(args):

    if args.logs_dir is None:
        print('Error: --logs-dir must be specified.')
    log_files = os.listdir(args.logs_dir)
    log_files = [item for item in log_files if item.split('.')[-1] == 'txt' and item.startswith('onnx')]

    merged_lines = []
    header = ''
    for log_file in log_files:
        with open(os.path.join(args.logs_dir, log_file)) as f:
            lines = f.readlines()
            for line in lines:
                if line.startswith('#'):
                    header = line
                else:
                    line_processed = process_line(args, line, args.single_node_model)
                    merged_lines.append(line_processed)

    merged_lines.sort()

    if args.single_node_model:
        header = header.strip('\n') + ',parallel_nodes\n'
    
    with open(args.output, 'w') as f:
        f.write(header)
        f.writelines(merged_lines)


if __name__ == '__main__':

    parser = argparse.ArgumentParser('Merge GPU logs from run_ort.')
    parser.add_argument('--logs-dir', type = str, default = None)
    parser.add_argument('--output', type = str, default = './experiment_data/onnx_output/merged_run_logs.txt')
    parser.add_argument('--gpu-id', type = int, default = 0)
    parser.add_argument('--single-node-model', action = 'store_true', default = False)

    args = parser.parse_args()
    print(args)

    main(args)