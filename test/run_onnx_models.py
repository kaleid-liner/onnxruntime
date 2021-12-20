import os
import sys
import argparse
import subprocess


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

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    
    if args.exec_path is None or not os.path.exists(args.exec_path):
        print('Error: executable is not specified or does not exist.')
        sys.exit(-1)
    
    os.makedirs(args.output_dir, exist_ok = True)
    output_sub_dir = os.path.join(args.output_dir, args.mode)
    os.makedirs(output_sub_dir, exist_ok = True)

    base_name = path_base_name(args.model_path)

    if args.mode == 'profile':
        profile_out = os.path.join(output_sub_dir, base_name)
        cmd = [args.exec_path, '-i', args.model_path, '-p', profile_out]
        cmd = ' '.join(cmd)
        print('command:', cmd)
        subprocess.run(cmd, shell = True)
    else:
        gpu_reading = os.path.join(output_sub_dir, base_name + '.csv')
        logs_txt = os.path.join(output_sub_dir, base_name + '.txt')
        profile_out = os.path.join(output_sub_dir, base_name)
        cmd = [args.exec_path, '-i', args.model_path, '-g', gpu_reading, 
                '-l', logs_txt, '-r 1000', '-p', profile_out]
        cmd = ' '.join(cmd)
        print('command:', cmd)
        subprocess.run(cmd, shell = True)


if __name__ == '__main__':

    parser = argparse.ArgumentParser('Run ONNX Runtime with Onnx Models.')
    parser.add_argument('--model-path', type = str, default = None)
    parser.add_argument('--model-dir', type = str, default = None)
    parser.add_argument('--output-dir', type = str, default = './experiment_data/onnx_output')
    parser.add_argument('--mode', type = str, default = 'run', choices = ['run', 'profile'])
    parser.add_argument('--exec-path', type = str, default = None)

    args = parser.parse_args()
    print(args)

    if args.model_path:
        main(args)
    elif args.model_dir:
        models = os.listdir(args.model_dir)
        models = [item for item in models if item.split('.')[-1] == 'onnx']
        count = 1
        for model in models:
            args.model_path = os.path.join(args.model_dir, model)
            try:
                print('[{}/{}] running'.format(count, len(models)), args.model_path)
                main(args)
            except Exception as e:
                print('Error:', e)
            count += 1
    else:
        print('Error: model not specified.')
