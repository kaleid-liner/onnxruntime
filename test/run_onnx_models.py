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

    base_name = path_base_name(args.model_path)
    profile_out = os.path.join(args.output_dir, base_name)

    cmd = [args.exec_path, '-i', args.model_path, '-p', profile_out]
    cmd = ' '.join(cmd)
    subprocess.run(cmd, shell = True)


if __name__ == '__main__':

    parser = argparse.ArgumentParser('Run ONNX Runtime with Onnx Models.')
    parser.add_argument('--model-path', type = str, default = None)
    parser.add_argument('--model-dir', type = str, default = None)
    parser.add_argument('--output-dir', type = str, default = './onnx_output')
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
