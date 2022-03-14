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


def run_onnx_model(exec_path : str, model_path : str):
    import re
    pattern = re.compile(r'#latency:([0-9]+[\.]?[0-9]*),energy:([0-9]+[\.]?[0-9]*)')
    cmd = [exec_path, '-i', model_path, '-r 10000', '-w', '-x 2', '-t 10.0']
    cmd = ' '.join(cmd)
    ret = subprocess.run(cmd, shell = True, stdout = subprocess.PIPE, stderr = subprocess.PIPE)
    ret = str(ret.stdout)
    res = pattern.findall(ret)[0]
    return {'latency' : res[0], 'energy' : res[1]}


def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    if args.exec_path is None or not os.path.exists(args.exec_path):
        print('Error: executable is not specified or does not exist.')
        sys.exit(-1)
    res = run_onnx_model(args.exec_path, args.model_path)
    print(res)
    return res


if __name__ == '__main__':

    parser = argparse.ArgumentParser('Run ONNX Runtime with Onnx Models.')
    parser.add_argument('--model-path', type = str, default = None)
    parser.add_argument('--model-dir', type = str, default = None)
    parser.add_argument('--exec-path', type = str, default = './build/run_ort_clean')
    parser.add_argument('--build', action = 'store_true', default = False)
    parser.add_argument('--project-dir', type = str, default = './')

    args = parser.parse_args()
    print(args)
    
    if args.build:
        build_dir = os.path.join(args.project_dir, 'build')
        os.makedirs(build_dir, exist_ok = True)
        cmd = 'cd {} && cmake .. && make -j'.format(build_dir)
        subprocess.run(cmd, shell = True)
        args.exec_path = os.path.join(build_dir, 'run_ort_clean')

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
