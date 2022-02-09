from email.policy import default
import os
import sys
import argparse
import onnxruntime as ort


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
    if args.model_path is None:
        print('Error: --model-path must be specified.')
        return
    os.makedirs(args.output_dir, exist_ok = True)
    base_name = path_base_name(args.model_path)

    sess_options = ort.SessionOptions()
    sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    sess_options.execution_order = ort.ExecutionOrder.DEFAULT
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.optimized_model_filepath = os.path.join(args.output_dir, 'optimized_' + base_name + '.onnx')

    EP_list = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    sess = ort.InferenceSession(args.model_path, sess_options, EP_list)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Convert onnx model to ORT-optimized model')
    parser.add_argument('--model-path', type = str, default = None)
    parser.add_argument('--model-dir', type = str, default = None)
    parser.add_argument('--output-dir', type = str, default = './experiment_data/onnx_models/ort_optimized/')
    args = parser.parse_args()
    print(args)

    if args.model_path:
        main(args)
    elif args.model_dir:
        models = os.listdir(args.model_dir)
        models = [i for i in models if i.split('.')[-1] == 'onnx']
        for model in models:
            print('optimizing model:', model)
            args.model_path = os.path.join(args.model_dir, model)
            try:
                main(args)
            except Exception as e:
                print('Error:', e)
    else:
        print('Error: at least one of --model-path and --model-dir must be specified.')
