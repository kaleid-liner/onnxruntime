import os
import sys
import json
import pickle

import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

from onnx_kernel_sampler import ConvKernelSampler
from onnx_kernel_constructor import ConvKernelConstructor
from onnx_model_runner import OnnxModelRunner
from utils import get_accuracy, get_metrics, get_large_error_indices


class ConvKernelPredictor(object):
    def __init__(self, metric : str = 'energy'):
        if not metric in ['latency', 'energy']:
            raise RuntimeError('Invalid metric, only latency and energy is supported.')
        self.metric = metric
        self.model = None
    
    def calculate_output_shape(self, cfg):
        # const int output_h = (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
        # const int output_w = (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
        height, width = cfg['image_shape']
        kernel_h, kernel_w = cfg['kernel_shape']
        pad_h, pad_w = cfg['pads']
        dilation_h, dilation_w = cfg['dilations']
        stride_h, stride_w = cfg['strides']
        output_h = (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) // stride_h + 1
        output_w = (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) // stride_w + 1
        output_shape = [cfg['batch_size'], cfg['output_channels'], output_h, output_w]
        return output_shape
    
    def calculate_gmacs(self, cfg):
        def reduce_list_prod(x):
            from functools import reduce
            return reduce(lambda a, b : a * b, x)
        output_shape = self.calculate_output_shape(cfg)
        gmacs = reduce_list_prod(output_shape) * reduce_list_prod(cfg['kernel_shape']) * cfg['input_channels']
        return gmacs * 1e-9
    
    def construct_feature_vec(self, cfg):
        feature_vec = [
            cfg['image_shape'][0],
            cfg['input_channels'],
            cfg['output_channels'],
            cfg['kernel_shape'][0],
            cfg['strides'][0],
            self.calculate_gmacs(cfg),
        ]
        return feature_vec

    def preprocess_data(self, profiling_data, split : bool = False):
        features = [self.construct_feature_vec(profiling_data[k]['config']) for k in profiling_data.keys()]
        y_true = [profiling_data[k][self.metric] for k in profiling_data.keys()]
        raw_cfgs = [profiling_data[k]['config'] for k in profiling_data.keys()]
        if split:
            features_train, features_val, y_train, y_val, cfgs_train, cfgs_val = train_test_split(features, y_true, raw_cfgs)
            train_split = {
                'feature' : features_train,
                self.metric : y_train,
                'raw_cfg' : cfgs_train,
            }
            val_split = {
                'feature' : features_val,
                self.metric : y_val,
                'raw_cfg' : cfgs_val,
            }
            return train_split, val_split
        else:
            train_data = {
                'feature' : features,
                self.metric : y_true,
                'raw_cfg' : raw_cfgs,
            }
            return train_data
    
    def train_predictor(self, profiling_data):
        train_data, val_data = self.preprocess_data(profiling_data, True)
        # train
        n_features = len(train_data['feature'][0])
        model = RandomForestRegressor(
            max_depth=50,
            n_estimators=370,
            min_samples_leaf=1,
            min_samples_split=2,
            max_features=n_features,
            oob_score=True,
            random_state=10
        )
        model.fit(train_data['feature'], train_data[self.metric])
        # val
        predicts = model.predict(val_data['feature'])
        print(f'{self.metric} Results:')
        get_metrics(predicts, val_data[self.metric])
        self.model = model
    
    def get_model(self):
        return self.model


class ConvKernelPredictorPipeline(object):
    def __init__(self, workspace_dir : str, num_initial_cfgs : int = 10000, 
                 num_finegrained_samples_per_cfg : int = 20, num_adaptive_sampling_iters : int = 3):
        # params
        self.num_initial_cfgs = num_initial_cfgs
        self.num_finegrained_samples_per_cfg = num_finegrained_samples_per_cfg
        self.num_adaptive_sampling_iters = num_adaptive_sampling_iters
        # workspace dirs
        self.workspace = workspace_dir
        os.makedirs(self.workspace, exist_ok = True)
        self.runner_workspace = os.path.join(self.workspace, 'running')
        os.makedirs(self.runner_workspace, exist_ok = True)
        self.config_save_dir = os.path.join(self.workspace, 'configs')
        os.makedirs(self.config_save_dir, exist_ok = True)
        self.conv_kernel_dir = os.path.join(self.workspace, 'conv_kernels')
        os.makedirs(self.conv_kernel_dir, exist_ok = True)
        self.results_dir = os.path.join(self.workspace, 'results')
        os.makedirs(self.results_dir, exist_ok = True)
        # aux classes
        self.conv_sampler = ConvKernelSampler()
        self.conv_constructor = ConvKernelConstructor()
        self.conv_runner = OnnxModelRunner(self.runner_workspace, False, True, True, False)
    
    def profile_kernels(self, kernels, name = 'profile'):
        profiling_results = {}
        count = 0
        total_num_kernels = len(kernels)
        for k in kernels.keys():
            kernel_path, cfg = kernels[k]
            count += 1
            print('[{}] Profiling {}/{} kernels.'.format(name, count, total_num_kernels))
            try:
                res = self.conv_runner.run(kernel_path, 10000, 3, 10.0)
                profiling_results[k] = {
                    'kernel_path' : kernel_path,
                    'config' : cfg,
                    'latency' : res['latency'],
                    'energy' : res['energy'],
                }
            except Exception as e:
                print('Error:', e)
        return profiling_results
    
    def build_predictor(self, reference_info):
        # initial sampling
        print('Initial sampling...')
        configs = self.conv_sampler.initial_sampling(reference_info, self.num_initial_cfgs)
        with open(os.path.join(self.config_save_dir, 'initial_configs.json'), 'w') as f:
            json.dump(configs, f, indent = 4)
        print('Initial sampling finished.')
        print('Construct initial kernels...')
        kernels = self.conv_constructor.generate_kernels(self.conv_kernel_dir, configs)
        with open(os.path.join(self.results_dir, 'initial_kernels_info.bin'), 'wb') as f:
            pickle.dump(kernels, f)
        print('Construct initial kernels finished.')
        # initial profiling
        print('Initial profiling...')
        profiling_results = self.profile_kernels(kernels, 'initial profiling')
        with open(os.path.join(self.results_dir, 'initial_profiling_results.bin'), 'wb') as f:
            pickle.dump(profiling_results, f)
        print('Initial profiling finished.')
        # build initial predictor
        print('Build initial predictor...')
        metrics = ['latency', 'energy']
        predictors = {}  # used for saving models, across adaptive sampling
        train_val_data = {}  # used for saving data, across adaptive sampling
        for metric in metrics:
            p = ConvKernelPredictor(metric)
            p.train_predictor(profiling_results)
            predictors[metric] = p
            train_val_data[metric] = profiling_results.copy()
        with open(os.path.join(self.results_dir, 'initial_predictor.bin'), 'wb') as f:
            pickle.dump(predictors, f)
        print('Build initial predictor finished.')
        # adaptive sampling
        print('Start adaptive sampling.')
        for iter in range(self.num_adaptive_sampling_iters):
            # get large error items, consider all data
            large_error_cfgs = {}
            for metric in metrics:
                p = predictors[metric]
                total_data = p.preprocess_data(train_val_data[metric])
                large_error_indices = get_large_error_indices(p.get_model().predict(total_data['feature']), total_data[metric], 0.15)
                cfgs = [total_data['raw_cfg'][i] for i in large_error_indices]
                large_error_cfgs[metric] = cfgs
            with open(os.path.join(self.results_dir, 'large_error_cfgs_iter_{}.json'.format(iter)), 'w') as f:
                json.dump(large_error_cfgs, f)
            # sample fine-grained cfgs
            fine_grained_cfgs = {}
            for metric in metrics:
                fine_grained_cfgs[metric] = self.conv_sampler.fine_grained_sampling(large_error_cfgs[metric], self.num_finegrained_samples_per_cfg)
            with open(os.path.join(self.results_dir, 'fine_grained_cfgs_iter_{}.json'.format(iter)), 'w') as f:
                json.dump(fine_grained_cfgs, f)
            # generate kernels
            fine_grained_kernels = {}
            for metric in metrics:
                fine_grained_kernels[metric] = self.conv_constructor.generate_kernels(self.conv_kernel_dir, fine_grained_cfgs[metric])
            with open(os.path.join(self.results_dir, 'fine_grained_kernel_infos_iter_{}.bin'.format(iter)), 'wb') as f:
                pickle.dump(fine_grained_kernels, f)
            # profiling
            fine_grained_profiling = {}
            for metric in metrics:
                fine_grained_profiling[metric] = self.profile_kernels(fine_grained_kernels[metric], 'finegrained_iter_{}'.format(iter))
            with open(os.path.join(self.results_dir, 'fine_grained_profiling_results_iter_{}.bin'.format(iter)), 'wb') as f:
                pickle.dump(fine_grained_profiling, f)
            # build predictor
            for metric in metrics:
                train_val_data[metric] = {**train_val_data[metric], **fine_grained_profiling[metric]}
                p = ConvKernelPredictor(metric)
                p.train_predictor(train_val_data[metric])
                predictors[metric] = p
            with open(os.path.join(self.results_dir, 'fine_grained_predictors_iter_{}.bin'.format(iter)), 'wb') as f:
                pickle.dump(predictors, f)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('usage:', sys.argv[0], 'onnx_model_conv_infos.json')
        sys.exit(0)
    reference_info_path = sys.argv[1]
    with open(reference_info_path) as f:
        reference_info = json.load(f)
    pipeline = ConvKernelPredictorPipeline('./workspace')
    pipeline.build_predictor(reference_info)
