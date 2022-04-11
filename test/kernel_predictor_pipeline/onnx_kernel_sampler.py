import json
import sampling_conv as sc


class OnnxKernelSampler(object):
    def __init__(self) -> None:
        pass
    
    def initial_sampling(self, reference_infos, num_configs):
        pass
    
    def fine_grained_sampling(self, old_configs, samples_per_config):
        pass


class ConvKernelSampler(OnnxKernelSampler):
    def __init__(self) -> None:
        super().__init__()
 
    def to_nnmeter_format(self, configs):
        ncfgs = []
        for cfg in configs:
            c = {
                'HW' : cfg['image_shape'][0],
                'CIN' : cfg['input_channels'],
                'COUT' : cfg['output_channels'],
                'KERNEL_SIZE' : cfg['kernel_shape'][0],
                'STRIDES' : cfg['strides'][0],
            }
            ncfgs.append(c)
        return ncfgs

    def from_nnmeter_format(self, ncfgs):
        configs = []
        for c in ncfgs:
            conv_config = {
                # flexible params
                'image_shape' : [c['HW'], c['HW']],
                'input_channels' : c['CIN'],
                'kernel_channels' : c['CIN'],
                'output_channels' : c['COUT'],
                # constrained params
                'batch_size' : 1,
                'kernel_shape' : [c['KERNEL_SIZE'], c['KERNEL_SIZE']],
                'pads' : [c['KERNEL_SIZE'] // 2, c['KERNEL_SIZE'] // 2],
                'dilations' : [1, 1],
                'strides' : [c['STRIDES'], c['STRIDES']],
                'group' : 1,
                'has_bias' : True,
            }
            configs.append(conv_config)
        return configs

    def generate_config_from_reference(self, reference_info):
        # input and kernel shapes
        node_inputs = reference_info['input']
        if len(node_inputs) not in [2, 3]:
            print('Error: number of inputs must be 2 or 3, got', len(node_inputs))
            return None
        batch_size, input_channels, *image_shape = node_inputs[0]
        output_channels, kernel_channels, *kernel_shape = node_inputs[1]
        has_bias = True if len(node_inputs) == 3 else False
        # attributes
        attributes = reference_info['attribute']
        # correctiveness check
        if input_channels != kernel_channels * attributes['group']:
            print('Error: imcompatible input and kernel channels.')
            return None
        if has_bias and node_inputs[2][0] != output_channels:
            print('Warning: length of bias not equal to output_channels in reference.')
        if attributes['kernel_shape'] != kernel_shape:
            print('Error: different kernel_shape present in attributes and inputs.')
            return None
        # construct config
        conv_config = {
            # flexible params
            'image_shape' : image_shape,
            'input_channels' : input_channels,
            'kernel_channels' : kernel_channels,
            'output_channels' : output_channels,
            # constrained params
            'batch_size' : batch_size,
            'kernel_shape' : kernel_shape,
            'pads' : attributes['pads'][:2],
            'dilations' : attributes['dilations'],
            'strides' : attributes['strides'],
            'group' : attributes['group'],
            'has_bias' : has_bias,
        }
        return conv_config

    def resample_configs_nnmeter(self, configs_old, num_configs : int = 0):
        def helper():
            hws = [cfg['image_shape'][0] for cfg in configs_old]
            cins = [cfg['input_channels'] for cfg in configs_old]
            couts = [cfg['output_channels'] for cfg in configs_old]
            ks = [cfg['kernel_shape'][0] for cfg in configs_old]
            strides = [cfg['strides'][0] for cfg in configs_old]
            return hws, cins, couts, ks, strides
        ncfgs = sc.sampling_conv(num_configs, helper)
        return self.from_nnmeter_format(ncfgs)
    
    def initial_sampling(self, reference_infos, num_configs):
        configs = [self.generate_config_from_reference(conv_info) for conv_info in reference_infos]
        configs = self.resample_configs_nnmeter(configs, num_configs)
        return configs

    def fine_grained_sampling(self, old_configs, samples_per_config):
        ncfgs = self.to_nnmeter_format(old_configs)
        ncfgs = sc.finegrained_sampling_conv(ncfgs, samples_per_config)
        return self.from_nnmeter_format(ncfgs)
