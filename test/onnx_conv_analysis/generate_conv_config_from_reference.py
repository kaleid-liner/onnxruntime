from email.policy import default
import json
import argparse
import random
import numpy as np
import sampling_conv as sc

def inverse_transform_sampling(data, n_bins = 40, n_samples = 1000):
    ''' calculate inversed cdf, for sampling by possibility
    '''
    import scipy.interpolate as interpolate
    hist, bin_edges = np.histogram(data, bins = n_bins, density = True)
    cum_values = np.zeros(bin_edges.shape)
    cum_values[1:] = np.cumsum(hist * np.diff(bin_edges))
    inv_cdf = interpolate.interp1d(cum_values, bin_edges)
    r = np.random.rand(n_samples)
    data = inv_cdf(r)
    ndata = [int(x) for x in data]
    return ndata


def sample_based_on_distribution(data, count):
    ''' use data to calculate a inversed cdf, and sample `count` data from such distribution
    '''
    return inverse_transform_sampling(data, n_samples=count)


def generate_config_from_reference(reference_info):
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


def remove_repeat_items(old_list):
    new_list = []
    for item in old_list:
        if item not in new_list:
            new_list.append(item)
    return new_list


def resample_configs_onnx(configs_old, resample : int = 0):
    '''
    resampled params: image_shape, input_channels, kernel_channels, output_channels, group
                      input_channels = kernel_channels * group
    '''
    if resample == 0:
        return configs_old
    # normalize kernel channels and group
    for cfg in configs_old:
        cfg['kernel_channels'] = cfg['kernel_channels'] * cfg['group']
        cfg['group'] = 1
    # constrained params
    batch_sizes = remove_repeat_items([cfg['batch_size'] for cfg in configs_old])
    kernel_shapes = remove_repeat_items([cfg['kernel_shape'] for cfg in configs_old])
    pads = remove_repeat_items([cfg['pads'] for cfg in configs_old])
    dilations = remove_repeat_items([cfg['dilations'] for cfg in configs_old])
    strides = remove_repeat_items([cfg['strides'] for cfg in configs_old])
    has_biases = remove_repeat_items([cfg['has_bias'] for cfg in configs_old])
    # flexible params
    image_sizes = [cfg['image_shape'][0] for cfg in configs_old]
    kernel_channels = [cfg['kernel_channels'] for cfg in configs_old]
    groups = [cfg['group'] for cfg in configs_old]
    output_channels = [cfg['output_channels'] for cfg in configs_old]
    # resample
    image_sizes_new = sample_based_on_distribution(image_sizes, resample)
    kernel_channels_new = sample_based_on_distribution(kernel_channels, resample)
    groups_new = sample_based_on_distribution(groups, resample)
    output_channels_new = sample_based_on_distribution(output_channels, resample)
    # construct configs
    configs_new = []
    def check_kernel_shape(kernel_shape, image_shape) -> bool:
        if kernel_shape[0] > image_shape[0] or kernel_shape[1] > image_shape[1]:
            return False
        else:
            return True
    for H_W, kC, G, N in zip(image_sizes_new, kernel_channels_new, groups_new, output_channels_new):
        conv_config = {
            # flexible params
            'image_shape' : [H_W, H_W],
            'input_channels' : kC * G,
            'kernel_channels' : kC,
            'output_channels' : N,
            # constrained params
            'batch_size' : random.choice(batch_sizes),
            'kernel_shape' : random.choice(kernel_shapes),
            'pads' : random.choice(pads),
            'dilations' : random.choice(dilations),
            'strides' : random.choice(strides),
            'group' : G,
            'has_bias' : random.choice(has_biases),
        }
        # check params
        while not check_kernel_shape(conv_config['kernel_shape'], conv_config['image_shape']):
            conv_config['kernel_shape'] = random.choice(kernel_shapes)
        if conv_config['output_channels'] % conv_config['group'] != 0:
            conv_config['output_channels'] = conv_config['output_channels'] // conv_config['group'] * conv_config['group']
        configs_new.append(conv_config)
    return configs_new


def resample_configs_nnmeter(configs_old, resample : int = 0, use_builtin_data : bool = False):
    if use_builtin_data:
        def helper():
            return sc.read_conv_zoo()
    else:
        def helper():
            hws = [cfg['image_shape'][0] for cfg in configs_old]
            cins = [cfg['input_channels'] for cfg in configs_old]
            couts = [cfg['output_channels'] for cfg in configs_old]
            ks = [cfg['kernel_shape'][0] for cfg in configs_old]
            strides = [cfg['strides'][0] for cfg in configs_old]
            return hws, cins, couts, ks, strides
    ncfgs = sc.sampling_conv(resample, helper)
    configs_new = []
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
        configs_new.append(conv_config)
    return configs_new


def main(args):
    # read reference
    with open(args.reference) as f:
        conv_infos = json.load(f)
    # generate configs
    configs = [generate_config_from_reference(conv_info) for conv_info in conv_infos]
    if args.resample:
        use_builtin_data = False
        configs = resample_configs_nnmeter(configs, args.resample, use_builtin_data)
    # save to file
    with open(args.save_path, 'w') as f:
        json.dump(configs, f, indent = 4)


if __name__ == '__main__':

    parser = argparse.ArgumentParser('Generate Gemm kernel config from collected gemm infos.')
    parser.add_argument('reference', type = str, default = None)
    parser.add_argument('--save-path', type = str, default = './experiment_data/conv_kernel_infos/conv_kernel_configs.json')
    parser.add_argument('--resample', type = int, default = 0)
    # parser.add_argument('--builtin-data', type = str, default = None)
    args = parser.parse_args()
    print(args)
    main(args)
