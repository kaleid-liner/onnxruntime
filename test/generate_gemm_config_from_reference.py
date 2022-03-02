from distutils.command.config import config
from email.policy import default
import json
import argparse
from random import sample
import numpy as np


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


def generate_configs_from_reference(reference_MKNs, resample : int = 0):
    if resample > 0:
        Ms, Ks, Ns = zip(*reference_MKNs)
        Ms = sample_based_on_distribution(Ms, resample)
        Ks = sample_based_on_distribution(Ks, resample)
        Ns = sample_based_on_distribution(Ns, resample)
        configs = [{'M' : M, 'K' : K, 'N' : N} for M, K, N in zip(Ms, Ks, Ns)]
    else:
        configs = [{'M' : x[0], 'K' : x[1], 'N' : x[2]} for x in reference_MKNs]
    return configs


def main(args):
    # read reference
    with open(args.reference) as f:
        gemm_infos = json.load(f)
    MKNs = [x['MKN'] for x in gemm_infos if x['MKN']]
    # generate configs
    configs = generate_configs_from_reference(MKNs, args.resample)
    # save to file
    with open(args.save_path, 'w') as f:
        json.dump(configs, f, indent = 4)


if __name__ == '__main__':

    parser = argparse.ArgumentParser('Generate Gemm kernel config from collected gemm infos.')
    parser.add_argument('reference', type = str, default = None)
    parser.add_argument('--save-path', type = str, default = './experiment_data/gemm_kernel_configs.json')
    parser.add_argument('--resample', type = int, default = 0)
    args = parser.parse_args()
    print(args)
    main(args)
