import os
import random
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def inverse_transform_sampling(data, n_bins = 40, n_samples = 1000):
    ''' calculate inversed cdf, for sampling by possibility
    '''
    import scipy.interpolate as interpolate
    hist, bin_edges = np.histogram(data, bins=n_bins, density=True)
    cum_values = np.zeros(bin_edges.shape)
    cum_values[1:] = np.cumsum(hist*np.diff(bin_edges))
    inv_cdf = interpolate.interp1d(cum_values, bin_edges)
    r = np.random.rand(n_samples)
    data = inv_cdf(r)
    ndata = [int(x) for x in data]
    return ndata


def sample_based_on_distribution(data, count):
    ''' use data to calculate a inversed cdf, and sample `count` data from such distribution
    '''
    return inverse_transform_sampling(data, n_samples=count)


def data_validation(data, cdata):
    ''' convert sampled data to valid configuration, e.g.,: kernel size = 1, 3, 5, 7

    @params:
    data: the origin data value.
    cdata: valid configuration value.
    '''
    newlist = []
    for da in cdata:
        value = [abs(da - x) for x in data]
        newlist.append(value)

    newlist = list(np.asarray(newlist).T)    
    cda = [list(d).index(min(d)) for d in newlist]
    redata = [cdata[x] for x in cda]
    return redata


def sampling_conv(count : int, read_conv_zoo : callable):
    ''' 
    Sampling configs for conv kernels based on conv_zoo, which contains configuration values from existing model zoo for conv kernel. 
    The values are stored in prior_config_lib/conv.csv.
    Returned params include: (hw, cin, cout, kernel_size, strides)
    '''
    hws, cins, couts, kernel_sizes, strides = read_conv_zoo()
    new_cins = sample_based_on_distribution(cins, count)
    new_couts = sample_based_on_distribution(couts, count)

    # 70% of sampled data are from prior distribution
    count1 = int(count * 0.7)
    new_hws = sample_based_on_distribution(hws, count1)
    new_kernel_sizes = sample_based_on_distribution(kernel_sizes, count1)
    new_strides = sample_based_on_distribution(strides, count1)

    new_kernel_sizes = data_validation(new_kernel_sizes, [1, 3, 5, 7])
    new_strides = data_validation(new_strides, [1, 2, 4])
    new_hws = data_validation(new_hws, [1, 3, 7, 8, 13, 14, 27, 28, 32, 56, 112, 224])
    
    # since conv is the largest and most-challenging kernel, we add some frequently used configuration values
    new_hws.extend([112] * int((count - count1) * 0.2) + [56] * int((count - count1) * 0.4) + [28] * int((count - count1) * 0.4)) # frequent settings
    new_kernel_sizes.extend([5] * int((count - count1) * 0.4) + [7] * int((count - count1) * 0.6)) # frequent settings
    new_strides.extend([2] * int((count - count1) * 0.4) + [1] * int((count - count1) * 0.6)) # frequent settings
    random.shuffle(new_hws)
    random.shuffle(new_strides)
    random.shuffle(new_kernel_sizes)

    ncfgs = []
    for hw, cin, cout, kernel_size, stride in zip(new_hws, new_cins, new_couts, new_kernel_sizes, new_strides):
        c = {
            'HW': hw,
            'CIN': cin,
            'COUT': cout,
            'KERNEL_SIZE': kernel_size,
            'STRIDES': stride,
        }
        ncfgs.append(c)
    return ncfgs


def sampling_conv_random(count):
    ''' sampling configs for conv kernels based on random
    Returned params include: (hw, cin, cout, kernel_size, strides)
    '''
    hws = [1, 7, 8, 13, 14, 27, 28, 32, 56, 112, 224]
    kernel_sizes = [1, 3, 5, 7]
    strides = [1, 2, 4]
    
    cins = list(range(3, 2160))
    couts = list(range(16, 2048))
    new_hws = random.sample(hws * int(count / len(hws)) * 10, count)
    new_kernel_sizes = random.sample(kernel_sizes * int(count / len(kernel_sizes) * 10), count)
    new_strides = random.sample(strides * int(count / len(strides) * 10), count)
    new_cins = random.sample(cins * 10, count)
    new_couts = random.sample(couts * 18, count)
    random.shuffle(new_cins)
    random.shuffle(new_couts)

    ncfgs = []
    for hw, cin, cout, kernel_size, stride in zip(new_hws, new_cins, new_couts, new_kernel_sizes, new_strides):
        c = {
            'HW': hw,
            'CIN': cin,
            'COUT': cout,
            'KERNEL_SIZE': kernel_size,
            'STRIDES': stride,
        }
        ncfgs.append(c)
    return ncfgs


def sample_in_range(mind, maxd, sample_num):
    '''sample #sample_num data from a range [mind, maxd)
    '''
    # if the sample_num is bigger than sample population, we only keep the number of population to avoid repetition
    if maxd - mind <= sample_num:
        data = list(range(mind, maxd))
        random.shuffle(data)
        return data
    else:
        return random.sample(range(mind, maxd), sample_num)


def sample_cin_cout(cin, cout, sample_num): 
    '''fine-grained sample #sample_num data in the cin and cout dimensions, respectively
    '''
    cins = sample_in_range(int(cin * 0.5), int(cin * 1.2), sample_num)
    couts = sample_in_range(int(cout * 0.5), int(cout * 1.2), sample_num)
    l = min(len(cins), len(couts)) # align the length of cins and couts
    cins, couts = cins[:l], couts[:l]
    return cins, couts


def finegrained_sampling_conv(cfgs, count):
    ''' 
    Sampling configs for conv kernels
    Returned params include: (hw, cin, cout, kernel_size, strides)
    '''
    ncfgs = []
    for cfg in cfgs:
        cins, couts = sample_cin_cout(cfg['CIN'], cfg['COUT'], count)
        for cin, cout in zip(cins, couts):
            c = {
                'HW': cfg['HW'],
                'CIN': cin,
                'COUT': cout,
                'KERNEL_SIZE': cfg['KERNEL_SIZE'],
                'STRIDES': cfg['STRIDES'],
            }
            ncfgs.append(c)
    return ncfgs


def read_conv_zoo():
    filename = os.path.join(BASE_DIR, 'data/conv.csv')
    conv_df = pd.read_csv(filename)
    hws = conv_df["input_h"]
    cins = conv_df["cin"]
    couts = conv_df["cout"]
    ks = conv_df["ks"]
    strides = conv_df["stride"]
    return hws, cins, couts, ks, strides