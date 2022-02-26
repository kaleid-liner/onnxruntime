from distutils.command.config import config
from email.policy import default
import json
import argparse


def generate_configs_from_reference(reference_MKNs, resample = 0):
    configs = [{'M' : x[0], 'K' : x[1], 'N' : x[2]} for x in reference_MKNs if x]
    return configs


def main(args):
    # read reference
    with open(args.reference) as f:
        gemm_infos = json.load(f)
    MKNs = [x['MKN'] for x in gemm_infos]
    # generate configs
    configs = generate_configs_from_reference(MKNs)
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
