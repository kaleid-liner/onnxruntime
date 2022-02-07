import csv
import torch
import torchvision.models as models
from deepspeed.profiling.flops_profiler import get_model_profile


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))


def main(model_name):
    assert(model_name in model_names)
    model = models.__dict__[model_name](pretrained = True)
    macs, params = get_model_profile(model = model, # model
                                     input_res = (1, 3, 224, 224), # input shape or input to the input_constructor
                                     input_constructor = None, # if specified, a constructor taking input_res is used as input to the model
                                     print_profile = False, # prints the model graph with the measured profile attached to each module
                                     detailed = True, # print the detailed profile
                                     module_depth = -1, # depth into the nested modules with -1 being the inner most modules
                                     top_modules = 3, # the number of top modules to print aggregated profile
                                     warm_up = 10, # the number of warm-ups before measuring the time of each module
                                     as_string = False, # print raw numbers (e.g. 1000) or as human-readable strings (e.g. 1k)
                                     output_file = None, # path to the output file. If None, the profiler prints to stdout.
                                     ignore_modules = None) # the list of modules to ignore in the profiling
    return macs, params


if __name__ == '__main__':
    res = []
    for model in model_names:
        try:
            macs, params = main(model)
            res.append({
                    'model' : model,
                    'macs' : macs,
                    'params' : params,
                })
        except Exception as e:
            print('Error:', e)
    f = open('experiment_data/torchvison_macs.csv', 'w')
    f_csv = csv.DictWriter(f, ['model', 'macs', 'params'])
    f_csv.writeheader()
    f_csv.writerows(res)
    f.close()
