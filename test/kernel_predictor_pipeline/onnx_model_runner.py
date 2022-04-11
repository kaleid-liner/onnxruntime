import os
from utils import path_base_name
from runner_backend import onnx_model_runner_backend as C

class OnnxModelRunner(object):
    '''
    Run Onnx model.
    '''
    def __init__(self, workspace_dir : str, save_optimized_model : bool = False, 
                 save_gpu_readings : bool = False, save_running_logs : bool = False, 
                 enable_profiling : bool = False) -> None:
        if workspace_dir is None:
            raise RuntimeError('workspace_dir must be specified.')
        self.workspace = workspace_dir
        os.makedirs(self.workspace, exist_ok = True)
        if save_optimized_model:
            self.optimized_model_dir = os.path.join(self.workspace, 'optimized_models')
            os.makedirs(self.optimized_model_dir, exist_ok = True)
        else:
            self.optimized_model_dir = None
        if save_gpu_readings:
            self.gpu_readings_dir = os.path.join(self.workspace, 'gpu_readings')
            os.makedirs(self.gpu_readings_dir, exist_ok = True)
        else:
            self.gpu_readings_dir = None
        if save_running_logs:
            self.running_logs_dir = os.path.join(self.workspace, 'running_logs')
            os.makedirs(self.running_logs_dir, exist_ok = True)
        else:
            self.running_logs_dir = None
        if enable_profiling:
            self.profile_dir = os.path.join(self.workspace, 'profilings')
            os.makedirs(self.profile_dir, exist_ok = True)
        else:
            self.profile_dir = None
    
    def run(self, model_path : str, num_repeat : int = 1, warmup_repeat : int = 0, 
            time_limit : float = 10.0, gpu_id : int = 0, gpu_sampling_interval : float = 0.01):
        if model_path is None or not os.path.exists(model_path) or not os.path.isfile(model_path):
            print('Error: Invalid path to model: ' + model_path)
        base_name = path_base_name(model_path)
        cfg = C.ONNXRunConfig()
        cfg.num_repeat = num_repeat
        cfg.warm_up_repeat = warmup_repeat
        cfg.time_limit = time_limit
        cfg.gpu_id = gpu_id
        cfg.gpu_sampling_interval = gpu_sampling_interval
        if self.optimized_model_dir:
            cfg.optimized_model_save_path = os.path.join(self.optimized_model_dir, base_name + '.onnx')
        if self.gpu_readings_dir:
            cfg.gpu_readings_csv_save_path = os.path.join(self.gpu_readings_dir, base_name + '.csv')
        if self.running_logs_dir:
            cfg.running_logs_save_path = os.path.join(self.running_logs_dir, base_name + '.txt')
        if self.profile_dir:
            cfg.profile_save_path = os.path.join(self.profile_dir, base_name)
        res = C.run_onnx_model(model_path, cfg)
        return {'latency' : res[0], 'energy' : res[1]}
