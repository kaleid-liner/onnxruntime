import os
import json
import onnx
from onnx import helper, shape_inference, numpy_helper
from onnx import AttributeProto, TensorProto, GraphProto, NodeProto, ValueInfoProto


class OnnxKernelConstructor(object):
    def __init__(self) -> None:
        pass
    
    def construct_kernel(self, cfg):
        pass
    
    def make_kernel_save_name(self, cfg):
        pass
    
    def generate_kernels(self, kernel_save_dir, config_list):
        os.makedirs(kernel_save_dir, exist_ok = True)
        generated_kernels = {}
        for i, cfg in enumerate(config_list):
            if (i + 1) % 100 == 0:
                print('generate {}/{} kernels.'.format(i + 1, len(config_list)))
            try:
                kernel = self.construct_kernel(cfg)
                kernel_save_name = self.make_kernel_save_name(cfg)
                kernel_save_path = os.path.join(kernel_save_dir, kernel_save_name + '.onnx')
                onnx.save(kernel, kernel_save_path)
                onnx.checker.check_model(kernel_save_path)
                generated_kernels[kernel_save_name] = (kernel_save_path, cfg)
            except Exception as e:
                print('Error:', e)
        print('generated {} kernels out of {} configs.'.format(len(generated_kernels.keys()), len(config_list)))
        return generated_kernels


class ConvKernelConstructor(OnnxKernelConstructor):
    def __init__(self) -> None:
        super().__init__()

    def make_attribute_dict(self, cfg):
        attribute = {
            'kernel_shape' : cfg['kernel_shape'],
            'pads' : cfg['pads'] + cfg['pads'],
            'dilations' : cfg['dilations'],
            'strides' : cfg['strides'],
            'group' : cfg['group'],
        }
        return attribute

    def make_input_shape(self, cfg):
        return [cfg['batch_size'], cfg['input_channels']] + cfg['image_shape']

    def make_weight_shapes(self, cfg):
        shape_w = [cfg['output_channels'], cfg['kernel_channels']] + cfg['kernel_shape']
        shape_b = [cfg['output_channels']]
        return shape_w, shape_b

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
    
    def make_kernel_save_name(self, cfg):
        save_name = 'onnx_conv'
        # input shape NCHW
        save_name += '_in_{}_{}_{}_{}'.format(cfg['batch_size'], cfg['input_channels'], cfg['image_shape'][0], cfg['image_shape'][1])
        # kernel shape NHW
        save_name += '_kernel_{}_{}_{}_{}'.format(cfg['output_channels'], cfg['kernel_channels'], cfg['kernel_shape'][0], cfg['kernel_shape'][1])
        # attributes (assume isotropic)
        save_name += '_attr_{}_{}_{}_{}_{}_{}_{}_{}'.format(cfg['pads'][0], cfg['pads'][1], cfg['dilations'][0], cfg['dilations'][1], cfg['strides'][0], cfg['strides'][1], cfg['group'], int(cfg['has_bias']))
        # finished
        # save_name += '.onnx'
        return save_name

    def construct_kernel(self, cfg):
        # make node
        op_type = 'Conv'
        node_input = ['X', 'W', 'B'] if cfg['has_bias'] else ['X', 'W']
        node_output = ['Y']
        node_name = 'conv_node'
        attribute = self.make_attribute_dict(cfg)
        node = helper.make_node(op_type, node_input, node_output, node_name, **attribute)
        # prepare input/weights/output
        input_shape = self.make_input_shape(cfg)
        shape_w, shape_b = self.make_weight_shapes(cfg)
        output_shape = self.calculate_output_shape(cfg)
        X = helper.make_tensor_value_info('X', TensorProto.FLOAT, input_shape)
        W = helper.make_tensor_value_info('W', TensorProto.FLOAT, shape_w)
        if cfg['has_bias']:
            B = helper.make_tensor_value_info('B', TensorProto.FLOAT, shape_b)
        Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, output_shape)
        # make graph
        graph_inputs = [X, W, B] if cfg['has_bias'] else [X, W]
        graph_outputs = [Y]
        graph = helper.make_graph([node], 'graph with single conv node.', graph_inputs, graph_outputs)
        # make kernel
        kernel = helper.make_model(graph)
        return kernel
