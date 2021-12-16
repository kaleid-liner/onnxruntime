import os
import sys
import csv
import argparse

import numpy as np
import matplotlib.pyplot as plt

ALL_OPS = {'Gather', 'Split', 'Add', 'MaxPool', 'Conv', 'LayerNormalization', 'Concat', 'FusedConv', 'Gemm', 'SkipLayerNormalization', 
            'FusedMatMul', 'Relu', 'ReduceMean', 'AveragePool', 'Transpose', 'Softmax', 'BiasGelu', 'MatMul', 'Pad', 'GlobalAveragePool', 
            'BatchNormalization', 'Div', 'HardSigmoid', 'Reshape', 'Clip', 'Unsqueeze', 'Mul', 'Flatten'}



if __name__ == '__main__':

    parser = argparse.ArgumentParser('Plot node profiling results.')
    