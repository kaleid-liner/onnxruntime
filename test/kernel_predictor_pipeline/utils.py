import os
import numpy as np
from sklearn.metrics import mean_squared_error


def file_base_name(file_name):
    if '.' in file_name:
        separator_index = file_name.index('.')
        base_name = file_name[:separator_index]
        return base_name
    else:
        return file_name


def path_base_name(path):
    file_name = os.path.basename(path)
    return file_base_name(file_name)


def get_accuracy(y_pred, y_true, threshold = 0.01):
    a = (y_true - y_pred) / y_true
    b = np.where(abs(a) <= threshold)
    return len(b[0]) / len(y_true)


def get_metrics(y_pred, y_true):
    rmspe = (np.sqrt(np.mean(np.square((y_true - y_pred) / y_true)))) * 100
    rmse = np.sqrt(mean_squared_error(y_pred, y_true))
    error = rmse / np.mean(y_true)
    acc5 = get_accuracy(y_pred, y_true, threshold=0.05)
    acc10 = get_accuracy(y_pred, y_true, threshold=0.10)
    acc15 = get_accuracy(y_pred, y_true, threshold=0.15)
    print(f"rmse: {rmse:.4f}; rmspe: {rmspe:.4f}; error: {error:.4f}; 5% accuracy: {acc5:.4f}; 10% accuracy: {acc10:.4f}; 15% accuracy: {acc15:.4f}.")


def get_large_error_indices(y_pred, y_true, threshold = 0.01):
    a = (y_true - y_pred) / y_true
    b = np.where(abs(a) > threshold)
    return b[0]

