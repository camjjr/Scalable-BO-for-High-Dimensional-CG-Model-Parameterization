import sys
import numpy as np


def get_mean(file_name):
    data = np.loadtxt(file_name)
    return np.mean(data[:,1])
