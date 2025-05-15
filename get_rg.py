import sys
import numpy as np


def get_rg():
    rg = []   
    
    for i in range(28,43):
        aux = str(2) + "." + str(i) + "_gyration_0_vaccum.txt"
        data = np.loadtxt(aux)
        rg.append(np.mean(data[5:,1]))
    rg = [float(value) for value in list(rg)]
    return rg

#
#print(get_rg())

