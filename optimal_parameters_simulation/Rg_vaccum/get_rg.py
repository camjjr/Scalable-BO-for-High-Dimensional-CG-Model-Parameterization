import sys
import numpy as np


def get_mean(file_name):
    data = np.loadtxt(file_name)
    return np.mean(data[500:,1]),np.std(data[500:,1])

for i in range(28,43):
    values_mean = []
    values_std = []
    aux = str(2) + "." + str(i) + "_gyration_0_vaccum.txt"
    aux = get_mean(aux)
    values_mean.append(aux[0])
    values_std.append(aux[1])


    values_mean = np.array(values_mean)

    print(np.mean(values_mean)) 

#print(get_mean(sys.argv[1]))
