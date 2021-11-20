'''
Date         : 2021-11-13 22:25:00
Author       : ssyze
Description  : 
'''
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

x_index = []
y_index = []

for _ in range(100):
    x = np.random.rand() * 10
    y = np.random.rand() * 10
    x_index.append(x)
    y_index.append(y)

plt.scatter(x_index,y_index)

x_mean = np.mean(x_index)
x_std = np.std(x_index)
y_mean = np.mean(y_index)
y_std = np.std(y_index)

x_index = (x_index - x_mean) / x_std
y_index = (y_index - y_mean) / y_std

plt.scatter(x_index, y_index)
plt.show()
