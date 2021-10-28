import numpy as np

x = np.array([[1,2],[3,4]])

tmp = (x**2).sum(axis=1, keepdims=True).T

y = np.array([[1,2],[3,4]])

print(tmp.shape, tmp)

print(tmp + y)