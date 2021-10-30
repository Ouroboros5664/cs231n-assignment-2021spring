import numpy as np

a = np.matrix(((1),(2),(1),(1)))
a = a.T
b = np.matrix(((2,0,3,0)))

print(np.dot(a,b))