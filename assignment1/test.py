import numpy as np

a = np.array([[1,2],[3,4]])

b = np.zeros(a.shape[0])

b = np.argmax(a, axis=1)

print(b, np.argmax(a))