import numpy as np

a = np.array(((-1,2,3),(1,2,3)))

b = np.array(((3,2,1),(3,2,1)))

print(np.maximum(a,0)*b)