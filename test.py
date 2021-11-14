'''
Date         : 2021-11-13 22:25:00
Author       : ssyze
Description  : 
'''
import numpy as np

a = np.array(((1,2,3),(4,5,6)))

b = np.array((1,2,3))

print(a*b, a.shape, b.shape)

for reg in [0, 3.14]:
    print(reg)