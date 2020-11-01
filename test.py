import numpy as np
z = np.array([1,2,3,4,5])
d = [2,3,3]
i = np.where(z==d[0])
print(i)
c = np.array([0,0,0,0,0])
c[i] += 1
print(c)