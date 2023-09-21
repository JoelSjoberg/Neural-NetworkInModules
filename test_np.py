import numpy as np

a = np.random.random_sample((100, 10))

b = np.random.random_sample((30, 10))

np.subtract(a.T, b.T)

