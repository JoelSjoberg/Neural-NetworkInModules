import numpy as np


a = np.random.random_sample((1, 100))
b = np.ones((128, 100))

print((b*a).shape)
print((b*a))