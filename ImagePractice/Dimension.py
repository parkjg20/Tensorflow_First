import numpy as np

d1 = np.array([1, 2, 3])
d2 = np.array([[1, 2, 3 ], [4, 5, 6], [7, 8, 9]])
d3 = np.array([[d1, d1, d1], [d1, d1, d1], [d1, d1, d1]])
d4 = np.array([[d1]])

print(d1.shape)
print(d2.shape)
print(d3.shape)
print(d4.shape)