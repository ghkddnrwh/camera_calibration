import sympy
import numpy as np

x = np.array([1 , 0, 0])
y = np.array([0, 1, 0])
print(np.cross(x, y))

r = np.array([[1, 0 ,0], [0, 1, 0],[0, 0, 1]], dtype=np.float64)
a = np.array([[np.sqrt(3)/2, -0.5, -2], [0.5, np.sqrt(3)/2, 2], [0, 0, 1]], dtype=np.float64)

print(r.dot(np.linalg.inv(a)))
print(np.linalg.inv(a))
print(np.dot(np.linalg.inv(a), np.array([[0], [0], [1]])))