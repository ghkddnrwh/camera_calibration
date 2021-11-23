from numpy import array, cross, dtype
from numpy.linalg import solve, norm

import numpy as np

def cal_shortest_point(point1, point2, point3, point4):
    vec1 = point2 - point1
    vec2 = point4 - point3
    vec3 = np.cross(vec1, vec2)

    co_mat = np.zeros((3, 3), dtype = np.float64)
    const_mat = np.zeros((3, 1), dtype = np.float64)

    co_mat[0, 0] = vec1[0]
    co_mat[0, 1] = - vec2[0]
    co_mat[0, 2] = vec3[0]

    co_mat[1, 0] = vec1[1]
    co_mat[1, 1] = - vec2[1]
    co_mat[1, 2] = vec3[1]

    co_mat[2, 0] = vec1[2]
    co_mat[2, 1] = - vec2[2]
    co_mat[2, 2] = vec3[2]

    const_mat[0, 0] = point3[0] - point1[0]
    const_mat[1, 0] = point3[1] - point1[1]
    const_mat[2, 0] = point3[2] - point1[2]

    sol = np.dot(np.linalg.inv(co_mat), const_mat)

    target_point1 = point1 + vec1 * sol[0]
    target_point2 = point3 + vec2 * sol[1]

    print(target_point1, target_point2)
    
    return (target_point1+ target_point2)/2

X0 = array([1, 1, 1])
X1 = array([1, 0, 0])

Y0 = array([0, 0, 0])
Y1 = array([0, 0, 699])

V1 = (X1 - X0)
V2 = (Y1 - Y0)
V3 = cross(V1, V2)

a = np.zeros((3, 3), dtype = np.float64)
a[0, 0] = V1[0]
a[0, 1] = - V2[0]
a[0, 2] = V3[0]

a[1, 0] = V1[1]
a[1, 1] = - V2[1]
a[1, 2] = V3[1]

a[2, 0] = V1[2]
a[2, 1] = - V2[2]
a[2, 2] = V3[2]

b = np.zeros((3, 1), dtype = np.float64)
b[0, 0] = Y0[0] - X0[0]
b[1, 0] = Y0[1] - X0[1]
b[2, 0] = Y0[2] - X0[2]

sol = np.dot(np.linalg.inv(a), b)
print(sol)

target_point1 = X0 + V1 * sol[0]
target_point2 = Y0 + V2 * sol[1]

print(target_point1, target_point2)


print(cal_shortest_point(X0, X1, Y0, Y1))

