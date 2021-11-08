import math
import sys
import os
import pip
import cv2
import argparse
import numpy as np
import sympy


print(cv2.__version__)

MULTIPLE = 100000000000000

H = 8000
M = 1000

def MouseLeftClick(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        global index
        corner_points[index, 0] = x
        corner_points[index, 1] = y
        index = index + 1
        print("points : ", corner_points)

        print("clicked point : ", x, y)

def fixToMidPoint(_corner_points, _center_point):
    _corner_points[:, 0] = _corner_points[:, 0] - _center_point[0]
    _corner_points[:, 1] = -_corner_points[:, 1] + _center_point[1]

def extensionRatio(_corner_points, _mid_points):
    x0_x1 = (_corner_points[0, 0] - _mid_points[0, 0]) / (_mid_points[0, 0] - _corner_points[1, 0])
    x0_x3 = (_corner_points[0, 0] - _mid_points[3, 0]) / (_mid_points[3, 0] - _corner_points[3, 0])

    return x0_x1, x0_x3

def calDistance(_corner_points, ratio, h, m):
    x0_x1, x0_x3 = ratio
    x0 = m / math.sqrt((_corner_points[0, 0] - _corner_points[1, 0] * x0_x1)**2 + (_corner_points[0, 1] - _corner_points[1, 1] * x0_x1)**2 + 1000**2 * (1 - x0_x1)**2)
    x1 = x0 * x0_x1
    x3 = x0 * x0_x3

    return x0, x1, x3

def calPlaneEq(_corner_points, distance, h):
    x0, x1, x3 = distance
    value = np.zeros((3, 3))
    value[0, 0] = _corner_points[0, 0] * x0
    value[0, 1] = _corner_points[0, 1] * x0
    value[0, 2] = h * x0

    value[1, 0] = _corner_points[1, 0] * x1
    value[1, 1] = _corner_points[1, 1] * x1
    value[1, 2] = h * x0

    value[2, 0] = _corner_points[3, 0] * x0
    value[2, 1] = _corner_points[3, 1] * x0
    value[2, 2] = h * x0

    return np.linalg.inv(value).dot(np.ones(3) * MULTIPLE)

def calposition(point, plane_const):
    x = MULTIPLE / (point.dot(plane_const))

    return x * point

def pointDistance(point1, point2):
    print(point1)
    print(point2)
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2 + (point1[2] - point2[2])**2)

corner_points = np.zeros((4,3))
mid_points = np.zeros((4,3))
center_point = np.zeros(3)
points = np.zeros((3, 3))

index = 0

corner_points[0, 0] = -265.
corner_points[0, 1] = -343.
corner_points[0, 2] = H

corner_points[1, 0] = -418.
corner_points[1, 1] = 185.
corner_points[1, 2] = H

corner_points[2, 0] = 102.
corner_points[2, 1] = 287.
corner_points[2, 2] = H

corner_points[3, 0] = 414.
corner_points[3, 1] = -98.
corner_points[3, 2] = H

mid_points[0, 0] = -356.
mid_points[0, 1] = -28.
mid_points[0, 1] = H

mid_points[1, 0] = -135.
mid_points[1, 1] = 239.
mid_points[1, 1] = H

mid_points[2, 0] = 234.
mid_points[2, 1] = 124.
mid_points[2, 1] = H

mid_points[3, 0] = 113.
mid_points[3, 1] = -207.
mid_points[3, 1] = H

points[0, 0] = -270
points[0, 1] = -16
points[0, 2] = H

points[1, 0] = -70
points[1, 1] = 85
points[1, 2] = H

points[2, 0] = 197
points[2, 1] = 89
points[2, 2] = H

center_point[0] = center_point[1] = 0

# img = cv2.imread('picture/cam7.jpg')

# center_point[0] = img.shape[1] / 2
# center_point[1] = img.shape[0] / 2

# print("mid point : ", center_point)

# cv2.imshow('img', img)
# cv2.setMouseCallback('img', MouseLeftClick)
# cv2.waitKey()

# fixToMidPoint(corner_points, center_point)

# print("corner point : ", corner_points)

# x0 = sympy.Symbol('x0')
# x1 = sympy.Symbol('x1')
# x2 = sympy.Symbol('x2')
# x3 = sympy.Symbol('x3')


# equation0 = x1 - (corner_points[0, 0] - mid_points[0, 0]) / (mid_points[0, 0] - corner_points[1, 0]) * x0
# equation1 = x2 - (corner_points[1, 0] - mid_points[1, 0]) / (mid_points[1, 0] - corner_points[2, 0]) * x1
# equation2 = x3 - (corner_points[0, 0] - mid_points[3, 0]) / (mid_points[3, 0] - corner_points[3, 0]) * x0
# equation3 = (corner_points[0, 0] * x0 - corner_points[1, 0] * x1)**2 + (corner_points[0, 1] * x0 - corner_points[1, 1] * x1)**2 + (H * x0 - H * x1)**2 - M**2

# Result = sympy.solve(equation0, equation1, equation2, equation3)
# print(Result)

ratio = extensionRatio(corner_points, mid_points)
print(ratio)
distance = calDistance(corner_points, ratio, H, M)
print(distance)
plane_const = calPlaneEq(corner_points, distance, H)
print(plane_const)

final_point = calposition(points[0], plane_const)

print(pointDistance(final_point, corner_points[0]))