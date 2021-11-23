import numpy as np
import glob, cv2
import math
from sympy import Symbol, solve

def get_camera_matrix(images):
    criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    objp = np.zeros((6*9, 3), np.float32)
    objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1, 2)

    objpoints = []
    imgpoints = []

    for name in images:
        img = cv2.imread(name)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        print(gray.shape[::-1])

        ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)

        if ret == True:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

            # img = cv2.drawChessboardCorners(img, (9, 6), corners2, ret)
            # cv2.imshow('img', img)
            # cv2.waitKey(0)

    return cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)

def MouseLeftClick(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        global index
        corner_points[index, 0] = x
        corner_points[index, 1] = y
        index = index + 1
        print("points : ", corner_points)

        print("clicked point : ", x, y)

def MouseLeftClick2(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        global index
        mid_points[index, 0] = x
        mid_points[index, 1] = y
        index = index + 1
        print("points : ", mid_points)

        print("clicked point : ", x, y)

def MouseLeftClick3(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        global index
        test_points[index, 0] = x
        test_points[index, 1] = y
        index = index + 1
        print("points : ", test_points)

        print("clicked point : ", x, y)

def fixToMidPoint(_corner_points, _center_point):
    _corner_points[:, 0] = _corner_points[:, 0] - _center_point[0]
    _corner_points[:, 1] = -_corner_points[:, 1] + _center_point[1]

def cal_distance(points, index1, index2):
    point1 = points[index1]
    point2 = points[index2]
    return math.sqrt((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2 + (point1[2]-point2[2])**2)

def cal_distance2(point1, point2):
    return math.sqrt((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2 + (point1[2]-point2[2])**2)

def cal_focus_length(point1, point2, point4):
    A = (point1[0] - point2[0])**2 + (point1[1] - point2[1])**2
    B = (point1[0] - point4[0])**2 + (point1[1] - point4[1])**2
    c = (point1[2] - point2[2])**2
    d = (point1[2] - point4[2])**2

    return math.sqrt((B - A) / (c - d))

def cal_intrinsic_param(point0, point1, point2, point3, point4):
    eq1 = (point0[0] - point1[0])**2 + (point0[1] - point1[1])**2 + (point0[2] - point1[2])**2 - (point0[0] - point3[0])**2 - (point0[1] - point3[1])**2 - (point0[2] - point3[2])**2
    # eq2 = (point0[0] - point1[0])**2 + (point0[1] - point1[1])**2 + (point0[2] - point1[2])**2 - (point1[0] - point2[0])**2 - (point1[1] - point2[1])**2 - (point1[2] - point2[2])**2
    # eq3 = (point0[0] - point1[0])**2 + (point0[1] - point1[1])**2 + (point0[2] - point1[2])**2 - (point3[0] - point4[0])**2 - (point3[1] - point4[1])**2 - (point3[2] - point4[2])**2
    eq4 = (point0[0] - point1[0]) * (point0[0] - point3[0]) + (point0[1] - point1[1]) * (point0[1] - point3[1]) + (point0[2] - point1[2]) * (point0[2] - point3[2])
    return solve(eq1, eq4, Hx, Hy)

cali_images = glob.glob("./picture2/*.jpg")
ret, mtx, dists, rvecs, tvecs = get_camera_matrix(cali_images)

print("mtx : ", mtx)

U = Symbol('U')
V = Symbol('V')
Hx = Symbol('Hx')
Hy = Symbol('Hy')


H = 940.0
H = 918.5032416076951
H = 1
M = 3000
corner_points = np.zeros((4,3))
mid_points = np.zeros((4,3))
object_points = np.zeros((4,3))
mid_object_points = np.zeros((4,3))
plane_points = np.zeros((3, 3))
test_points = np.zeros((4, 3), dtype= np.float64)
index = 0

mound_images = glob.glob("./picture3/*jpg")
for name in mound_images:
    img = cv2.imread(name)
    center_point = np.zeros(3)
    center_point[0] = img.shape[1] / 2
    center_point[1] = img.shape[0] / 2
    center_point[2] = H

    # cv2.imshow('img', img)
    # cv2.setMouseCallback('img', MouseLeftClick)
    # cv2.waitKey()

    # fixToMidPoint(corner_points, center_point)
    # corner_points[0,2]=corner_points[1,2]=corner_points[2,2]=corner_points[3,2] = H
    
    # index = 0
    # cv2.imshow('img', img)
    # cv2.setMouseCallback('img', MouseLeftClick2)
    # cv2.waitKey()
    # fixToMidPoint(mid_points, center_point)
    # mid_points[0,2]=mid_points[1,2]=mid_points[2,2]=mid_points[3,2] = H


    # index = 0
    # cv2.imshow('img', img)
    # cv2.setMouseCallback('img', MouseLeftClick3)
    # cv2.waitKey()

    # fixToMidPoint(test_points, center_point)
    # test_points[0,2]=test_points[1,2]=test_points[2,2]=test_points[3,2] = H

    # print(test_points)

# /////////////////////////////////

    corner_points = np.array(
        [[ 219 ,  29,  H],
        [ -32,  221,  H],
        [-412,  117,  H],
        [-253, -178,  H]], dtype=np.float64)
    mid_points = np.array(
        [[  75,  139 , H],
        [-203 , 174 , H],
        [-345 ,  -2 , H],
        [  11 , -59 , H]], dtype=np.float64)
    test_points = np.array(
        [[  52.,  116.,  H],
        [-143. ,  91. , H],
        [-282. ,  13. , H],
        [-640. , 480. , H]], dtype = np.float64)


    print(center_point)
    print("corner_points : ", corner_points)
    print("mid_points : ", mid_points)

    first_to_second_ratio = (corner_points[0, 0] - mid_points[0, 0]) / (mid_points[0, 0] - corner_points[1, 0])
    first_to_first_mid_ratio = (1 + first_to_second_ratio) / 2
    first_to_forth_ratio = (corner_points[0, 0] - mid_points[3, 0]) / (mid_points[3, 0] - corner_points[3, 0])
    first_to_forth_mid_ration = (1 + first_to_forth_ratio) / 2
    print(first_to_second_ratio)
    print(first_to_first_mid_ratio)
    print(first_to_forth_ratio)
    print(first_to_forth_mid_ration)

    # corner_points = corner_points - np.array([U, V, 0])
    # corner_points = corner_points * np.array([Hx, Hy, 1])

    # mid_points = mid_points - np.array([U, V, 0])
    # mid_points = mid_points * np.array([Hx, Hy, 1])

    # print(corner_points * M)
    # print(mid_points)

    # object_points = corner_points
    # mid_object_points = mid_points

    # object_points[0] = M * object_points[0]
    # object_points[1] = M * first_to_second_ratio * object_points[1]
    # object_points[3] = M * first_to_forth_ratio * object_points[3]

    # mid_object_points[0] = M * first_to_first_mid_ratio * mid_points[0]
    # mid_object_points[3] = M * first_to_forth_mid_ration * mid_points[3]

    # print(object_points)

    # print(mid_object_points)

    # print("hello")
    # a = cal_intrinsic_param(object_points[0], mid_points[0], object_points[1], mid_points[3], object_points[3])
    # print(a)
    # print("hello2")
    plane_points[0] = object_points[0] = M * corner_points[0]
    plane_points[1] = object_points[1] = M * first_to_second_ratio * corner_points[1]
    plane_points[2] = object_points[3] = M * first_to_forth_ratio * corner_points[3]

    print(object_points)
    print(plane_points)

    # print("focus length : ", cal_focus_length(object_points[0], object_points[1], object_points[3]))

    plane_matrix = np.linalg.inv(plane_points)
    print(plane_matrix)
    plane_matrix = plane_matrix.sum(axis = 1)
    print(plane_matrix)

    first_test_ratio = 1 / test_points[0].dot(plane_matrix)
    second_test_ratio = 1 / test_points[1].dot(plane_matrix)
    third_test_ratio = 1 / test_points[2].dot(plane_matrix)

    print("distance first test to forth : ", cal_distance2(test_points[0] * first_test_ratio, object_points[3]))
    print("distance second test to forth : ", cal_distance2(test_points[1] * second_test_ratio, object_points[3]))
    print("distance third test to forth : ", cal_distance2(test_points[2] * third_test_ratio, object_points[3]))
    
    print("distnace first to second : ", cal_distance(object_points, 0, 1))
    print("distnace first to forth : ", cal_distance(object_points, 0, 3))
    print("distnace seconde to forth : ", cal_distance(object_points, 1, 3))

    break


cv2.destroyAllWindows()