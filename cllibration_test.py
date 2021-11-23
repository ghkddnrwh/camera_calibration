import numpy as np
import glob, cv2
import math
from sympy import Symbol, solve

from hello import extensionRatio

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

def cal_object_points(_corner_points):
    cor_choice = 0
    if abs(_corner_points[0, 0] - _corner_points[4, 0]) > abs(_corner_points[0, 1] - _corner_points[4, 1]):
        cor_choice = 0
    else:
        cor_choice = 1
    
    print("cor choice : ", cor_choice)
    home_to_second_ratio  = (_corner_points[0, cor_choice] - _corner_points[4, cor_choice]) / (_corner_points[4, cor_choice] - _corner_points[2, cor_choice])
    
    if abs(_corner_points[1, 0] - _corner_points[4, 0]) > abs(_corner_points[1, 1] - _corner_points[4, 1]):
        cor_choice = 0
    else:
        cor_choice = 1

    first_to_third_ratio = (_corner_points[1, cor_choice] - _corner_points[4, cor_choice]) / (_corner_points[4, cor_choice] - _corner_points[3, cor_choice])
    print("cor choice : ", cor_choice)

    mound_to_home_ratio = 2 / (1 + home_to_second_ratio)
    mound_to_second_ratio = 2 * home_to_second_ratio / (1 + home_to_second_ratio)
    mound_to_first_ratio = 2 / (1 + first_to_third_ratio)
    mound_to_third_ratio = 2 * first_to_third_ratio / (1 + first_to_third_ratio)

    print("mound to home ration : ", mound_to_home_ratio)
    print("mound to first ration : ", mound_to_first_ratio)
    print("mound to second ration : ", mound_to_second_ratio)
    print("mound to third ration : ", mound_to_third_ratio)

    _object_points = np.zeros((5, 3), dtype=np.float64)

    _object_points[0] =  mound_to_home_ratio * _corner_points[0]
    _object_points[1] =  mound_to_first_ratio * _corner_points[1]
    _object_points[2] =  mound_to_second_ratio * _corner_points[2]
    _object_points[3] =  mound_to_third_ratio * _corner_points[3]
    _object_points[4] =  _corner_points[4]

    return _object_points

def cal_two_focus_length(object_points):
    _a0 = object_points[0, 0] - object_points[2 , 0]
    _a1 = object_points[0, 1] - object_points[2 , 1]
    _a2 = object_points[0, 2] - object_points[2 , 2]

    _b0 = object_points[1, 0] - object_points[3 , 0]
    _b1 = object_points[1, 1] - object_points[3 , 1]
    _b2 = object_points[1, 2] - object_points[3 , 2]

    coefficient_mat = np.zeros((2, 2), dtype = np.float64)
    coefficient_mat[0, 0] = _a0**2 - _b0**2
    coefficient_mat[0, 1] = _a1**2 - _b1**2
    coefficient_mat[1, 0] = _a0 * _b0
    coefficient_mat[1, 1] = _a1 * _b1

    const_mat = np.zeros((2, 1), dtype=np.float64)
    const_mat[0, 0] = _b2**2 - _a2**2
    const_mat[1, 0] = - _a2 * _b2

    print(coefficient_mat)
    print(const_mat)

    focus_np = np.dot(np.linalg.inv(coefficient_mat), const_mat)

    print(focus_np)
    print("focus length : ", 1 / np.sqrt(focus_np))

    sum_mat = coefficient_mat.sum(axis=1)

    h1 = 1 / math.sqrt(const_mat[0, 0] / sum_mat[0])
    h2 = 1 / math.sqrt(const_mat[1, 0] / sum_mat[1])

    print("h val : ", h1, h2)

    
    # 1 / np.sqrt(focus_np)
    return np.array([h1, h1])

def cal_real_object_points(_object_points, _test_points):
    plane_mat = _object_points[:3, :]
    plane_mat = np.linalg.inv(plane_mat)
    plane_mat = plane_mat.sum(axis = 1)
    print(plane_mat)
    print(_test_points)
    ratio_mat = 1 / _test_points.dot(plane_mat)
    print(ratio_mat)

    _test_points[0, :] *= ratio_mat[0]
    _test_points[1, :] *= ratio_mat[1]
    _test_points[2, :] *= ratio_mat[2]
    _test_points[3, :] *= ratio_mat[3]

    return _test_points

def cal_extrinsic_mat(_object_points):
    x_vec = _object_points[1] - _object_points[0]
    y_vec = _object_points[3] - _object_points[0]

    norm_size = np.linalg.norm(x_vec)
    x_vec = x_vec / norm_size
    y_vec = y_vec / np.linalg.norm(y_vec)
    z_vec = np.cross(x_vec, y_vec)
    t_vec = _object_points[0]

    extrinsic_mat = np.expand_dims(x_vec, axis=1)
    extrinsic_mat = np.append(extrinsic_mat, np.expand_dims(y_vec, axis=1), axis=1)
    extrinsic_mat = np.append(extrinsic_mat, np.expand_dims(z_vec, axis=1), axis=1)
    extrinsic_mat = np.append(extrinsic_mat, np.expand_dims(t_vec, axis=1), axis=1)
    extrinsic_mat = np.append(extrinsic_mat, np.array([[0, 0, 0, 1]]), axis=0)

    print("extrinsic matrix")
    print(x_vec)
    print(y_vec)
    print(z_vec)
    print(t_vec)
    print(extrinsic_mat)
    print(np.linalg.inv(extrinsic_mat))
    return np.linalg.inv(extrinsic_mat) / norm_size

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

cali_images = glob.glob("./picture5/*.jpg")
ret, mtx, dists, rvecs, tvecs = get_camera_matrix(cali_images)

print("mtx : ", mtx)

H = -1
# H = 918.5032416076951
# M = 3000
# corner_points = np.zeros((4,3))
# mid_points = np.zeros((4,3))
# object_points = np.zeros((4,3))
# mid_object_points = np.zeros((4,3))

index = 0

total_test_points = np.zeros((2, 4, 3), dtype = np.float64)
total_origin_points = np.zeros((2, 1, 3), dtype= np.float64)

mound_images = glob.glob("./picture4/*jpg")
cnt = 0
for name in mound_images:
    corner_points = np.zeros((5, 3), dtype=np.float64)
    object_points = np.zeros((5, 3), dtype=np.float64)
    plane_points = np.zeros((3, 3))
    test_points = np.zeros((4, 3), dtype= np.float64)
    cnt += 1
    # if cnt == 1 or cnt ==2:
    #     continue
    if cnt == 3:
        break
    # if cnt == 0 or cnt == 1 or cnt == 2:
    #     cnt += 1
    #     continue
    img = cv2.imread(name)
    center_point = np.zeros(3)
    center_point[0] = img.shape[1] / 2
    center_point[1] = img.shape[0] / 2
    center_point[2] = H

    index = 0
    cv2.imshow('img', img)
    cv2.setMouseCallback('img', MouseLeftClick)
    cv2.waitKey()

    fixToMidPoint(corner_points, center_point)
    corner_points[0,2]=corner_points[1,2]=corner_points[2,2]=corner_points[3,2] = corner_points[4,2] = H

    # index = 0
    # cv2.imshow('img', img)
    # cv2.setMouseCallback('img', MouseLeftClick2)
    # cv2.waitKey()
    # fixToMidPoint(mid_points, center_point)
    # mid_points[0,2]=mid_points[1,2]=mid_points[2,2]=mid_points[3,2] = H


    index = 0
    cv2.imshow('img', img)
    cv2.setMouseCallback('img', MouseLeftClick3)
    cv2.waitKey()

    fixToMidPoint(test_points, center_point)
    test_points[0,2]=test_points[1,2]=test_points[2,2]=test_points[3,2] = H


# /////////////////////////////////

    # if cnt == 3:
    #     corner_points = np.array(
    #         [[ 295., -346. ,  -1.],
    #         [ 410. , -12. ,  -1.],
    #         [  -9. ,  68. ,  -1.],
    #         [-273. ,-157. ,  -1.],
    #         [  97. , -78. ,  -1.]],dtype = np.float64
    #     )
    #     test_points = np.array(
    #         [[-229.,  246. ,  -1.],
    #         [-211. , 135.  , -1.],
    #         [-720. , 540.  , -1.],
    #         [-720. , 540.  , -1.]], dtype = np.float64
    #     )
    # elif cnt ==4:
    #     corner_points = np.array(
    #         [[ -45., -395. ,  -1.],
    #         [ 331. , -98.  , -1.],
    #         [ -18. ,  32. ,  -1.],
    #         [-377. ,-132. ,  -1.],
    #         [ -28. ,-115. ,  -1.]], dtype = np.float64)
    #     test_points = np.array(
    #         [[-229.,  213. ,  -1.],
    #         [-210. , 114.  , -1.],
    #         [-720. , 540. ,  -1.],
    #         [-720. , 540. ,  -1.]], dtype = np.float64 )


    corner_points[4] = cal_shortest_point(corner_points[0], corner_points[2], corner_points[1], corner_points[3])
    print("center_point : ", center_point)
    print("corner_points : ", corner_points)
    print("test points : ", test_points)
    
    object_points = cal_object_points(corner_points)
    print("object points : ", object_points)


    focus_len = cal_two_focus_length(object_points)

    # focus_len = np.array([1070, 1070])
    # focus_len = np.array([1500, 1500])

    corner_points[:,0] /= focus_len[0]
    corner_points[:,1] /= focus_len[1]

    object_points[:,0] /= focus_len[0]
    object_points[:,1] /= focus_len[1]


    test_points[:,0] /= focus_len[0]
    test_points[:,1] /= focus_len[1]

    test_points = cal_real_object_points(object_points, test_points)

    print("corner points : ", corner_points)
    print("object points : ", object_points)

    print("distnace home to first : ", cal_distance(object_points, 0, 1))
    print("distnace home to first : ", cal_distance(object_points, 2, 1))
    print("distnace home to first : ", cal_distance(object_points, 3, 2))
    print("distnace home to third : ", cal_distance(object_points, 0, 3))
    print("distnace first to third : ", cal_distance(object_points, 1, 3))

    print("distance first test to home : ", cal_distance2(test_points[0], object_points[0]))
    print("distance first test to home : ", cal_distance2(test_points[1], object_points[0]))
    print("distance first test to home : ", cal_distance2(test_points[2], object_points[0]))

    extrinsic_mat = cal_extrinsic_mat(object_points)

    origin_point = np.array([[0], [0], [0], [1]])

    test_points = test_points.T
    test_points = np.append(test_points, np.array([[1, 1, 1, 1,]]), axis=0)

    object_points = object_points.T
    object_points = np.append(object_points, np.array([[1, 1, 1, 1, 1]]), axis=0)

    print("origin points  : ", origin_point)
    print("trans points : ", test_points)
    print("object points : ", object_points)
    
    origin_point = np.dot(extrinsic_mat, origin_point).T
    test_points = np.dot(extrinsic_mat, test_points).T
    object_points = np.dot(extrinsic_mat, object_points).T

    print("result object points : ", object_points)
    print("result test points : ", test_points)
    print("result origin point : ", origin_point)

    print("distance home to first : ", cal_distance2(object_points[0], object_points[1]))
    print("distance home to first : ", cal_distance2(object_points[2], object_points[1]))
    print("distance home to first : ", cal_distance2(object_points[3], object_points[2]))
    print("distance home to first : ", cal_distance2(object_points[0], object_points[3]))

    print("distance first test to home : ", cal_distance2(test_points[0], object_points[0]))
    print("distance first test to home : ", cal_distance2(test_points[1], object_points[0]))
    print("distance first test to home : ", cal_distance2(test_points[2], object_points[0]))
    print("distance first test to home : ", cal_distance2(test_points[3], object_points[0]))
    print("distance origin to home : ", cal_distance2(origin_point[0], object_points[0]))

    total_test_points[cnt - 1, :, :] = test_points[:, :3]
    total_origin_points[cnt - 1, :, :] = origin_point[:, :3]
    print(total_test_points)
    print(total_origin_points)

print("finish")
print(cal_shortest_point(total_test_points[0, 0], total_origin_points[0, 0], total_test_points[1, 0], total_origin_points[1, 0]))
print(cal_shortest_point(total_test_points[0, 1], total_origin_points[0, 0], total_test_points[1, 1], total_origin_points[1, 0]))

cv2.destroyAllWindows()