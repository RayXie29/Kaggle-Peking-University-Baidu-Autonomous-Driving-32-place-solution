'''
Utills library for data processing, prediction decode and prediction postpprocessing
Some functions are from several kaggle kernels
https://www.kaggle.com/zstusnoopy/visualize-the-location-and-3d-bounding-box-of-car
https://www.kaggle.com/hocop1/centernet-baseline
https://www.kaggle.com/ebouteillon/augmented-reality
Thanks a lot for their shareing
'''

import cv2
import numpy as np
from math import cos, sin
from project_config import *
import matplotlib.pyplot as plt

def str2coords(s, names):
    '''
    Function for decoding the prediction string
    ------------------------------------------------------------
    '''
    coords = []
    for l in np.array(s.split()).reshape([-1, 7]):
        coords.append(dict(zip(names, l.astype('float'))))
    return coords

def PredStr_2_6dof(predstr):
    '''
    Function for decoding the prediction string
    ------------------------------------------------------------
    '''
    tag=['id','pitch','yaw','roll','x','y','z']
    coordinate = []
    predstr = np.array(predstr.split()).reshape(-1,7)
    for ps in predstr:
        coordinate.append(dict(zip(tag,ps.astype('float'))))
        coordinate[-1]['id'] = int(coordinate[-1]['id'])

    return coordinate



def SIXDOF_2_ImageCoordinate(coordinate, maximum_objects=100):
    '''
    Function for generating 2d coordinate of cars in image
    ------------------------------------------------------------
    Input:

    Decoded prediction string

    Output:

    Car 2d coordinate(y, x format) in original image scale
    '''
    if len(coordinate) > maximum_objects:
        coordinate = np.random.choice(coordinate, maximum_objects)

    X = [c['x'] for c in coordinate]
    Y = [c['y'] for c in coordinate]
    Z = [c['z'] for c in coordinate]

    real_world_coordinates = np.array(list(zip(X,Y,Z))).T
    image_coordinate = np.dot(IntrinsicMatrix, real_world_coordinates).T
    image_coordinate[:,0] /= image_coordinate[:,2]
    image_coordinate[:,1] /= image_coordinate[:,2]

    return np.array( [image_coordinate[:,1], image_coordinate[:,0]], dtype=np.float).T



def euler_to_Rot(yaw, pitch, roll):
    '''
    Function for transform the euler angle to rotation matrix
    ------------------------------------------------------------
    Input :

    Eular angles

    Output:

    Rotation matrix
    '''
    Y = np.array([[cos(yaw), 0, sin(yaw)],
                  [0, 1, 0],
                  [-sin(yaw), 0, cos(yaw)]])

    P = np.array([[1, 0, 0],
                  [0, cos(pitch), -sin(pitch)],
                  [0, sin(pitch), cos(pitch)]])

    R = np.array([[cos(roll), -sin(roll), 0],
                  [sin(roll), cos(roll), 0],
                  [0, 0, 1]])

    return np.dot(Y, np.dot(P,R))



def Visual3D(image, args):
    '''
    Function for visualize the 3D brounding box of cars
    ------------------------------------------------------------
    Input:

    image
    6dof in `pitch, yaw, roll, x, y, z` order

    Output:

    image with drawed 3d bounding box
    '''
    def draw_line(image, points):
        color = (255, 0, 0)
        cv2.line(image, tuple(points[1][:2]), tuple(points[2][:2]), color, 8)
        cv2.line(image, tuple(points[1][:2]), tuple(points[4][:2]), color, 8)

        cv2.line(image, tuple(points[1][:2]), tuple(points[5][:2]), color, 8)
        cv2.line(image, tuple(points[2][:2]), tuple(points[3][:2]), color, 8)
        cv2.line(image, tuple(points[2][:2]), tuple(points[6][:2]), color, 8)
        cv2.line(image, tuple(points[3][:2]), tuple(points[4][:2]), color, 8)
        cv2.line(image, tuple(points[3][:2]), tuple(points[7][:2]), color, 8)

        cv2.line(image, tuple(points[4][:2]), tuple(points[8][:2]), color, 8)
        cv2.line(image, tuple(points[5][:2]), tuple(points[8][:2]), color, 8)

        cv2.line(image, tuple(points[5][:2]), tuple(points[6][:2]), color, 8)
        cv2.line(image, tuple(points[6][:2]), tuple(points[7][:2]), color, 8)
        cv2.line(image, tuple(points[7][:2]), tuple(points[8][:2]), color, 8)
        return image


    def draw_points(image, points):
        image = np.array(image)
        for (p_x, p_y, p_z) in points:
            cv2.circle(image, (p_x, p_y), 5, (255, 0, 0), -1)
        return image

    x_l = 1.02
    y_l = 0.80
    z_l = 2.31

    img = image.copy()
    pitch, yaw, roll, x, y, z = args

    yaw = -yaw
    pitch = -pitch
    roll = -roll

    RT_mat = np.eye(4)
    RT_mat[:3,3] = np.array([x,y,z])
    RT_mat[:3,:3] = euler_to_Rot(yaw, pitch,roll).T
    RT_mat = RT_mat[:3,:]

    P = np.array([[0, 0, 0, 1],
                  [x_l, y_l, -z_l, 1],
                  [x_l, y_l, z_l, 1],
                  [-x_l, y_l, z_l, 1],
                  [-x_l, y_l, -z_l, 1],
                  [x_l, -y_l, -z_l, 1],
                  [x_l, -y_l, z_l, 1],
                  [-x_l, -y_l, z_l, 1],
                  [-x_l, -y_l, -z_l, 1]]).T

    points = np.dot(IntrinsicMatrix, np.dot(RT_mat, P)).T
    points[:,0] /= points[:,2]
    points[:,1] /= points[:,2]

    points = points.astype('int')
    img = draw_line(img, points)
    img = draw_points(img, points)

    return img

def rotate(x, angle):
    '''
    Function for encoding the roll ground truth
    ------------------------------------------------------------
    '''
    x = x + angle
    x = x - (x + np.pi) // (2 * np.pi) * 2 * np.pi
    return x


def rotateImage(alpha=0, beta=0, gamma=0, dx=1686.2379, dy=1354.9849):
    '''
    Function for generating the 2D and 3D rotation transformation matrix
    ------------------------------------------------------------
    Input:

    alpha : rotation angle around x-axis in radian format
    beta : rotation angle around y-axis in radian format
    gamma : rotation angle around z-axis in radian fotmat
    dx : Camera principal point of x-axis
    dy : Camera principal point of y-axis

    Output:

    TransMat : 2D transformation matrix of rotation
    RotMat : 3D transformation matrix of rotation
    '''
    fx, dx = 2304.5479, dx
    fy, dy = 2305.8757, dy

    A1 = np.array([[1/fx, 0, -dx/fx],
                   [0, 1/fy, -dy/fy],
                   [0, 0,    1],
                   [0, 0,    1]])

    RX = np.array([[1,          0,           0, 0],
             [0, cos(alpha), -sin(alpha), 0],
             [0, sin(alpha),  cos(alpha), 0],
             [0,          0,           0, 1]])

    RY = np.array([[cos(beta), 0, -sin(beta), 0],
              [0, 1,          0, 0],
              [sin(beta), 0,  cos(beta), 0],
              [0, 0,          0, 1]])
    RZ = np.array([[cos(gamma), -sin(gamma), 0, 0],
              [sin(gamma),  cos(gamma), 0, 0],
              [0,          0,           1, 0],
              [0,          0,           0, 1]])

    A2 = np.array([[fx, 0, dx, 0],
                   [0, fy, dy, 0],
                   [0, 0,   1, 0]])

    RotMat = np.dot(RZ, np.dot(RX,RY))
    trans = np.dot(A2, np.dot(RotMat, A1))

    return trans, RotMat

############################################################################

def clear_predictions(prediction, tags = ['pitch', 'yaw', 'roll', 'x', 'y', 'z']):
    '''
    Function for prediction postprocessing, which will discard the predictions which are too close to other prediction
    Also discarding the predictions that the confidence value is lower than threshold
    ------------------------------------------------------------
    '''
    new_prediction = []
    tag=['pitch','yaw','roll','x','y','z']

    pitchs, yaws, rolls, xs, ys, zs, confidences = np.split(prediction,7)

    predstr = np.array([pitchs, yaws, rolls, xs, ys, zs], dtype=np.float32).transpose(1,0)
    coordinate = []
    for ps in predstr:
        coordinate.append(dict(zip(tag,ps.astype('float'))))
    image_coordinates = SIXDOF_2_ImageCoordinate(coordinate)
    ys_2d = image_coordinates[:,0]
    xs_2d = image_coordinates[:,1]

    for idx, (x1, y1, c1) in enumerate(zip(xs_2d, ys_2d, confidences)):
        for idx2, (x2, y2, c2) in enumerate(zip(xs_2d, ys_2d, confidences)):
            if idx != idx2:
                distance = np.sqrt((x1-x2)**2 + (y1-y2)**2)
                if distance < config['DUPLICATE_CAR_DISTANCE_THRESHOLD']:
                    if c1 <= c2:
                        confidences[idx] = -1

    for p, y, r, X, Y, Z, c in zip(pitchs, yaws, rolls, xs, ys, zs, confidences):
        if (c >= config['CONFIDENCE_THRESHOLD']) :
            new_prediction.append(np.array([p,y,r,X,Y,Z,c]))

    if len(new_prediction) == 0:
        return np.array([])
    return np.stack(new_prediction)


def decode_predictions(prediction, fx=2304.5479, fy=2305.8757, cx=1686.2379, cy=1354.9849):
    '''
    Function for decoding the prediction into desire format
    ------------------------------------------------------------
    '''
    pitchs, yaws, rolls, zs, confidences, kp_indices = np.split(prediction, 6)

    xs_2d = ( kp_indices % config['OUTPUT_SIZE'][1] ) / config['OUTPUT_SIZE'][1] * 3384
    ys_2d = ( kp_indices // config['OUTPUT_SIZE'][1]) / config['OUTPUT_SIZE'][0] * (2710-config['ORIGINAL_Y_CROPPED']) + config['ORIGINAL_Y_CROPPED']

    #Transform 2D image coordinates into X and Y (world coordinate)
    xs = ( xs_2d - cx ) * zs / fx
    ys = ( ys_2d - cy ) * zs / fy

    new_prediction = np.concatenate([pitchs, yaws, rolls, xs, ys, zs, confidences], axis=0)
    return new_prediction
