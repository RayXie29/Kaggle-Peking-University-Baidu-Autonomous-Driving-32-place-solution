import os
import cv2
import warnings
import numpy as np
import pandas as pd
from Parameters import *
from math import cos, sin
from utills import Visual3D
from project_config import *
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R



warnings.filterwarnings("ignore")

def __drawGaussian2D__(center_x, center_y, sigma):

    x_grid = np.linspace(0, 3384-1, 3384)
    y_grid = np.linspace(0, 2710-1, 2710)

    x_grid, y_grid = np.meshgrid(x_grid, y_grid)
    x_grid -= center_x
    y_grid -= center_y

    heatmap = np.exp(-((x_grid**2)+(y_grid**2)) / 2*(sigma**2))
    return heatmap

def PredStr_2_6dof(predstr):

    tag=['id','pitch','yaw','roll','x','y','z']
    coordinate = []
    predstr = np.array(predstr.split()).reshape(-1,7)
    for ps in predstr:
        coordinate.append(dict(zip(tag,ps.astype('float'))))
        coordinate[-1]['id'] = int(coordinate[-1]['id'])

    return coordinate

def SIXDOF_2_ImageCoordinate(coordinate, maximum_objects=100):

    if len(coordinate) > maximum_objects:
        coordinate = np.random.choice(coordinate, maximum_objects)

    X = [c['x'] for c in coordinate]
    Y = [c['y'] for c in coordinate]
    Z = [c['z'] for c in coordinate]

    real_world_coordinates = np.array(list(zip(X,Y,Z))).T
    image_coordinate = np.dot(IntrinsicMatrix, real_world_coordinates).T
    image_coordinate[:,0] /= image_coordinate[:,2]
    image_coordinate[:,1] /= image_coordinate[:,2]

    return np.array( [image_coordinate[:,1], image_coordinate[:,0]], dtype=np.int).T

def get_xy_from_XYz(X, Y, z):
    x = (X - 1686.2379)*z/2304.5479
    y = (Y - 1354.9849)*z/2305.8757
    return x,y

def euler_to_Rot(yaw, pitch, roll):
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



def rotateImage(alpha=0, beta=0, gamma=0, dx=1686.2379, dy=1354.9849):

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

train_csv = pd.read_csv(TRAIN_CSV_DIR)
selected_index = 555
imageid = train_csv['ImageId'].iloc[selected_index]
predstr = train_csv['PredictionString'].iloc[selected_index]
coordinates = PredStr_2_6dof(predstr)

image_coordinates = SIXDOF_2_ImageCoordinate(coordinates)
image = plt.imread(os.path.join(TRAIN_IMAGE_DIR, imageid+'.jpg'))
tag=['pitch','yaw','roll','x','y','z']
keypoints = []
rotated_SixDof = []
original_image = image.copy()
image = image[1355:]
image = cv2.resize(image, (INPUT_IMAGE_SIZE[1],INPUT_IMAGE_SIZE[0]))
mask = np.zeros(OUTPUT_SIZE)

alpha, beta, gamma = 0, 0, 0
alpha = alpha*np.pi/180.0
beta = beta*np.pi/180.0
gamma = gamma*np.pi/180.0
RotTrans, RotMat = rotateImage(alpha, beta, gamma)

image_coordinates = np.array([image_coordinates[:,1], image_coordinates[:,0], np.ones(len(image_coordinates))]).transpose(1,0)
image_coordinates[:,1] -= 1355
image_coordinates[:,0] *= (INPUT_IMAGE_SIZE[1]/ORIGINAL_IMAGE_SIZE[1])
image_coordinates[:,1] *= (INPUT_IMAGE_SIZE[0]/(ORIGINAL_IMAGE_SIZE[0]-1355))
transformed_image_coordinates = np.dot(RotTrans, image_coordinates.T).T

transformed_image_coordinates[:,0] *= (OUTPUT_SIZE[1]/INPUT_IMAGE_SIZE[1])
transformed_image_coordinates[:,1] *= (OUTPUT_SIZE[0]/INPUT_IMAGE_SIZE[0])

keypoints.append(list(transformed_image_coordinates[:,:2]))

for c in coordinates:
    pitch, yaw, roll, x, y, z = c['pitch'], c['yaw'], c['roll'], c['x'], c['y'], c['z']
    x, y, z = np.dot(RotMat[:3,:3], np.array([x,y,z]).T).T

    r1 = R.from_euler('xyz', [-yaw, -pitch, -roll], degrees=False)
    r2 = R.from_euler('xyz', [beta, -alpha, -gamma], degrees=False)
    yaw, pitch, roll = (r2*r1).as_euler('xyz')*-1
    rotated_SixDof.append(dict(zip(tag, np.array([pitch, yaw, roll, x, y, z], dtype='float'))))

for img_coor in transformed_image_coordinates:
    x, y, _ = img_coor
    y = int(y)
    x = int(x)
    if (x>=0) & (x<OUTPUT_SIZE[1]) & (y>=0) & (y<OUTPUT_SIZE[0]):
        mask[y,x] = 1.0

original_transfomed_image = cv2.warpPerspective(original_image, M=RotTrans, dsize=(3384, 2710))
transformed_image = cv2.warpPerspective(image, M=RotTrans, dsize=(INPUT_IMAGE_SIZE[1],INPUT_IMAGE_SIZE[0]))
transformed_image = cv2.resize(transformed_image, (OUTPUT_SIZE[1], OUTPUT_SIZE[0]))


for c in coordinates:
    X,Y,Z = c['x'], c['y'], c['z']
    pitch, yaw, roll = c['pitch'], c['yaw'], c['roll']
    original_image = Visual3D(original_image, [pitch, yaw, roll, X, Y, Z])

for c in rotated_SixDof:
    X,Y,Z = c['x'], c['y'], c['z']
    pitch, yaw, roll = c['pitch'], c['yaw'], c['roll']
    original_transfomed_image = Visual3D(original_transfomed_image, [pitch, yaw, roll, X, Y, Z])



plt.figure(figsize=(12,12))
plt.imshow(original_image)
plt.show()
plt.figure(figsize=(12,12))
plt.imshow(original_transfomed_image)
plt.show()


'''
from Dataset import *
import albumentations as albu

train_df, valid_df = train_test_split(train_csv, test_size=0.2, random_state=2019)

train_augumentation = albu.Compose([
                                    albu.OneOf([
                                        albu.RandomBrightness(limit=0.25),
                                        albu.RandomContrast(limit=0.3),
                                        albu.RandomGamma(),
                                    ], p=0.5),
                                    albu.GaussNoise(p=0.25),
])


train_generator = InputGenerator(data_df = train_df,
                                 input_size = INPUT_IMAGE_SIZE,
                                 image_augumentation = train_augumentation,
                                 output_size=OUTPUT_SIZE,
                                 batch_size=6,
                                 do_hflip=True,
                                 do_rotate=True
                                 )

len = train_generator.__len__()
[images, heatmaps, regr_label, keypoints_indices, regr_mask, TMs], dummy_label = train_generator.__getitem__(np.random.choice(range(len),1)[0])

for image, heatmap, label, keypoint_index, mask, TM in zip(images, heatmaps, regr_label, keypoints_indices, regr_mask, TMs):

    gt_len = np.sum(mask).astype('int')
    temp = cv2.warpPerspective(image, np.linalg.inv(TM), (INPUT_IMAGE_SIZE[1], INPUT_IMAGE_SIZE[0]))
    temp = cv2.resize(temp, (3384, 2710-1526))
    original_image = np.zeros((2710, 3384, 3))
    original_image[1526:] = temp

    original_image = cv2.warpPerspective(original_image, TM, (3384, 2710))

    for dof in label[:gt_len]:
        yaw, pitch, roll, x, y, z = dof
        original_image = Visual3D(original_image, [pitch, yaw, roll, x, y, z])

    plt.figure(figsize=(12,12))
    plt.imshow(original_image)
    plt.show()

    small_image = cv2.resize(image, (OUTPUT_SIZE[1], OUTPUT_SIZE[0]))
    plt.figure(figsize=(12,12))
    plt.imshow(small_image)
    plt.imshow(np.squeeze(heatmap, axis=-1), alpha=0.65, cmap='gray')
    plt.show()
'''
