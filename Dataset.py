'''
Keras data generator for training/validation/testing.
Image and ground truth rotation is referred to https://www.kaggle.com/outrunner/rotation-augmentation
'''

import os
import cv2
import keras
import random
import numpy as np
from utills import *
from project_config import *
from random import randrange
from keras.preprocessing.image import load_img
from scipy.spatial.transform import Rotation as R


class InputGenerator(keras.utils.Sequence):

    def __init__(self,
                 data_df,
                 input_size,
                 image_augumentation=None,
                 output_size=(128,128),
                 batch_size = 2,
                 training = True,
                 image_load_dir = './',
                 mask_load_dir = './',
                 do_hflip=True,
                 do_rotate=True,
                 X_rot_angle = 4,
                 Y_rot_angle = 4,
                 Z_rot_angle = 4,
                 hflip_prob = 0.5,
                 rot_prob = 0.5
                 ):
        '''
        data_df : pandas dataframe for imageid and predictionstring(ground truth)
        input_size : input image size
        image_augumentation : augmentation for image (albumentations module)
        output_size : prediction output size
        training : flag for determinding the generator is for training or validating/testing
        do_hflip : flag for determinding whether perform the horizontal flip augmentation on image and ground truth
        do_rotate : flag for determinding whether perform the rotation augmentation on image and ground truth
        X_rot_angle : X-axis rotation angle, will work when do_rotate is True
        Y_rot_angle : Y-axis rotation angle, will work when do_rotate is True
        Z_rot_angle : Z-axis rotation angle, will work when do_rotate is True
        hflip_prob : probability of performing image and ground truth horizontal flip
        rot_prob : probability of performing image and ground truth rotation
        '''

        self.data_df = data_df
        self.input_size = input_size
        self.image_augumentation = image_augumentation
        self.output_size = output_size
        self.indices = range(len(data_df))
        self.batch_size = batch_size
        self.maximum_objects = config['MAX_OBJECTS']
        self.training = training
        self.image_load_dir = image_load_dir
        self.mask_load_dir = mask_load_dir
        self.do_hflip = do_hflip
        self.do_rotate = do_rotate
        self.X_rot_angle = np.arange(-X_rot_angle, X_rot_angle+0.5, 0.5)
        self.Y_rot_angle = np.arange(-Y_rot_angle, Y_rot_angle+0.5, 0.5)
        self.Z_rot_angle = np.arange(-Z_rot_angle, Z_rot_angle+0.5, 0.5)
        self.hflip_prob = hflip_prob
        self.rot_prob = rot_prob
        '''
        CX : camera principal point of x-axis
        Y_cropped : Amount for cropping the upper region of original image(Since upper region don't have cars in it)
        '''
        self.CX = np.floor(IntrinsicMatrix[0,2]).astype(np.int)
        self.Y_cropped = config['ORIGINAL_Y_CROPPED']

    def __len__(self):
        return self.data_df.shape[0] // self.batch_size

    def on_epoch_start(self):
        if training:
            np.random.shuffle(data_df)

    def __getitem__(self, index):
        '''
        Function for generating input for training/validating/testing model
        ------------------------------------------------------------
        '''
        batch_indices = self.indices[index * self.batch_size : (index+1) * self.batch_size]
        ImageIds = self.data_df['ImageId'].iloc[batch_indices].values
        images = self.__getimages__(ImageIds)

        if self.training:

            PredStrs = self.data_df['PredictionString'].iloc[batch_indices].values
            SixDofs = [PredStr_2_6dof(Pstrs) for Pstrs in PredStrs]

            if self.do_hflip:
                images, SixDofs = self.h_flip(images, SixDofs)
            if self.do_rotate:
                images, SixDofs, keypoints, keypoints_indices, regr_mask = self.__Rotate_image_n_gt__(SixDofs, images)
            else:
                keypoints, keypoints_indices, regr_mask = self.__getkeypoints__keypointindices__(SixDofs)

            #get regression label function must executed after horizonal fliping, getting keypoints or Rotating images and ground truth
            regr_label = self.__get_regression_gt__(SixDofs)
            #Same as get heatmap function
            heatmaps = self.__getheatmaps__(keypoints, SixDofs)
            #Dummy ground truth for model training
            dummy_label = np.zeros((self.batch_size,))

            return [images/255.0, heatmaps, regr_label, keypoints_indices, regr_mask], dummy_label
        else:
            return images/255.0

    def __getimages__(self, ImageIds):
        '''
        Function for loading and preprocessing the image
        ------------------------------------------------------------
        '''
        images = []
        for idx, id in enumerate(ImageIds):
            image = self.get_masked_data(id, self.image_load_dir, self.mask_load_dir)
            #Upper side of image is useless(no car information)
            image = image[self.Y_cropped:]
            image = cv2.resize(image, (self.input_size[1], self.input_size[0]))
            if self.image_augumentation:
                image = self.image_augumentation(image=image)['image']

            images.append(image)

        return np.stack(images)

    def get_masked_data(self, imageid, image_load_dir, mask_load_dir):
        '''
        Function for loading and masking out the useless car on image
        ------------------------------------------------------------
        '''
        image = np.array(load_img(os.path.join(image_load_dir, imageid+'.jpg'), False, 'rgb', config['ORIGINAL_IMAGE_SIZE'], 'bicubic'))

        try:
            mask = np.array(load_img(os.path.join(mask_load_dir, imageid+'.jpg'), True, 'grayscale', config['ORIGINAL_IMAGE_SIZE'], 'bicubic'))
            mask = cv2.bitwise_not(mask)
            masked_image = cv2.bitwise_and(image, image, mask=mask)
            return masked_image
        except:
            return image

    def __get_regression_gt__(self, SixDofs):
        '''
        Function for collecting the regression ground truth
        ------------------------------------------------------------
        Input:

        6dof ground truth

        Output:

        6dof ground truth in training format
        '''
        _sixdofs = np.zeros((self.batch_size, self.maximum_objects, 5))
        for idx,sd in enumerate(SixDofs):
            for idx2, s in enumerate(sd):
                _sixdofs[idx,idx2,:] = np.array([s['yaw'],
                                                  np.sin(s['pitch']),
                                                  np.cos(s['pitch']),
                                                  rotate(s['roll'],np.pi),
                                                  s['z']/100], dtype=np.float32)
        return _sixdofs

    def h_flip(self, images, SixDofs):
        '''
        Function for performing the horizontal flip on image and ground truth
        ------------------------------------------------------------
        Input:

        input images
        6dof ground truth

        Output:

        fliped images
        fliped 6dof ground truth
        '''
        for idx, image in enumerate(images):
            if random.uniform(0,1) > self.hflip_prob:
                #Images need to be fliped through principal point
                images[idx] = np.concatenate([image[:, self.CX:, :][:, ::-1, :], image[:, :self.CX, :][:,::-1,:]], axis=1)
                for j in range(len(SixDofs[idx])):
                    SixDofs[idx][j]['yaw'] = -SixDofs[idx][j]['yaw']
                    SixDofs[idx][j]['roll'] = -SixDofs[idx][j]['roll']
                    SixDofs[idx][j]['x'] = -SixDofs[idx][j]['x']
        return images, SixDofs

    def __Rotate_image_n_gt__(self, SixDofs, images, tag=['pitch','yaw','roll','x','y','z']):
        '''
        Function for rotating the image and corresponding ground truth(pitch, yaw, roll, x, y and z)
        XYZ axes rotation angle will be decided on random number in pre-defined range
        ------------------------------------------------------------
        Input:

        6Dof ground truth
        input images

        Output:

        Car keypoints coordinate
        Rotated image
        Rotated 6dof
        Keypoint indices(for locating the keypoint coordinate during training)
        Regression mask(mask for discard the unwant ground truths, since maximum objects is 50)
        '''

        keypoints = np.zeros((self.batch_size, self.maximum_objects, 2), dtype = int)
        keypoint_indices = np.zeros((self.batch_size, self.maximum_objects))
        regr_mask = np.zeros((self.batch_size, self.maximum_objects), dtype=np.float32)

        for idx, (image, SixDof) in enumerate(zip(images, SixDofs)):

            rotated_SixDof = []

            x_angle = 0
            y_angle = 0
            z_angle = 0
            if random.uniform(0,1) > self.rot_prob:
                #Random select the angle between -angle to angle, and convert them into radian
                x_angle = np.random.choice(self.X_rot_angle, 1)[0] * np.pi / 180
                y_angle = np.random.choice(self.Y_rot_angle, 1)[0] * np.pi / 180
                z_angle = np.random.choice(self.Z_rot_angle, 1)[0] * np.pi / 180

            #Getting the 2D and 3D rotate transformation matrix
            TransMat, RotMat = rotateImage(x_angle, y_angle, z_angle)
            #Rotate the image
            images[idx] = cv2.warpPerspective(image, TransMat, (self.input_size[1], self.input_size[0]), flags=cv2.INTER_LANCZOS4)
            kps = SIXDOF_2_ImageCoordinate(SixDof)
            #Discard the keypoints outside the original image
            kps, SixDof = self.__discard_outlier__(kps, SixDof, config['ORIGINAL_IMAGE_SIZE'][1], config['ORIGINAL_IMAGE_SIZE'][0])
            kps = np.array([kps[:,1], kps[:,0]-self.Y_cropped, np.ones(len(kps))]).transpose(1,0)

            kps[:,0] *= (self.input_size[1]/config['ORIGINAL_IMAGE_SIZE'][1])
            kps[:,1] *= (self.input_size[0]/(config['ORIGINAL_IMAGE_SIZE'][0]-self.Y_cropped))

            #Rotate the keypoints
            transformed_kps = np.dot(TransMat, kps.T).T
            #Divide the scale factor
            transformed_kps[:,0] /= transformed_kps[:,2]
            transformed_kps[:,1] /= transformed_kps[:,2]
            #Scale down the keypoint to heatmap size
            transformed_kps[:,0] *= (self.output_size[1]/self.input_size[1])
            transformed_kps[:,1] *= (self.output_size[0]/self.input_size[0])
            transformed_kps = transformed_kps[:,[1,0]]

            #Discard the keypoint outside the heatmap size
            transformed_kps, SixDof = self.__discard_outlier__(transformed_kps, SixDof, self.output_size[1], self.output_size[0])

            #Ground truth mask
            regr_mask[idx, :len(transformed_kps)] = 1.0
            #Ground truth keypoint
            keypoints[idx, :len(transformed_kps), :] = transformed_kps
            #Turn ground truth keypoint coordinate in 1D array
            keypoint_indices[idx, :] = (keypoints[idx,:,0] * self.output_size[1]) + keypoints[idx,:,1]
            for c in SixDof:
                pitch, yaw, roll, x, y, z = c['pitch'], c['yaw'], c['roll'], c['x'], c['y'], c['z']
                #Rotate the X, Y and Z ground truths
                x, y, z, _ = np.dot(RotMat, np.array([x,y,z, 1]).T).T
                #Rotate the Pitch, yaw, roll ground truths
                r1 = R.from_euler('xyz', [-yaw, -pitch, -roll], degrees=False)
                r2 = R.from_euler('xyz', [y_angle, -x_angle, -z_angle], degrees=False)
                yaw, pitch, roll = (r2*r1).as_euler('xyz')*-1
                rotated_SixDof.append(dict(zip(tag, np.array([pitch, yaw, roll, x, y, z], dtype='float'))))

            SixDofs[idx] = rotated_SixDof

        return images, SixDofs, keypoints, keypoint_indices, regr_mask

    def __discard_outlier__(self, coords, SixDof, x_upper_bound, y_upper_bound):
        '''
        Function for discarding the coordinates outlier(outside the image)
        '''
        remove_indices =  np.where( ((coords[:,1]<0) | (coords[:,1]>=x_upper_bound) | (coords[:,0]<0) |  (coords[:,0]>=y_upper_bound)) )[0]
        remove_count = 0
        for ri in remove_indices:
            del SixDof[ri-remove_count]
            remove_count += 1
        coords = np.delete(coords, remove_indices , axis=0)

        return coords, SixDof

    def __getkeypoints__keypointindices__(self, SixDofs):
        '''
        Function for generating the keypoint, keypoint indices and regression mask if do_rotate flag is False
        ------------------------------------------------------------
        Input:

        6dof ground truth

        Output:

        keypoints coordinate
        keypoint indices
        regression mask
        '''
        keypoints = np.zeros((self.batch_size, self.maximum_objects, 2), dtype = int)
        keypoint_indices = np.zeros((self.batch_size, self.maximum_objects))
        regr_mask = np.zeros((self.batch_size, self.maximum_objects), dtype=np.float32)
        for idx in range(self.batch_size):
            kps = SIXDOF_2_ImageCoordinate(SixDofs[idx])
            #Discard the keypoint outlier(outside the image)
            kps, SixDofs[idx] = self.__discard_outlier(kps, SixDofs[idx], config['ORIGINAL_IMAGE_SIZE'][1], config['ORIGINAL_IMAGE_SIZE'][0])
            #Scale down the keypoint to heatmap size
            kps[:,1] = np.floor(kps[:,1] / config['ORIGINAL_IMAGE_SIZE'][1] * self.output_size[1]).astype('int')
            kps[:,0] = np.floor((kps[:,0]-self.Y_cropped) / (config['ORIGINAL_IMAGE_SIZE'][1] - self.Y_cropped) * self.output_size[0]).astype('int')
            #Ground truth mask
            regr_mask[idx, :len(kps)] = 1
            #Ground truth keypoint
            keypoints[idx, :len(kps), :] = kps
            #Turn ground truth keypoint into 1D array
            keypoint_indices[idx, :] = (keypoints[idx,:,0] * self.output_size[1]) + keypoints[idx,:,1]

        return keypoints, keypoint_indices, regr_mask

    def __getheatmaps__(self, keypoints, SixDofs):
        '''
        Function for generating the heatmap on keypoints
        The heatmap size is determined by the distances, which is Z
        ------------------------------------------------------------
        Input:

        keypoints coordinate
        6dof ground truth

        Output:

        keypoints heatmap
        '''
        final_heatmap = np.zeros((self.batch_size, self.output_size[0], self.output_size[1], 1))

        for idx, (sixdof, keypoint) in enumerate(zip(SixDofs, keypoints)):
            total_distances = [c['z'] for c in sixdof]
            for kp_idx,distance in enumerate(total_distances):
                #Adaptive sigma value will result in adaptive size heatmap
                #The heatmap size is depends on the distance from car to camera
                sigma = config['SIGMA_BASE'] * distance / 75
                heatmap = self.__drawGaussian2D__(keypoint[kp_idx,1], keypoint[kp_idx,0], sigma)
                final_heatmap[idx, :, :, :] = np.maximum(final_heatmap[idx, :, :, :], heatmap[:,:,:])

        return final_heatmap

    def __drawGaussian2D__(self, center_x, center_y, sigma):
        '''
        Function for drawing gaussian heatmap on desire point with adaptive size
        ------------------------------------------------------------
        Input:

        keypoint coordinate
        sigma for gaussian function

        Output:

        single heatmap on certain point

        '''
        x_grid = np.linspace(0, self.output_size[1]-1, self.output_size[1])
        y_grid = np.linspace(0, self.output_size[0]-1, self.output_size[0])

        x_grid, y_grid = np.meshgrid(x_grid, y_grid)
        x_grid -= center_x
        y_grid -= center_y

        heatmap = np.exp(-((x_grid**2)+(y_grid**2)) / 2*(sigma**2))
        return np.expand_dims(heatmap, axis=2)
