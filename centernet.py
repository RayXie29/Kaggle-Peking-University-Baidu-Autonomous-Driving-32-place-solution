'''
CenterNet with UNet++ decoder for keras
CenterNet refer to :
https://github.com/xuannianz/keras-CenterNet
https://arxiv.org/pdf/1904.07850.pdf

UNet++ refer to :
https://github.com/MrGiovanni/UNetPlusPlus
'''

import copy
import keras
from utills import *
from losses import *
import keras.layers as KL
import keras.backend as KB
from project_config import *
from resnetv1_model import get_resnet_model, _Swish
from efficientnet_model import get_efficientnet_model
from resnetv1_model import _DEFAULT_CONV_PARAMS, _DEFAULT_BN_PARAMS



def nms(hm_preds, kernel=3):
    '''
    Function for discard the duplicate predictions on same car
    ------------------------------------------------------------
    '''
    hm_max = tf.nn.max_pool2d(hm_preds, (kernel, kernel), strides=1, padding='SAME')
    hm_preds = tf.where(tf.equal(hm_max, hm_preds), hm_preds, tf.zeros_like(hm_preds))

    return hm_preds

def get_keypoint_and_confidence(hm_preds):
    '''
    Function for getting the top-k confidences and indices on heatmap output tensor
    ------------------------------------------------------------
    '''
    hm_preds = nms(hm_preds)
    pred_shape = tf.shape(hm_preds)
    b, w = pred_shape[0], pred_shape[2]
    hm_preds = tf.reshape(hm_preds,(b,-1))
    confidences, indices = tf.nn.top_k(hm_preds, k=config['MAX_OBJECTS'])

    return confidences, indices


def get_4dof(regr_preds, indices):
    '''
    Function for collecting the dof prediction with top-k confidences
    ------------------------------------------------------------
    '''
    pred_shape = tf.shape(regr_preds)
    b, c = pred_shape[0], pred_shape[-1]

    regr_preds = tf.reshape(regr_preds, (b, -1, c))
    regr_preds = tf.gather(regr_preds, indices, batch_dims=1)

    return regr_preds

def reverse_regr(pitch_sin, pitch_cos, roll):
    '''
    Function for decoding the roll, pitch  into original format
    ------------------------------------------------------------
    '''
    roll = rotate(roll, -np.pi)

    pitch_sin_square = tf.pow(pitch_sin, 2)
    pitch_cos_square = tf.pow(pitch_cos, 2)

    pitch_sin = pitch_sin / tf.math.sqrt(pitch_sin_square + pitch_cos_square)
    pitch_cos = pitch_cos / tf.math.sqrt(pitch_sin_square + pitch_cos_square)

    pitch = tf.math.acos(pitch_cos) * tf.math.sign(pitch_sin)

    return pitch, roll

def convert_to_logits(heatmap_preds, epsilon=1e-3):
    '''
    Function for converting the heatmap output tensor into logits
    ------------------------------------------------------------
    '''
    heatmap_preds = tf.clip_by_value(heatmap_preds, epsilon, 1.0-epsilon)

    return tf.math.log(heatmap_preds / (1.0-heatmap_preds))


def reverse_sigmoid_transform(depth_preds, epsilon=1e-3):

    depth_preds = 1.0 / (depth_preds+epsilon) - 1.0
    return depth_preds

def decode(args):

    '''
    Function for decoding the prediction into desire format
    ------------------------------------------------------------
    '''
    regr_1_preds, regr_2_preds, hm_preds = args
    hm_preds = convert_to_logits(hm_preds)

    regr_preds = tf.concat([regr_1_preds, regr_2_preds], axis=-1)

    confidences, indices = get_keypoint_and_confidence(hm_preds)
    fourdofs = get_4dof(regr_preds, indices)

    yaw, pitch_sin, pitch_cos, roll, z = tf.split(fourdofs, [1,1,1,1,1], axis=-1)

    yaw = tf.squeeze(yaw, axis=-1)
    roll = tf.squeeze(roll, axis=-1)
    pitch_sin = tf.squeeze(pitch_sin, axis=-1)
    pitch_cos = tf.squeeze(pitch_cos, axis=-1)
    z = tf.squeeze(z, axis=-1) * 100


    pitch, roll = reverse_regr(pitch_sin, pitch_cos, roll)
    indices = tf.cast(indices, tf.float32)
    return tf.concat([pitch, yaw, roll, z, confidences,indices], axis=-1)


def U_Block(kernel_size=(3,3), filters=None):
    '''
    Function of Upsampling block for UNet++
    ------------------------------------------------------------
    '''
    def block(input_x):
        nonlocal filters
        Conv_Params = copy.deepcopy(_DEFAULT_CONV_PARAMS)
        Conv_Params['padding'] = 'same'

        if filters==None:
            filters = int(input_x._keras_shape[-1]/2)

        x = KL.UpSampling2D(size=(2,2))(input_x)
        x = KL.Conv2D(filters=filters, kernel_size=kernel_size, **Conv_Params)(x)
        x = KL.BatchNormalization(**_DEFAULT_BN_PARAMS)(x)
        x = KL.Activation(_Swish)(x)

        return x

    return block


def H_Block(kernel_size=(3,3), filters=None, name='c01'):
    '''
    Function of Concatenate & Convolution block for UNet++
    ------------------------------------------------------------
    '''
    def block(input_xs):
        nonlocal filters
        Conv_Params = copy.deepcopy(_DEFAULT_CONV_PARAMS)
        Conv_Params['padding'] = 'same'

        x = KL.concatenate(input_xs, name=name+'_concate')
        if filters==None:
            filters = int(x._keras_shape[-1]/2)
        x = KL.Conv2D(filters=filters, kernel_size=kernel_size, **Conv_Params, name=name+'_conv2d')(x)
        x = KL.BatchNormalization(**_DEFAULT_BN_PARAMS, name=name+'_BN')(x)
        x = KL.Activation(_Swish, name=name+'_activation')(x)

        return x
    return block


def get_heatmap_head(name='c01'):
    '''
    Function for creating heatmap head
    ------------------------------------------------------------
    '''
    def head(input_x):

        x = KL.ZeroPadding2D(padding=(1,1))(input_x)
        x = KL.Conv2D(filters=64, kernel_size=(3,3), **_DEFAULT_CONV_PARAMS)(x)
        x = KL.Activation(_Swish)(x)
        hm_output = KL.Conv2D(filters=1, kernel_size=(1,1), **_DEFAULT_CONV_PARAMS, activation='sigmoid', name=name+'_heatmap_output')(x)

        return hm_output

    return head

def get_rotation_head(name='c01'):
    '''
    Function for creating rotation head
    ------------------------------------------------------------
    '''
    def head(input_x):
        x = KL.ZeroPadding2D(padding=(1,1))(input_x)
        x = KL.Conv2D(filters=64, kernel_size=(3,3), **_DEFAULT_CONV_PARAMS)(x)
        x = KL.Activation(_Swish)(x)
        rotation_output = KL.Conv2D(filters=4, kernel_size=(1,1), **_DEFAULT_CONV_PARAMS, name=name+'_rotation_output')(x)

        return rotation_output

    return head

def get_depth_head(name='c01'):
    '''
    Function for creating depth head
    ------------------------------------------------------------
    '''
    def head(input_x):
        x = KL.ZeroPadding2D(padding=(1,1))(input_x)
        x = KL.Conv2D(filters=64, kernel_size=(3,3), **_DEFAULT_CONV_PARAMS)(x)
        x = KL.Activation(_Swish)(x)
        x = KL.Conv2D(filters=1, kernel_size=(1,1), activation='sigmoid', **_DEFAULT_CONV_PARAMS)(x)
        depth_output = KL.Lambda(reverse_sigmoid_transform, name=name+'_depth_output')(x)

        return depth_output

    return head

def CenterUNetplusplus(backbone_name, input_shape, weights='imagenet', freeze_bn=False):
    '''
    Function for creating CenterNet keras model
    ------------------------------------------------------------
    Input:

    backbone_name : name of backbone, now only efficientnet and resnet v1 is allowed
    input_shape : input image size
    weights : Only 'imagenet' or None
    freeze_bn : Flag that determinding whether freeze the batch normalization layers in backbone

    Output:

    CenterNet training model
    CenterNet testing model
    heatmap loss layer output (Can add it into metric to observe the heatmap loss)
    rotation loss layer output (Can add it into metric to observe the rotation loss)
    depth loss layer output (Can add it into metric to observe the depth loss)
    '''

    hm_input = KL.Input(shape=(config['OUTPUT_SIZE'][0],config['OUTPUT_SIZE'][1],1) , name='heatmap_input')
    regr_input = KL.Input(shape=(config['MAX_OBJECTS'], 5), name = 'regr_input')
    regr_mask = KL.Input(shape=(config['MAX_OBJECTS'],), name = 'regr_mask_input')
    kp_indices_input = KL.Input(shape=(config['MAX_OBJECTS'],), name = 'keypoint_indices_input')

    if backbone_name.find('resnet') != -1:
        backbone = get_resnet_model(backbone_name, input_shape=input_shape, include_top=False, weights=weights)
    elif backbone_name.find('efficientnet') != -1:
        backbone = get_efficientnet_model(backbone_name, input_shape=input_shape, include_top=False, weights=weights)

    C20,C30,C40,C50 = backbone.outputs

    if freeze_bn:
        for layer in backbone.layers:

            if isinstance(layer, keras.layers.normalization.BatchNormalization):
                layer.trainable=False

    if backbone_name.find('resnet') != -1:
        C20 = KL.BatchNormalization(**_DEFAULT_BN_PARAMS)(C20)
        C20 = KL.Activation(_Swish)(C20)

        C30 = KL.BatchNormalization(**_DEFAULT_BN_PARAMS)(C30)
        C30 = KL.Activation(_Swish)(C30)

        C40 = KL.BatchNormalization(**_DEFAULT_BN_PARAMS)(C40)
        C40 = KL.Activation(_Swish)(C40)

        C50 = KL.BatchNormalization(**_DEFAULT_BN_PARAMS)(C50)
        C50 = KL.Activation(_Swish)(C50)


    C21 = H_Block(filters=64, name='C21')([C20, U_Block(filters=64)(C30)])
    C31 = H_Block(filters=128, name='C31')([C30, U_Block(filters=128)(C40)])
    C41 = H_Block(filters=256, name='C41')([C40, U_Block(filters=256)(C50)])

    C22 = H_Block(filters=64, name='C22')([C20, C21, U_Block(filters=64)(C31)])
    C32 = H_Block(filters=128, name='C32')([C30, C31, U_Block(filters=128)(C41)])

    C23 = H_Block(filters=64, name='C23')([C20, C21, C22, U_Block(filters=64)(C32)])

    hm_output = get_heatmap_head('C23')(C23)
    rot_output = get_rotation_head('C23')(C23)
    dep_output = get_depth_head('C23')(C23)

    hm_loss = KL.Lambda(hm_loss_compute, name='hm_loss')([hm_output, hm_input])
    rot_loss = KL.Lambda(rot_loss_compute, name='rot_loss')([rot_output, regr_input, kp_indices_input, regr_mask])
    dep_loss = KL.Lambda(depth_loss_compute, name='dep_loss')([dep_output, regr_input, kp_indices_input, regr_mask])
    total_loss = KL.Lambda(sum_total_loss, name='total_loss')([rot_loss, dep_loss, hm_loss])

    pred_head = KL.Lambda(decode, name='decoded_preds')([rot_output, dep_output, hm_output])

    model = keras.models.Model(inputs = [backbone.input, hm_input, regr_input, kp_indices_input, regr_mask], outputs=[total_loss])
    predict_model = keras.models.Model(inputs = [backbone.input], outputs=[pred_head])

    return model, predict_model, hm_loss, rot_loss, dep_loss
