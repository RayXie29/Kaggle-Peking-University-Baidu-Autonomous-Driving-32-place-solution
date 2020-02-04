import tensorflow as tf
from project_config import config

def focal_loss(hm_trues, hm_preds, alpha=1.5, beta=4, epsilon=1e-6):
    '''
    Focal loss for heatmap ground truth and model output tensor
    '''
    pos_mask = tf.cast(tf.equal(hm_trues, 1.0), tf.float32)
    neg_mask = tf.cast(tf.less(hm_trues, 1.0), tf.float32)

    hm_preds = tf.clip_by_value(hm_preds, epsilon, 1.0-epsilon)
    neg_weights = tf.pow(1.0-hm_trues, beta)

    pos_loss = tf.pow(1.0-hm_preds, alpha) * tf.math.log(hm_preds) * pos_mask
    neg_loss = neg_weights * tf.pow(hm_preds,alpha) * tf.math.log(1.0-hm_preds) * neg_mask

    pos_loss = tf.reduce_sum(pos_loss)
    neg_loss = tf.reduce_sum(neg_loss)
    pos_count = tf.reduce_sum(pos_mask)

    return tf.cond(tf.greater(pos_count,0), lambda: -(pos_loss+neg_loss)/pos_count, lambda:-neg_loss)


def reg_l1_loss(regr_trues, regr_preds, kp_true_indices, regr_mask, mask_expand_size=4, epsilon=1e-6):
    '''
    L1 loss for dof ground truth and model output tensor
    '''
    b = tf.shape(regr_preds)[0]
    c = tf.shape(regr_preds)[-1]
    regr_preds = tf.reshape(regr_preds, (b,-1,c))
    kp_true_indices = tf.cast(kp_true_indices, tf.int32)
    regr_preds = tf.gather(regr_preds, kp_true_indices, batch_dims=1)
    regr_mask = tf.expand_dims(regr_mask, axis=-1)
    regr_mask = tf.tile(regr_mask,(1,1,mask_expand_size))
    total_loss = tf.reduce_sum(tf.abs(regr_preds-regr_trues) * regr_mask)
    total_loss = total_loss / (tf.reduce_sum(regr_mask) + epsilon)

    return total_loss

def reg_huber_loss(regr_trues, regr_preds, kp_true_indices, regr_mask, mask_expand_size=4, epsilon=1e-6, delta=2.8):
    '''
    huber loss for dof ground truth and model output tensor
    '''
    b = tf.shape(regr_preds)[0]
    c = tf.shape(regr_preds)[-1]
    regr_preds = tf.reshape(regr_preds, (b,-1,c))
    kp_true_indices = tf.cast(kp_true_indices, tf.int32)
    regr_preds = tf.gather(regr_preds, kp_true_indices, batch_dims=1)
    regr_mask = tf.expand_dims(regr_mask, axis=-1)
    regr_mask = tf.tile(regr_mask,(1,1,mask_expand_size))
    total_loss = tf.where( tf.less_equal( abs(regr_trues-regr_preds), delta ) , 0.5 * tf.pow(regr_trues-regr_preds,2) , delta * (abs(regr_trues-regr_preds)-0.5*delta) )
    total_loss = tf.reduce_sum(total_loss * regr_mask) / (tf.reduce_sum(regr_mask)+epsilon)
    return total_loss

def rot_loss_compute(args):
    '''
    Function for calculating the rotation loss
    '''
    regr_rot_preds, regr_trues, kp_true_indices, regr_mask = args
    regr_rot_trues, regr_depth_trues = tf.split(regr_trues, [4, 1], axis=-1)
    regr_rot_loss = reg_l1_loss(regr_rot_trues, regr_rot_preds, kp_true_indices, regr_mask, 4)
    return regr_rot_loss

def depth_loss_compute(args):
    '''
    Function for calculating the depth loss
    '''
    regr_depth_preds, regr_trues, kp_true_indices, regr_mask = args
    regr_rot_trues, regr_depth_trues = tf.split(regr_trues, [4, 1], axis=-1)
    regr_depth_loss = reg_l1_loss(regr_depth_trues, regr_depth_preds, kp_true_indices, regr_mask, 1)
    return regr_depth_loss

def hm_loss_compute(args):
    '''
    Function for calculating the heatmap loss
    '''
    hm_preds, hm_trues = args

    return focal_loss(hm_trues, hm_preds)

def sum_total_loss(args, hm_weight=0.1, rot_weight=1.25, depth_weight=1.5):
    '''
    Function for suming all the losses(rotation, depth, heatmap) with pre-definced weights
    '''
    regr_rot_loss, regr_depth_loss, hm_loss = args
    return config['HEATMAP_LOSS_WEIGHT']*hm_loss+(1-config['HEATMAP_LOSS_WEIGHT'])*(config['ROTATION_LOSS_WEIGHT']*regr_rot_loss+config['DEPTH_LOSS_WEIGHT']*regr_depth_loss)
