'''
ResNet v1 model is referred to classification_models github repo of qubvel.
Also the imagenet pretrain weights are come from the same github repo.
https://github.com/qubvel/classification_models
Thanks for his wonderful work.
'''


import copy
import keras
import collections
import keras.layers as KL
import keras.backend as KB


Model_Params_Tuple = collections.namedtuple(
    'ModelParams',
    ['Name', 'NumBlocks', 'BlockType']
)


_DEFAULT_CONV_PARAMS = {
    'kernel_initializer' : 'he_normal',
    'use_bias' : False,
    'padding' : 'valid'
}

_DEFAULT_BN_PARAMS = {
    'axis' : 3,
    'momentum' : 0.99,
    'epsilon' : 2e-5,
    'center' : True,
    'scale' : True
}

def Load_Weights(model, model_name, classes, include_top):
    print('Load pretrain imagenet weights...')
    if include_top:
        if classes != 1000:
            raise ValueError('Only 1000 classes for include_top == True arg')

        if model_name == 'resnet18':
            model.load_weights('/Users/xiejialun/Desktop/ML/ResNetv1_weights/resnet18_imagenet_1000.h5')
        elif model_name == 'resnet34':
            model.load_weights('/Users/xiejialun/Desktop/ML/ResNetv1_weights/resnet34_imagenet_1000.h5')
        elif model_name == 'resnet50':
            model.load_weights('/Users/xiejialun/Desktop/ML/ResNetv1_weights/resnet50_imagenet_1000.h5')
        elif model_name == 'resnet101':
            model.load_weights('/Users/xiejialun/Desktop/ML/ResNetv1_weights/resnet101_imagenet_1000.h5')
        elif model_name == 'resnet152':
            model.load_weights('/Users/xiejialun/Desktop/ML/ResNetv1_weights/resnet152_imagenet_1000.h5')
        else:
            raise ValueError('No such model {}, please select from following list : {}' .format(model_name, list(MODEL_PARAMS.keys())))
    else:

        if model_name == 'resnet18':
            model.load_weights('/Users/xiejialun/Desktop/ML/ResNetv1_weights/resnet18_imagenet_1000_no_top.h5')
        elif model_name == 'resnet34':
            model.load_weights('/Users/xiejialun/Desktop/ML/ResNetv1_weights/resnet34_imagenet_1000_no_top.h5')
        elif model_name == 'resnet50':
            model.load_weights('/Users/xiejialun/Desktop/ML/ResNetv1_weights/resnet50_imagenet_1000_no_top.h5')
        elif model_name == 'resnet101':
            model.load_weights('/Users/xiejialun/Desktop/ML/ResNetv1_weights/resnet101_imagenet_1000_no_top.h5')
        elif model_name == 'resnet152':
            model.load_weights('/Users/xiejialun/Desktop/ML/ResNetv1_weights/resnet152_imagenet_1000_no_top.h5')
        else:
            raise ValueError('No such model {}, please select from following list : {}' .format(model_name, list(MODEL_PARAMS.keys())))

    return model




def _cn_bn_act(filters, kernel_size=(3,3), strides=(1,1), act_func='relu'):

    def LAYERS(input_x):
        x = KL.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, **_DEFAULT_CONV_PARAMS)(input_x)
        x = KL.BatchNormalization(**_DEFAULT_BN_PARAMS)(x)
        x = KL.Activation(act_func)(x)

        return x

    return LAYERS

def _basic_residual_block(filters, act_func='relu', first_block=False, first_stage=False):

    def BLOCK(input_x):

        if (first_block == True) and (first_stage == False):
            strides=(2,2)
        else:
            strides=(1,1)


        x = KL.BatchNormalization(**_DEFAULT_BN_PARAMS)(input_x)
        x = KL.Activation(act_func)(x)


        if first_block:
            shortcut = KL.Conv2D(filters=filters, kernel_size=(1,1), strides=strides, **_DEFAULT_CONV_PARAMS)(x)
        else:
            shortcut = input_x

        x = KL.ZeroPadding2D(padding=(1,1))(x)
        x = _cn_bn_act(filters=filters, strides=strides, act_func=act_func)(x)
        x = KL.ZeroPadding2D(padding=(1,1))(x)
        x = KL.Conv2D(filters=filters, kernel_size=(3,3), strides=(1,1), **_DEFAULT_CONV_PARAMS)(x)

        x = KL.Add()([x,shortcut])

        return x
    return BLOCK

def _bottleneck_residual_block(filters, act_func='relu', first_block=False, first_stage=False):

    def BLOCK(input_x):

        if (first_block == True) and (first_stage == False):
            strides=(2,2)
        else:
            strides=(1,1)

        x = KL.BatchNormalization(**_DEFAULT_BN_PARAMS)(input_x)
        x = KL.Activation(act_func)(x)

        if first_block:
            shortcut = KL.Conv2D(filters=filters*4, kernel_size=(1,1), strides=strides, **_DEFAULT_CONV_PARAMS)(x)
        else:
            shortcut = input_x

        x = _cn_bn_act(filters=filters, kernel_size=(1,1), strides=(1,1), act_func=act_func)(x)
        x = KL.ZeroPadding2D(padding=(1,1))(x)
        x = _cn_bn_act(filters=filters, strides=strides, act_func=act_func)(x)
        x = KL.Conv2D(filters=filters*4, kernel_size=(1,1), strides=(1,1), **_DEFAULT_CONV_PARAMS)(x)

        x = KL.Add()([x,shortcut])

        return x
    return BLOCK

def ResNet(model_params,
           input_shape=None,
           input_tensor=None,
           include_top=False,
           classes=1000,
           weights='imagenet',
           act_func='relu'):

    if input_tensor == None:
        input_x = KL.Input(shape=input_shape, name='DataInput')
    else:
        if not KB.is_keras_tensor(input_tensor):
            input_x = KL.Input(shape=input_shape, tensor=input_tensor)
        else:
            input_x = input_tensor

    residual_block = model_params.BlockType
    No_Scale_BN_Params = copy.deepcopy(_DEFAULT_BN_PARAMS)
    No_Scale_BN_Params['scale'] = False

    init_filters=64

    x = KL.BatchNormalization(**No_Scale_BN_Params)(input_x)
    x = KL.ZeroPadding2D(padding=(3,3))(x)
    x = _cn_bn_act(filters=init_filters, kernel_size=(7,7), strides=(2,2), act_func=act_func)(x)
    x = KL.ZeroPadding2D(padding=(1,1))(x)

    x = KL.MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='valid')(x)

    several_outputs = []
    for stage, block in enumerate(model_params.NumBlocks):
        filters = init_filters * (2 ** stage)
        for b in range(block):
            x = residual_block(filters=filters,
                               act_func=act_func,
                               first_block=(b==0),
                               first_stage=(stage==0))(x)

        several_outputs.append(x)

    x = KL.BatchNormalization(**_DEFAULT_BN_PARAMS)(x)
    x = KL.Activation(act_func)(x)


    if input_tensor != None:
        inputs = keras.utils.get_source_inputs(input_tensor)
    else:
        inputs = input_x
    temp_model = keras.models.Model(inputs=[inputs], outputs=[x])
    if include_top:
        x = KL.GlobalAveragePooling2D()(x)
        x = KL.Dense(units=classes)(x)
        x = KL.Activation(activation='softmax')(x)
        model = keras.models.Model(inputs=[inputs], outputs=[x])
    else:
        model = keras.models.Model(inputs=[inputs], outputs=several_outputs)

    if weights == 'imagenet':
        Load_Weights(temp_model, model_params.Name, classes, include_top)

        for src_layer, dst_layer in zip(temp_model.layers, model.layers):
            dst_layer.set_weights(src_layer.get_weights())

    return model


MODEL_PARAMS = {
    'resnet18' : Model_Params_Tuple('resnet18', (2,2,2,2), _basic_residual_block),
    'resnet34' : Model_Params_Tuple('resnet34', (3,4,6,3), _basic_residual_block),
    'resnet50' : Model_Params_Tuple('resnet50', (3,4,6,3), _bottleneck_residual_block),
    'resnet101': Model_Params_Tuple('resnet101',(3,4,23,3), _bottleneck_residual_block),
    'resnet152': Model_Params_Tuple('resnet152',(3,8,36,3), _bottleneck_residual_block)
}


def get_resnet_model(model_name = 'resnet18',
                     input_shape = (224,224,3),
                     input_tensor = None,
                     include_top = True,
                     classes = 1000,
                     weights = 'imagenet',
                     act_func = 'relu'):

    if  (model_name == 'resnet18') or (model_name == 'resnet34') or (model_name == 'resnet50') or (model_name == 'resnet101') or (model_name == 'resnet152') :

        model_params = MODEL_PARAMS[model_name]
        return ResNet(model_params, input_shape, input_tensor,
                        include_top, classes, weights, act_func)
    else:
        raise ValueError('No such model {}, please select from following list : {}' .format(model_name, list(MODEL_PARAMS.keys())))

def _Swish(x):
    return x * KB.sigmoid(x)
