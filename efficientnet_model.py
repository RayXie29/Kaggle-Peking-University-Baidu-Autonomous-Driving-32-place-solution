'''
Efficientnet model for keras.
The original model is called from efficientnet git repo of qubvel.
Also the imagenet of pretrain weight is come from efficientnet git repo of qubvel.
https://github.com/qubvel/efficientnet
Thanks for his wonderful work.
'''
import keras
import efficientnet.keras as efn

def get_efficientnet_model(model_name='efficientnetb0',
                           input_shape = (224,224,3),
                           input_tensor = None,
                           include_top = True,
                           classes = 1000,
                           weights = 'imagenet',
                           ):

    layer_names = [
              'block3a_expand_activation', #C2
              'block4a_expand_activation', #C3
              'block6a_expand_activation', #C4
              'top_activation' #C5
    ]

    Args = {'input_shape' : input_shape,
            'weights' : weights,
            'include_top' : include_top,
            'input_tensor' : input_tensor}

    if model_name == 'efficientnetb0':
        backbone = efn.EfficientNetB0(**Args)

    elif model_name == 'efficientnetb1':
        backbone = efn.EfficientNetB1(**Args)

    elif model_name == 'efficientnetb2':
        backbone = efn.EfficientNetB2(**Args)

    elif model_name == 'efficientnetb3':
        backbone = efn.EfficientNetB3(**Args)

    elif model_name == 'efficientnetb4':
        backbone = efn.EfficientNetB4(**Args)

    elif model_name == 'efficientnetb5':
        backbone = efn.EfficientNetB5(**Args)

    elif model_name == 'efficientnetb6':
        backbone = efn.EfficientNetB6(**Args)

    elif model_name == 'efficientnetb7':
        backbone = efn.EfficientNetB7(**Args)

    else:
        raise ValueError('No such model {}'.format(model_name))


    several_layers = []

    several_layers.append(backbone.get_layer(layer_names[0]).output)
    several_layers.append(backbone.get_layer(layer_names[1]).output)
    several_layers.append(backbone.get_layer(layer_names[2]).output)
    several_layers.append(backbone.get_layer(layer_names[3]).output)

    model = keras.models.Model(inputs=[backbone.input], outputs=several_layers)
    return model
