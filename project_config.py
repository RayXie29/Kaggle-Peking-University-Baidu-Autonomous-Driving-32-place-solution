import os
import yaml
import numpy as np

def get_config():

    config_dict = {}
    with open('./config.yaml') as f:
        docs = yaml.load_all(f, Loader=yaml.FullLoader)

        for doc in docs:
            for k, v in doc.items():
                config_dict[k] = v

    return config_dict


IntrinsicMatrix = np.array([[2304.5479, 0, 1686.2379],
                            [0, 2305.8757, 1354.9849],
                            [0, 0, 1 ]], dtype = np.float32)

Bad_Image_List = [
    'ID_1a5a10365',
    'ID_4d238ae90',
    'ID_408f58e9f',
    'ID_bb1d991f6',
    'ID_c44983aeb'
    ]

thres_tr_list = [0.1, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01]
thres_ro_list = [50, 45, 40, 35, 30, 25, 20, 15, 10, 5]

config = get_config()
