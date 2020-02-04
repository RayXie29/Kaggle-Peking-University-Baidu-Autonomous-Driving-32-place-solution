import os
import yaml
import numpy as np

def Generate_config_file():
    MAIN_DIR = './Dataset/pku-autonomous-driving/'
    config_dict = {
        'MAIN_DIR' : MAIN_DIR,
        'TRAIN_CSV_DIR' : os.path.join(MAIN_DIR, 'train.csv'),
        'SAMPLE_SUBMISSION_CSV_DIR' : os.path.join(MAIN_DIR, 'sample_submission.csv'),
        'TRAIN_IMAGE_DIR' : os.path.join(MAIN_DIR, 'train_images'),
        'TRAIN_MASK_DIR' : os.path.join(MAIN_DIR, 'train_masks'),
        'TEST_IMAGE_DIR' : os.path.join(MAIN_DIR, 'test_images'),
        'TEST_MASK_DIR' : os.path.join(MAIN_DIR, 'test_masks'),
        'OUTPUT_FILE_PATH' : './OutputFile/',
        'ORIGINAL_IMAGE_SIZE' : (2710, 3384),
        'INPUT_IMAGE_SIZE' : (512, 1536),
        'OUTPUT_SIZE' : (128,384),
        'MAX_OBJECTS' : 50,
        'SIGMA_BASE' : 1,
        'BATCH_SIZE' : 2,
        'EPOCH' : 16,
        'BACKBONE_NAME' : 'efficientnetb2',
        'LEARNING_RATE' : 1e-4,
        'CENTERNET_PRETRAIN_PATH' : None,
        'CP_SAVING_PATH' : './Dataset/checkpoints/efficientnetb2_CenterNet.hdf5',
        'HFLIP_FOR_TRAIN' : True,
        'ROTATE_FOR_TRAIN' : True,
        'X_ROT_ANGLE' : 4,
        'Y_ROT_ANGLE' : 4,
        'Z_ROT_ANGLE' : 4,
        'HFLIP_PROB' : 0.5,
        'ROTATE_PROB' : 0.5,
        'DUPLICATE_CAR_DISTANCE_THRESHOLD' : 15,
        'ORIGINAL_Y_CROPPED' : 1526,
        'HEATMAP_LOSS_WEIGHT' : 0.1,
        'ROTATION_LOSS_WEIGHT' : 1.25,
        'DEPTH_LOSS_WEIGHT' : 1.5,
        'CONFIDENCE_THRESHOLD' : 0.1
    }


    with open('./config.yaml', 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False)

if __name__ == '__main__':
    Generate_config_file()
