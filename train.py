import keras
import pandas as pd
from project_config import *
import albumentations as albu
from callbacks import mAP_callback
from Dataset import InputGenerator
from centernet import CenterUNetplusplus
from sklearn.model_selection import train_test_split

def run():

    train_csv = pd.read_csv(config['TRAIN_CSV_DIR'])
    for bad_image_id in Bad_Image_List:
        train_csv = train_csv[train_csv['ImageId'] != bad_image_id]

    train_df, valid_df = train_test_split(train_csv, test_size=0.2, random_state=2019)
    train_augumentation = albu.Compose([
                                    albu.OneOf([
                                        albu.RandomBrightness(limit=0.25),
                                        albu.RandomContrast(limit=0.3),
                                        albu.RandomGamma(),
                                    ], p=0.5),
                                    albu.GaussNoise(p=0.35),
                                    albu.Blur(blur_limit=5, p=0.25)
    ])


    train_generator = InputGenerator(data_df = train_df,
                                     input_size = config['INPUT_IMAGE_SIZE'],
                                     image_augumentation = train_augumentation,
                                     output_size = config['OUTPUT_SIZE'],
                                     batch_size = config['BATCH_SIZE'],
                                     training = True,
                                     image_load_dir = config['TRAIN_IMAGE_DIR'],
                                     mask_load_dir = config['TRAIN_MASK_DIR'],
                                     do_hflip = config['HFLIP_FOR_TRAIN'],
                                     do_rotate = config['ROTATE_FOR_TRAIN'],
                                     X_rot_angle = config['X_ROT_ANGLE'],
                                     Y_rot_angle = config['Y_ROT_ANGLE'],
                                     Z_rot_angle = config['Z_ROT_ANGLE'],
                                     hflip_prob = config['HFLIP_PROB'],
                                     rot_prob = config['ROTATE_PROB'])

    valid_generator = InputGenerator(data_df = valid_df,
                                     input_size = config['INPUT_IMAGE_SIZE'],
                                     output_size = config['OUTPUT_SIZE'],
                                     batch_size = config['BATCH_SIZE'],
                                     training = True,
                                     image_load_dir = config['TRAIN_IMAGE_DIR'],
                                     mask_load_dir = config['TRAIN_MASK_DIR'],
                                     do_hflip = config['HFLIP_FOR_TRAIN'],
                                     do_rotate = False,
                                     hflip_prob = config['HFLIP_PROB']
                                     )

    centernet, predict_centernet, hm_loss, rot_loss, depth_loss = CenterUNetplusplus(config['BACKBONE_NAME'], (*config['INPUT_IMAGE_SIZE'],3))
    centernet.summary()
    '''
    centernet.add_metric(hm_loss,'hm_loss')
    centernet.add_metric(regr_1_loss,'rot_loss')
    centernet.add_metric(regr_2_loss,'depth_loss')
    '''
    if config['CENTERNET_PRETRAIN_PATH']:
        print('Load centernet pretrain weights...')
        centernet.load_weights(config['CENTERNET_PRETRAIN_PATH'])
        predict_centernet.load_weights(config['CENTERNET_PRETRAIN_PATH'])

    optimizer = keras.optimizers.Adam(lr = config['LEARNING_RATE'])
    centernet.compile(optimizer=optimizer, loss = {'total_loss': lambda y_true, y_pred : y_pred})

    map_callback = mAP_callback(predict_centernet, train_csv, valid_df, checkpoint_path=config['CP_SAVING_PATH'])
    training_callbacks = [map_callback]


    history = centernet.fit_generator(generator=train_generator,
                                      steps_per_epoch = train_generator.__len__(),
                                      epochs=config['EPOCH'],
                                      verbose=1,
                                      callbacks=training_callbacks
                                     )

    centernet.save_weights(config['CP_SAVING_PATH'][:-5] + '_finalcheckpoint' + config['CP_SAVING_PATH'][-5:])

def main():
    run()

if __name__ == '__main__':
    main()
