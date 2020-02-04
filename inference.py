import os
import cv2
import argparse
import numpy as np
import pandas as pd
from project_config import *
import matplotlib.pyplot as plt
from Dataset import InputGenerator
from centernet import CenterUNetplusplus
from sklearn.model_selection import train_test_split
from utills import decode_predictions, clear_predictions, Visual3D

def vis_evaluate_prediction(predict_model, df, eval_size, load_image_dir, load_mask_dir):

    random_idx = np.random.choice(range(len(df)), eval_size)
    eval_df = df.iloc[random_idx]

    eval_generator = InputGenerator(data_df = eval_df,
                                    input_size = config['INPUT_IMAGE_SIZE'],
                                    output_size = config['OUTPUT_SIZE'],
                                    batch_size = 1,
                                    training = False,
                                    image_load_dir = load_image_dir,
                                    mask_load_dir = load_mask_dir)
    images = []
    for idx in range(eval_size):
        imageid = eval_df['ImageId'].iloc[idx]
        images.append(plt.imread(os.path.join(load_image_dir,imageid+'.jpg')))

    predictions = predict_model.predict_generator(eval_generator, verbose=1)

    for prediction, image in zip(predictions,images):
        img = np.zeros((*config['ORIGINAL_IMAGE_SIZE'], 3), dtype=np.uint8)
        image = cv2.resize(image, (config['ORIGINAL_IMAGE_SIZE'][1], (config['ORIGINAL_IMAGE_SIZE'][0]-config['ORIGINAL_Y_CROPPED'])))
        img[[config['ORIGINAL_Y_CROPPED']:] = image
        prediction = decode_predictions(prediction)
        preds = clear_predictions(prediction)
        for pred in preds:
            p, y, r, X, Y, Z, c = pred
            img = Visual3D(img, (p,y,r,X,Y,Z))

        plt.figure(figsize=(12,12))
        plt.imshow(img)
        plt.show()


def predict_test_data(predict_model, df, load_image_dir, load_mask_dir ):

    output_predictions = []
    TESTING_BATCH=500
    for idx in range(0, df.shape[0], TESTING_BATCH):

        batch_indices = range(idx, min(idx+TESTING_BATCH, df.shape[0]))
        b = 4 if len(batch_indices)%4 ==0 else 1

        testing_generator = InputGenerator(data_df = df.iloc[batch_indices],
                                           input_size = config['INPUT_IMAGE_SIZE'],
                                           output_size = config['OUTPUT_SIZE'],
                                           batch_size = b,
                                           training = False,
                                           image_load_dir = config['TEST_IMAGE_DIR'],
                                           mask_load_dir = config['TEST_IMAGE_DIR'])



        predictions = predict_model.predict_generator(testing_generator, verbose=1, workers=-1)

        for prediction in predictions:
            prediction = decode_predictions(prediction)
            preds = clear_predictions(prediction)
            s = []
            for pred in preds:
                for p in pred:
                    s.append(str(p))
            output_predictions.append(' '.join(s))


    return output_predictions


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', type=str, required=True, help='There are 3 modes only: v(visual), e(evaluate mAP) and s(submit)')
    args = vars(parser.parse_args())

    _, predict_centernet, hmloss, rotloss, depthloss = CenterUNetplusplus(config['BACKBONE_NAME'], config['INPUT_IMAGE_SIZE'])

    if config['CENTERNET_PRETRAIN_PATH']:
        predict_centernet.load_weights(config['CENTERNET_PRETRAIN_PATH'])
    else:
        raise ValueError('No pretrain weight path in config file')


    train_csv = pd.read_csv(config['TRAIN_CSV_DIR'])
    train_df, valid_df = train_test_split(train_csv, test_size=0.2, random_state=2019)
    sample_submission_df = pd.read_csv(config['SAMPLE_SUBMISSION_CSV_DIR'])

    if args['mode'] == 'v':
        vis_evaluate_prediction(predict_centernet, valid_df, 6, config['TRAIN_IMAGE_DIR'], config['TRAIN_MASK_DIR'])

    elif args['mode'] == 'e':
        output_predictions = predict_test_data(predict_centernet, valid_df, config['TRAIN_IMAGE_DIR'], config['TRAIN_MASK_DIR'])
        valid_df['PredictionString'] = output_predictions
        valid_df.to_csv(os.path.join(config['OUTPUT_FILE_PATH'], 'evaluation_prediction.csv'), index=False)

    elif args['mode'] == 's':
        output_predictions = predict_test_data(predict_centernet, sample_submission_df, config['TEST_IMAGE_DIR'], config['TEST_MASK_DIR'])
        sample_submission_df['PredictionString'] = output_predictions
        sample_submission_df.to_csv(os.path.join(config['OUTPUT_FILE_PATH'], 'submission.csv'), index=False)

    else:
        print('There are 3 modes only: v(visual), e(evaluate mAP) and s(submit)')

if __name__ == '__main__':
    main()
