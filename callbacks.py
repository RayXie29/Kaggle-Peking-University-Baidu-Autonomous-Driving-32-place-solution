'''
Customize keras callback function to calculate and save mAP during training at end of each epoch
Also can adjust the learning rate depends on the value of mAP change
Prediction evalulation and mAP calculation is referred to :
https://www.kaggle.com/its7171/metrics-evaluation-script
Thanks for tito's share
'''

import copy
import keras
import pandas as pd
import tensorflow as tf
from project_config import *
from Dataset import InputGenerator
from math import sqrt, acos, pi, sin, cos
from scipy.spatial.transform import Rotation as R
from sklearn.metrics import average_precision_score
from utills import str2coords, decode_predictions, clear_predictions


class mAP_callback(keras.callbacks.Callback):

    def __init__(self, eval_model, train_csv, eval_df, patience = 2, checkpoint_path = './model_checkpoint.hdf5', decay_factor=0.5):
        super(mAP_callback, self).__init__()
        '''
        Input :

        eval_model : testing/validating model
        train_csv : training dataframe
        eval_df : evaluation/validation dataframe
        patience : patience for modifying the learning rate if best mAP doesn't improve.
        checkpoint_path : model checkpoint/weights saving path if the model get current best mAP.
        decay_factor : learning rate decay factor if the best mAP doesn't improve within patience epochs.
        '''
        self.train_csv = train_csv
        self.eval_model = eval_model
        self.eval_df = eval_df
        self.checkpoint_path = checkpoint_path
        self.train_csv = self.train_csv[self.train_csv.ImageId.isin(eval_df.ImageId.unique())]
        self.patience_limit = patience
        self.decay_factor = decay_factor

        self.expand_train_df = self.__expand_df__(self.train_csv, ['model_type','pitch','yaw','roll','x','y','z'])
        self.map_history = []
        self.best_map = -1.0
        self.patience = 0.0
    def on_epoch_end(self, epoch, logs=None):

        mAP = self.__calculate_map__()
        print("Epoch : {} , validation mAP : {}".format(epoch, mAP))
        self.map_history.append(mAP)
        if mAP > self.best_map:
            self.best_map = mAP
            print('Saving weights to `{}`...'.format(self.checkpoint_path))
            self.model.save_weights(self.checkpoint_path)
            self.patience = 0
        else:
            self.patience += 1

        if self.patience >= self.patience_limit:

            lr = float(keras.backend.get_value(self.model.optimizer.lr))
            reduced_lr = lr * self.decay_factor
            keras.backend.set_value(self.model.optimizer.lr, reduced_lr)

            print("Epoch : {}, learning rate reduced to : {}".format(epoch, reduced_lr))
            self.patience = 0.0


    def __calculate_map__(self):
        eval_predictions = []
        EVAL_BATCH = 256
        for idx in range(0, self.eval_df.shape[0], EVAL_BATCH):
            batch_indices = range(idx, min(self.eval_df.shape[0], idx+EVAL_BATCH))
            b = 2 if (len(batch_indices)%2) == 0 else 1

            eval_generator = InputGenerator(data_df = self.eval_df.iloc[batch_indices],
                                            input_size = config['INPUT_IMAGE_SIZE'],
                                            output_size = config['OUTPUT_SIZE'],
                                            batch_size = b,
                                            training = False,
                                            image_load_dir = config['TRAIN_IMAGE_DIR'],
                                            mask_load_dir = config['TRAIN_MASK_DIR'])

            predictions = self.eval_model.predict_generator(eval_generator, verbose=2, workers=-1)
            for prediction in predictions:
                prediction = decode_predictions(prediction)
                preds = clear_predictions(prediction)
                s = []
                for pred in preds:
                    for p in pred:
                        s.append(str(p))
                eval_predictions.append(' '.join(s))

        eval_prediction_df = copy.deepcopy(self.eval_df)
        eval_prediction_df['PredictionString'] = eval_predictions
        eval_prediction_df = eval_prediction_df.fillna('')

        n_gt = len(self.expand_train_df)
        ap_list = []

        for idx in range(10):
            result_flg, _ = self.__check_match__(idx, eval_prediction_df)
            scores = np.random.rand(len(result_flg))
            if np.sum(result_flg) > 0:
                n_tp = np.sum(result_flg)
                recall = n_tp/n_gt
                ap = average_precision_score(result_flg, scores)*recall
            else:
                ap = 0

            ap_list.append(ap)
        mAP = np.mean(ap_list)
        return mAP

    def __check_match__(self, idx, eval_prediction_df):
        keep_gt=False
        thre_tr_dist = thres_tr_list[idx]
        thre_ro_dist = thres_ro_list[idx]
        train_dict = {imgID:str2coords(s, names=['carid_or_score', 'pitch', 'yaw', 'roll', 'x', 'y', 'z']) for imgID,s in zip(self.train_csv['ImageId'],self.train_csv['PredictionString'])}
        valid_dict = {imgID:str2coords(s, names=['pitch', 'yaw', 'roll', 'x', 'y', 'z', 'carid_or_score']) for imgID,s in zip(eval_prediction_df['ImageId'],eval_prediction_df['PredictionString'])}
        result_flg = [] # 1 for TP, 0 for FP
        scores = []
        MAX_VAL = 10**10
        for img_id in valid_dict:
            for pcar in sorted(valid_dict[img_id], key=lambda x: -x['carid_or_score']):
                # find nearest GT
                min_tr_dist = MAX_VAL
                min_idx = -1
                for idx, gcar in enumerate(train_dict[img_id]):
                    tr_dist = self.__TranslationDistance__(pcar,gcar)
                    if tr_dist < min_tr_dist:
                        min_tr_dist = tr_dist
                        min_ro_dist = self.__RotationDistance__(pcar,gcar)
                        min_idx = idx

                # set the result
                if min_tr_dist < thre_tr_dist and min_ro_dist < thre_ro_dist:
                    if not keep_gt:
                        train_dict[img_id].pop(min_idx)
                    result_flg.append(1)
                else:
                    result_flg.append(0)
                scores.append(pcar['carid_or_score'])

        return result_flg, scores

    def __TranslationDistance__(self, p,g, abs_dist = False):
        dx = p['x'] - g['x']
        dy = p['y'] - g['y']
        dz = p['z'] - g['z']
        diff0 = (g['x']**2 + g['y']**2 + g['z']**2)**0.5
        diff1 = (dx**2 + dy**2 + dz**2)**0.5
        if abs_dist:
            diff = diff1
        else:
            diff = diff1/diff0
        return diff

    def __RotationDistance__(self, p, g):
        true=[ g['pitch'] ,g['yaw'] ,g['roll'] ]
        pred=[ p['pitch'] ,p['yaw'] ,p['roll'] ]
        q1 = R.from_euler('xyz', true)
        q2 = R.from_euler('xyz', pred)
        diff = R.inv(q2) * q1
        W = np.clip(diff.as_quat()[-1], -1., 1.)

        W = (acos(W)*360)/pi
        if W > 180:
            W = 360 - W
        return W

    def __expand_df__(self, df, PredictionStringCols):
        df = df.dropna().copy()
        df['NumCars'] = [int((x.count(' ')+1)/7) for x in df['PredictionString']]

        image_id_expanded = [item for item, count in zip(df['ImageId'], df['NumCars']) for i in range(count)]
        prediction_strings_expanded = df['PredictionString'].str.split(' ',expand = True).values.reshape(-1,7).astype(float)
        prediction_strings_expanded = prediction_strings_expanded[~np.isnan(prediction_strings_expanded).all(axis=1)]
        df = pd.DataFrame(
            {
                'ImageId': image_id_expanded,
                PredictionStringCols[0]:prediction_strings_expanded[:,0],
                PredictionStringCols[1]:prediction_strings_expanded[:,1],
                PredictionStringCols[2]:prediction_strings_expanded[:,2],
                PredictionStringCols[3]:prediction_strings_expanded[:,3],
                PredictionStringCols[4]:prediction_strings_expanded[:,4],
                PredictionStringCols[5]:prediction_strings_expanded[:,5],
                PredictionStringCols[6]:prediction_strings_expanded[:,6]
            })
        return df
