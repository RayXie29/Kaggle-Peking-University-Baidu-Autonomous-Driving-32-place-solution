'''
Prediction evalulation and mAP calculation is referred to :
https://www.kaggle.com/its7171/metrics-evaluation-script
Thanks for tito's share
'''

import argparse
import numpy as np
import pandas as pd
from project_config import *
import matplotlib.pyplot as plt
from callbacks import mAP_callback
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score


def print_pr_curve(result_flg, scores, recall_total=1):
    average_precision = average_precision_score(result_flg, scores)
    precision, recall, _ = precision_recall_curve(result_flg, scores)
    recall *= recall_total
    plt.step(recall, precision, color='b', alpha=0.2, where='post')
    plt.fill_between(recall, precision, alpha=0.2, color='b')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.show()

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_dir', type=str, required=True, help='Evaluation csv file dir')
    parser.add_argument('-f', '--random_score_flag', type=int, default=0, help='Flag(0 or 1) of whether evaluate mAP with random confidence score or not')
    args = vars(parser.parse_args())

    try:
        evaluation_df = pd.read_csv(args['input_dir'])
    except :
        raise ValueError('Evaluation csv file dir error')


    evaluation_df = evaluation_df.fillna('')
    train_csv = pd.read_csv(config['TRAIN_CSV_DIR'])

    map_callback = mAP_callback(None, train_csv, evaluation_df)
    n_gt = len(map_callback.expand_train_df)
    ap_list = []
    for idx in range(10):
        print('calculated th: {}'.format(idx))
        result_flg, scores = map_callback.__check_match__(idx, evaluation_df)
        if args['random_score_flag'] == 1:
            scores = np.random.rand(len(result_flg))
        if np.sum(result_flg) > 0:
            n_tp = np.sum(result_flg)
            recall = n_tp/n_gt
            ap = average_precision_score(result_flg, scores)*recall
            print_pr_curve(result_flg, scores, recall)
        else:
            ap = 0
        ap_list.append(ap)
    map = np.mean(ap_list)
    print('Evaluation mAP:', map)

if __name__ == '__main__':
    main()
