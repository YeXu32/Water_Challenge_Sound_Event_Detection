# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 18:39:38 2023

@author: yexun
"""

from tensorflow import keras
import os
import numpy as np
import utils
import pandas as pd
from numpy import inf
from sklearn import preprocessing
import csv

def load_desc_file(_desc_file):
    _desc_dict = dict()
    skip = True
    for line in open(_desc_file):
        if skip:
            skip = False
            continue
        words = line.strip().split(',')
        name = words[0]
        if name not in _desc_dict:
            _desc_dict[name] = list()
        _desc_dict[name].append([float(words[1]), float(words[2]), __class_labels[float(words[-1])]])
    return _desc_dict


def preprocess_data(_X_test, _seq_len, _nb_ch):
    # split into sequences
    _X_test = utils.split_in_seqs(_X_test, _seq_len)
    #_Y_test = utils.split_in_seqs(_Y_test, _seq_len)

    _X_test = utils.split_multi_channels(_X_test, _nb_ch)
    return _X_test

evaluation_setup_folder = r'C:\Users\yn\Desktop\sed\evaluation_setup'
feat_train_folder = r'C:\Users\yn\Desktop\sed\feat_train'
feat_test_folder = r'C:\Users\yn\Desktop\sed\feat_test'
train_file = os.path.join(evaluation_setup_folder, 'training.csv')
evaluate_file = os.path.join(evaluation_setup_folder, 'test.csv')
train_dict = load_desc_file(train_file)
test_dict = load_desc_file(evaluate_file)
model_r = keras.models.load_model(r'C:\Users\yn\Desktop\sed\models\mon_2023_01_22_00_14_49_model.h5')
test_audio_path = r'C:\Users\yn\Desktop\sed\test_audio'
test_dict = []

# -----------------------------------------------------------------------
# Parameter Setups
# -----------------------------------------------------------------------
is_mono = False
__class_labels = {
    0: 0,
    1: 1
}

nb_ch = 1 if is_mono else 2
seq_len = 256
sr = 48000
nfft = 2048
win_len = nfft
hop_len = int(win_len/2)
frames_1_sec = int(sr/(nfft/2.0))


for audio_filename in os.listdir(test_audio_path):
    name, ext = os.path.splitext(audio_filename)
    name = '"' + name + '"'
    print(name)
    test_dict.append(name)

X_train, Y_train = None, None
for key in train_dict.keys():
    tmp_feat_file = os.path.join(feat_train_folder, '{}_{}.npz'.format(key.strip('"'), 'mon' if is_mono else 'bin'))
    dmp = np.load(tmp_feat_file)  
    tmp_mbe, tmp_label = dmp['arr_0'], dmp['arr_1']
    print(key.strip('"')+ " size is "+str(tmp_mbe.shape))
    if X_train is None:
        X_train, Y_train = tmp_mbe, tmp_label
    else:
        X_train, Y_train = np.concatenate((X_train, tmp_mbe), 0), np.concatenate((Y_train, tmp_label), 0)

scaler = preprocessing.StandardScaler()
X_train[X_train == -inf] = 0
X_train = scaler.fit_transform(X_train)


# -----------------------------------------------------------------------
# Prediction
# -----------------------------------------------------------------------
X_test = None
df_final = pd.DataFrame()
for key in test_dict:
    tmp_feat_file = os.path.join(feat_test_folder, '{}_{}.npz'.format(key.strip('"'), 'mon' if is_mono else 'bin'))
    dmp = np.load(tmp_feat_file)
    tmp_mbe = dmp['arr_0']
    X_test = tmp_mbe
    X_test[X_test == -inf] = 0
    X_test = scaler.transform(X_test)
    X_test= preprocess_data(X_test, seq_len, nb_ch )
    pred_r = model_r.predict(X_test)
    Y_pred_r = reverse_split_in_seqs(pred_r)
    Y_pred_r = np.around(Y_pred_r)
    start_time = 0
    end_time = len(tmp_mbe)*hop_len/sr
    
    time_stamp_raw_1 = 0 
    time_stamp_raw_0 = np.where(Y_pred_r[:,0][:-1] != Y_pred_r[:,0][1:])[0]
    time_stamp_raw_1 = np.where(Y_pred_r[:,1][:-1] != Y_pred_r[:,1][1:])[0]
    time_stamp_0 = time_stamp_raw_0 * hop_len/ sr
    time_stamp_1 = time_stamp_raw_1 * hop_len/ sr
    water = pd.DataFrame()
    data=[]  
    if len(time_stamp_raw_0) != 0:
        for i, x in enumerate(time_stamp_raw_0):
            y= time_stamp_raw_1[i]
            if x > y: 
                water = water.append([Y_pred_r[y][1]])
            else:
                water = water.append([Y_pred_r[x][1]])
        for x in time_stamp_0:
            if len(data) == 0:
                data = [key, 0, str(x)]
                df = pd.DataFrame([data])          
            else:
                temp = [key, tmp, str(x)]
                df = df.append([temp])
            tmp = str(x)
        df = pd.concat([df, water], axis=1)
        df.columns =['"file"', '"start"', '"stop"', '"water"']
        if df['"water"'].iloc[-1] == 1:
            last = 0
        else:
            last = 1
        end = pd.DataFrame([key, df['"stop"'].iloc[-1], end_time, last]).T
        end.columns =['"file"', '"start"', '"stop"', '"water"']
        df = df.append([end])
    else:
        #water = str(Y_pred_r[0][1])
        df = pd.DataFrame([key, 0, end_time, Y_pred_r[0][1]]).T
        df.columns =['"file"', '"start"', '"stop"', '"water"']
    df_final = pd.concat([df_final, df], axis=0) 
df_final['"water"'] = df_final['"water"'].astype(int)

##########################
df_final.to_csv(r'C:\Users\yn\Desktop\sed\test_result.csv',index=False, quoting=csv.QUOTE_NONE)


