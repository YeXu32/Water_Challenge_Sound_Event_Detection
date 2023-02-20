# -*- coding: utf-8 -*-
"""
Created on Sun Jan 15 18:42:08 2023

@author: yexun
"""

#####################
# Code blocks taken and adapted from repository: https://github.com/YashNita/sound-event-detection-winning-method
#
# Implementation of the Metrics in the following paper:
# Sharath Adavanne, Pasi Pertila and Tuomas Virtanen, "Sound event detection using spatial features and convolutional recurrent neural network" 
# IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP 2017)
# 
#####################

import numpy as np
import utils
import librosa
import os
import wave
from sklearn import preprocessing
from numpy import inf



def load_audio(filename, mono=True, fs=44100):
    file_base, file_extension = os.path.splitext(filename)
    if file_extension == '.wav':
        _audio_file = wave.open(filename)

        # Audio info
        sample_rate = _audio_file.getframerate()
        sample_width = _audio_file.getsampwidth()
        number_of_channels = _audio_file.getnchannels()
        number_of_frames = _audio_file.getnframes()
        
        # Read raw bytes
        data = _audio_file.readframes(number_of_frames)
        _audio_file.close()

        # Convert bytes based on sample_width
        num_samples, remainder = divmod(len(data), sample_width * number_of_channels)
        if remainder > 0:
            raise ValueError('The length of data is not a multiple of sample size * number of channels.')
        if sample_width > 4:
            raise ValueError('Sample size cannot be bigger than 4 bytes.')

        if sample_width == 3:
            # 24 bit audio
            a = np.empty((num_samples, number_of_channels, 4), dtype=np.uint8)
            raw_bytes = np.fromstring(data, dtype=np.uint8)
            a[:, :, :sample_width] = raw_bytes.reshape(-1, number_of_channels, sample_width)
            a[:, :, sample_width:] = (a[:, :, sample_width - 1:sample_width] >> 7) * 255
            audio_data = a.view('<i4').reshape(a.shape[:-1]).T
        else:
            # 8 bit samples are stored as unsigned ints; others as signed ints.
            dt_char = 'u' if sample_width == 1 else 'i'
            a = np.fromstring(data, dtype='<%s%d' % (dt_char, sample_width))
            audio_data = a.reshape(-1, number_of_channels).T

        if mono:
            # Down-mix audio
            audio_data = np.mean(audio_data, axis=0)

        # Convert int values into float
        audio_data = audio_data / float(2 ** (sample_width * 8 - 1) + 1)

        # Resample
        if fs != sample_rate:
            audio_data = librosa.core.resample(audio_data, sample_rate, fs)
            sample_rate = fs

        return audio_data, sample_rate
    return None, None


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


def extract_mbe(_y, _sr, _nfft, _hop_len, _nb_mel):
    spec, n_fft = librosa.core.spectrum._spectrogram(y=_y, n_fft=_nfft, hop_length=_hop_len, power=1)
    mel_basis = librosa.filters.mel(sr=_sr, n_fft=_nfft, n_mels=_nb_mel)
    return np.log(np.dot(mel_basis, spec))


# ###################################################################
#              Main script starts here
# ###################################################################

is_mono = False
__class_labels = {
    0: 0,
    1: 1
}

# location of data.
evaluation_setup_folder = r'C:\Users\yn\Desktop\sed\evaluation_setup'
audio_folder = r'C:\Users\yn\Desktop\sed\training_audio'

# Output
feat_folder = r'C:\Users\yn\Desktop\sed\feat_train'
utils.create_folder(feat_folder)

# User set parameters
nfft = 2048
win_len = nfft
hop_len = int(win_len/2)
nb_mel_bands = 40
sr = 48000

# -----------------------------------------------------------------------
# Feature extraction and label generation
# -----------------------------------------------------------------------
# Load labels
train_file = os.path.join(evaluation_setup_folder, 'training.csv')
evaluate_file = os.path.join(evaluation_setup_folder, 'val.csv')
desc_dict = load_desc_file(train_file) 
desc_dict.update(load_desc_file(evaluate_file)) # contains labels for all the audio in the dataset

###########################




# Extract features for all audio files, and save it along with labels
for audio_filename in os.listdir(audio_folder):
    audio_file = os.path.join(audio_folder, audio_filename)
    audio_name, audio_extension = os.path.splitext(audio_filename)
    print('Extracting features and label for : {}'.format(audio_filename))
    y, sr = load_audio(audio_file, mono=is_mono, fs=sr)
    mbe = None

    if is_mono:
        mbe = extract_mbe(y, sr, nfft, hop_len,nb_mel_bands).T
    else:
        for ch in range(y.shape[0]):
            mbe_ch = extract_mbe(y[ch, :], sr, nfft, hop_len, nb_mel_bands).T
            if mbe is None:
                mbe = mbe_ch
            else:
                mbe = np.concatenate((mbe, mbe_ch), 1)

    label = np.zeros((mbe.shape[0], len(__class_labels)))
    tmp_data = np.array(desc_dict['"' + str(audio_name) + '"'])
    frame_start = np.floor(tmp_data[:, 0] * sr / hop_len).astype(int)
    frame_end = np.ceil(tmp_data[:, 1] * sr / hop_len).astype(int)
    se_class = tmp_data[:, 2].astype(int)
    for ind, val in enumerate(se_class):
        label[frame_start[ind]:frame_end[ind], val] = 1
    tmp_feat_file = os.path.join(feat_folder, '{}_{}.npz'.format(audio_name, 'mon' if is_mono else 'bin'))
    #np.savez(tmp_feat_file, mbe, label)
    np.savez(tmp_feat_file, mbe)


# -----------------------------------------------------------------------
# Feature Normalization
# -----------------------------------------------------------------------
train_file = os.path.join(evaluation_setup_folder, 'training.csv')
evaluate_file = os.path.join(evaluation_setup_folder, 'val.csv')
train_dict = load_desc_file(train_file)
test_dict = load_desc_file(evaluate_file)

X_train, Y_train, X_test, Y_test = None, None, None, None
for key in train_dict.keys():
    tmp_feat_file = os.path.join(feat_folder, '{}_{}.npz'.format(key.strip('"'), 'mon' if is_mono else 'bin'))
    dmp = np.load(tmp_feat_file)  
    tmp_mbe, tmp_label = dmp['arr_0'], dmp['arr_1']
    print(key.strip('"')+ " size is "+str(tmp_mbe.shape))
    if X_train is None:
        X_train, Y_train = tmp_mbe, tmp_label
    else:
        X_train, Y_train = np.concatenate((X_train, tmp_mbe), 0), np.concatenate((Y_train, tmp_label), 0)


for key in test_dict.keys():
    tmp_feat_file = os.path.join(feat_folder, '{}_{}.npz'.format(key.strip('"'), 'mon' if is_mono else 'bin'))
    dmp = np.load(tmp_feat_file)
    tmp_mbe, tmp_label = dmp['arr_0'], dmp['arr_1']
    if X_test is None:
        X_test, Y_test = tmp_mbe, tmp_label
    else:
        X_test, Y_test = np.concatenate((X_test, tmp_mbe), 0), np.concatenate((Y_test, tmp_label), 0)


# Normalize the training data, and scale the testing data using the training data weights
scaler = preprocessing.StandardScaler()
X_train[X_train == -inf] = 0
X_test[X_test == -inf] = 0

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

normalized_feat_file = os.path.join(feat_folder, 'mbe_{}.npz'.format('mon' if is_mono else 'bin'))
np.savez(normalized_feat_file, X_train, Y_train, X_test, Y_test)
np.savez(normalized_feat_file, X_train, Y_train)
print('normalized_feat_file : {}'.format(normalized_feat_file))
