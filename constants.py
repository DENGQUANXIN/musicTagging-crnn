#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   constants.py
@Time    :   2019/06/24 12:14:41
@Author  :   Four0Eight
@Version :   1.0
@Desc    :   None
'''

# here put the import lib
import os

# DIRECTORY
# DATA_DIR = "./data"
DATA_DIR = "/home/doubtd/Documents/datasets/music/MagnaTagATune/data"
ANNOTA_PATH = os.path.join(DATA_DIR, "annotations_final.csv")  # annotations file
MP3_DIR = os.path.join(DATA_DIR, 'mp3')  # directory for saving mp3 format audio
SAVE_DIR = os.path.join(DATA_DIR, 'npys')  # directory for saving features
CHECKPOINT_DIR = "./checkpoints"

# MEL-SPECTROGRAM
SR = 12000
N_FFT = 512
N_MELS = 96
HOP_LEN = 256
DURA = 29.12  # to make it 1366 frame..

# SPLIT DATASET
VAL_RATIO = 0.08
TEST_RATIO = 0.16
TRAIN_RATIO = 1 - VAL_RATIO - TEST_RATIO

# TRAIN and TEST
TAGS = ['choral', 'female voice', 'metal', 'country', 'weird', 'no voice',
        'cello', 'harp', 'beats', 'female vocal', 'male voice', 'dance',
        'new age', 'voice', 'choir', 'classic', 'man', 'solo', 'sitar', 'soft',
        'pop', 'no vocal', 'male vocal', 'woman', 'flute', 'quiet', 'loud',
        'harpsichord', 'no vocals', 'vocals', 'singing', 'male', 'opera',
        'indian', 'female', 'synth', 'vocal', 'violin', 'beat', 'ambient',
        'piano', 'fast', 'rock', 'electronic', 'drums', 'strings', 'techno',
        'slow', 'classical', 'guitar']
BATCH_SIZE = 16
THRESHOLD = 0.5