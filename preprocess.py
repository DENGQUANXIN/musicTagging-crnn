#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   preprocess.py
@Time    :   2019/06/20 21:35:15
@Author  :   Four0Eight
@Version :   1.0
@Desc    :   None
'''

# here put the import lib
import os
import librosa
import numpy as np
import pandas as pd
from progress.bar import Bar
# here put the local import lib
import constants as c

def compute_melgram(audio_path):
    src, sr = librosa.load(audio_path, sr=c.SR)  # whole signal
    n_sample = src.shape[0]
    n_sample_fit = int(c.DURA * c.SR)

    if n_sample < n_sample_fit:  # if too short
        src = np.hstack((src, np.zeros((int(c.DURA * c.SR) - n_sample,))))
    elif n_sample > n_sample_fit:  # if too long
        src = src[(n_sample-n_sample_fit)//2:(n_sample+n_sample_fit)//2]
    logam = librosa.amplitude_to_db
    melgram = librosa.feature.melspectrogram
    ret = logam(melgram(y=src, sr=c.SR, hop_length=c.HOP_LEN,
                        n_fft=c.N_FFT, n_mels=c.N_MELS)**2)
    ret = ret[np.newaxis, np.newaxis, :]
    return ret


if __name__ == "__main__":
    df = pd.read_csv(c.ANNOTA_PATH, delimiter='\t')
    audio_paths = df['mp3_path']

    total = len(audio_paths)
    bar = Bar('Processing', max=len(audio_paths), fill='#', suffix='%(percent)d%%')
    for audio_path in audio_paths:
        bar.next()
        save_file = os.path.splitext(audio_path)[0] + '.npy'
        if os.path.exists(os.path.join(c.SAVE_DIR, save_file)):
            print("Skip exsist file:", os.path.join(c.SAVE_DIR, save_file))
            continue
        try:
            melgram = compute_melgram(os.path.join(c.MP3_DIR, audio_path))
            np.save(os.path.join(c.SAVE_DIR, save_file), melgram)
        except Exception as e:
            print(e)
    bar.finish()

