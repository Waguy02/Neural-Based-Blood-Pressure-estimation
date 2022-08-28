""" Preprocessing script for PPG data downloaded from the MIMIC-III databse

This script preprocesses the downloaded PPG data. ABP and PPG signals are divided into windows of a defined length and
overlap. Ground truth SBP and DBP values are extracted from the ABP signals. PPG signals are filtered. Different
plausibility and sanity checks are performed to exclude artifacts from the dataset. PPG signal windows and associated
SBP/DBP value pairs are stored in a .h5 file.
All BP values outside a physiologically plausible range of 75 to 165 mmHg
and 40 to 80 mmHg for systolic and diastolic BP, respectively, were discarded. Median
heart rates in each window that exceeded the ranges of 50 to 140 bpm were also rejected.
Input: MIMIC records
Output: PPG signal windows and SBP/DBP values in .h5 file
"""
import csv
import json
import os
from os.path import join, expanduser, isfile, splitext, isdir
from os import listdir, scandir
from random import shuffle
import random
from sys import argv
import datetime
import argparse

import h5py
import numpy as np
import scipy
import sklearn

from joblib import Parallel, delayed
from matplotlib import pyplot as plt
from scipy.signal import butter, freqs, filtfilt, argrelextrema
from sklearn.covariance import EllipticEnvelope
from sklearn.preprocessing import StandardScaler

from constants import ROOT_DIR

""" this function create the window for dividing the ABP and PPG signal
     Arg: sampling frequency, window length, number of samples
     Out: start index and stop index
"""
import heartpy as hp
PPG_SAMPLE_RATE=125
N_FEATURES=21
SBP_min = 40
SBP_max = 200
DBP_min = 40
DBP_max = 120
WIN_LEN=700
def plot(wave):
    plt.plot(wave)
    plt.show()
def resample(segment):
    """
    resample an input wave signal using 1D interpolation

    """
    segment_len=len(segment)
    step = (segment_len - 1) / WIN_LEN
    interpolator = scipy.interpolate.interp1d(list(range(segment_len)), segment)
    segment_resampled = interpolator(np.arange(0, segment_len - 1, step))
    return segment_resampled.tolist()



def create_windows(win_len, N_samp, overlap):
    win_len = win_len * PPG_SAMPLE_RATE
    overlap = np.round(overlap*win_len)
    N_samp = N_samp - win_len + 1

    idx_start = np.round(np.arange(0,N_samp, win_len-overlap)).astype(int)
    idx_stop = np.round(idx_start + win_len - 1)

    return idx_start, idx_stop

def extract_segments(record, n_segments=5000, MAX_WIN=60000):
    waves = []
    sbps = []
    dbps = []

    ppg_record = record[0][:MAX_WIN]
    abp_record = record[1][:MAX_WIN]
    sbp_peaks_idxs = record[2][:MAX_WIN]
    dbp_peaks_idxs = record[3][:MAX_WIN]
    feet=np.squeeze(record[4][:MAX_WIN])

    for i in range(1, len(feet) - 1):
        idx_start = feet[i]
        idx_stop = feet[i + 1]
        segment_len = idx_stop - idx_start
        if segment_len < 70 or segment_len > 130: continue
        try:
            sbp_sys_idx=sbp_peaks_idxs[np.logical_and(sbp_peaks_idxs >= idx_start, sbp_peaks_idxs <= idx_stop)]
            dbp_sys_idx = dbp_peaks_idxs[np.logical_and(dbp_peaks_idxs >= idx_start, dbp_peaks_idxs <= idx_stop)]
            if sbp_sys_idx.shape!=dbp_sys_idx.shape:continue
            sbp = float(abp_record[sbp_sys_idx])
            dbp = float(abp_record[dbp_sys_idx])
            wave = ppg_record[idx_start:idx_stop]

            delta = abs(wave[0] - wave[-1])
            if delta >0.15:continue
            if sbp<SBP_min or sbp>SBP_max:continue
            if dbp<DBP_min or dbp>DBP_max:continue
            if delta >0.15:continue
            sbps.append(sbp)
            dbps.append(dbp)
            waves.append(wave)
        except:
            continue

    if len(waves) < 2: return []
    segments = []
    for i in range(len(waves)):
        wave=waves[i]
        segment = resample(wave)
        if len(segment)!=WIN_LEN:continue
        segment = segment + [sbps[i], dbps[i]]
        segments.append(segment)
    if len(segments) > n_segments: segments = random.sample(segments, n_segments)
    return segments


def extract_single_file_segments( file, segments_per_record=500):
    try:
        with h5py.File(file, "r") as f:
            data = {}
            for key in f.keys():
                data[key] = np.array(f[key]).transpose()
    except TypeError:
        print("could not read file. Skipping.")
        return []
    ABP = data['val'][0]
    b, a = butter(4, 10.5, 'lowpass', fs=PPG_SAMPLE_RATE)
    PPG = data['val'][1]
    PPG = filtfilt(b, a, PPG)
    PPG = PPG - np.mean(PPG)
    PPG = PPG / np.std(PPG)

    if not 'nB2' in data:
        return []

    sbp_peaks_idxs = data['nA3'] - 1
    dbp_peaks_idxs = data['nB3'] - 1
    ppg_peaks_idxs = data["nB2"] - 1

    record = (PPG, ABP, sbp_peaks_idxs, dbp_peaks_idxs, ppg_peaks_idxs)
    segments = extract_segments(record, n_segments=segments_per_record)
    return  segments


def prepare_MIMIC_dataset_windows(DataPath, segments_per_subjects=500):
    SubjectDirs = scandir(DataPath)
    NumSubjects = sum(1 for x in SubjectDirs)
    SubjectDirs = scandir(DataPath)
    all_seqs=[]
    N_samp_total = 0
    for idx, dirs in enumerate(SubjectDirs):
        print(f'{datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S")}: Processing subject {idx+1} of {NumSubjects} ({dirs.name}): {N_samp_total} total samples ')
        DataFiles = [join(DataPath, dirs, f) for f in listdir(join(DataPath,dirs)) if isfile(join(DataPath, dirs,f)) and f.endswith('.h5')]
        shuffle(DataFiles)
        # seqs_slices=Parallel(n_jobs=4)(delayed(extract_single_file_segments)(file) for i, file in enumerate(DataFiles))
        seqs_slices=[extract_single_file_segments(file) for i, file in enumerate(DataFiles)]
        subjects_samples=[]
        for s in seqs_slices:subjects_samples+=s

        # if len(subjects_samples)>segments_per_subjects:
        #     subjects_samples=random.choices(subjects_samples,k=segments_per_subjects)

        N_samp_total+=len(subjects_samples)
        all_seqs+=subjects_samples
    return all_seqs




if __name__ == "__main__":

    TEMPORAL_WINDOWS=False

    input_data_directory= r"F:\Projets\Gaby project\NeuralnetworkBPestimation\Rec_mimic"
    dataset_output_path= "data/windows_data_mmic/"



    if not isdir(dataset_output_path):
        os.makedirs(dataset_output_path)
    train_path = join(dataset_output_path, 'train')
    if not isdir(train_path):
        os.makedirs(train_path)
    eval_path = join(dataset_output_path, 'val')
    if not isdir(eval_path):
        os.makedirs(eval_path)
    test_path = join(dataset_output_path, 'test')
    if not isdir(test_path):
        os.makedirs(test_path)


    all_seqs=prepare_MIMIC_dataset_windows(DataPath=input_data_directory)

    all_seqs=np.stack(all_seqs)


    train_seqs, test_seqs = sklearn.model_selection.train_test_split(all_seqs, train_size=0.8)  # 80% for train and 20% for test
    train_seqs, val_seqs = sklearn.model_selection.train_test_split(train_seqs,
                                                                    test_size=0.1)  # 10% of train for validation
    datas = {"train": train_seqs, "test": test_seqs, "val": val_seqs}
    directories = {"train": train_path, "test": test_path, "val": eval_path}

    for modus in ["train", "test", "val"]:
        seqs = datas[modus]
        if len(seqs) == 0: continue
        with open(os.path.join(directories[modus], "all_subjects.csv"), "w") as input:
            writer = csv.writer(input)
            writer.writerows(seqs)


    with open(os.path.join(dataset_output_path, "dataset_info.json"), "w") as input:
        json.dump({"n_samples": len(all_seqs)}, input)