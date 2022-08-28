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
from matplotlib import pyplot as plt
from scipy.signal import butter, freqs, filtfilt, argrelextrema
from sklearn.covariance import EllipticEnvelope
from sklearn.preprocessing import StandardScaler

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




def prepare_MIMIC_dataset_windows(DataPath, segments_per_record=500):


    SubjectDirs = scandir(DataPath)
    NumSubjects = sum(1 for x in SubjectDirs)
    SubjectDirs = scandir(DataPath)
    all_seqs=[]
    N_samp_total = 0
    for idx, dirs in enumerate(SubjectDirs):
        print(f'{datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S")}: Processing subject {idx+1} of {NumSubjects} ({dirs.name}): {N_samp_total} total samples ')
        DataFiles = [f for f in listdir(join(DataPath,dirs)) if isfile(join(DataPath, dirs,f)) and f.endswith('.h5')]
        shuffle(DataFiles)
        for file in DataFiles:
            try:
                with h5py.File(join(DataPath, dirs, file), "r") as f:
                    data = {}
                    for key in f.keys():
                        data[key] = np.array(f[key]).transpose()
            except TypeError:
                print("could not read file. Skipping.")
            b, a = butter(4, 10.5, 'lowpass', fs=PPG_SAMPLE_RATE)
            PPG = data['val'][1]
            PPG = filtfilt(b, a, PPG)
            PPG = PPG - np.mean(PPG)
            PPG = PPG / np.std(PPG)

            ABP = data['val'][0]
            if 'nB2' not in data:
                continue
            sbp_peaks_idxs = data['nA3'] - 1
            dbp_peaks_idxs = data['nB3'] -1
            ppg_peaks_idxs = data["nB2"] -1
            record=(PPG,ABP,sbp_peaks_idxs,dbp_peaks_idxs,ppg_peaks_idxs)
            segments=extract_segments(record,n_segments=segments_per_record)
            all_seqs+=segments
            N_samp_total+=len(segments)
            #
            # abp_peaks_idxs = data['nA3']-1
            # ABP_dia_idx = data['nB3']-1
            # create start and stop indizes for time windows
            # N_samples = ABP.shape[0]
            # win_start, win_stop = CreateWindows(win_len, fs, N_samples, win_overlap)
            #
            #
            # N_win = len(win_start)
            # N_samp_total += N_win
            #
            # if savePPGData:
            #     ppg_record = np.zeros((N_win, win_len*fs))
            #
            # output = np.zeros((N_win, 2))
            #
            # # loop over windows
            # # for i in range(0, N_win):
            #     idx_start = win_start[i]
            #     idx_stop = win_stop[i]
            #
            #     # extract peak idx of the current windows and the corresponding ABP signal values
            #     peak_idx = np.where(np.logical_and(sys_p >= idx_start, sys_p < idx_stop))
            #     sys_p_win = sys_p[peak_idx]
            #     N_sys_p = len(sys_p_win)
            #
            #     # check if HR is in a plausible range
            #     if N_sys_p < (win_len/60)*40 or N_sys_p > (win_len/60)*120:
            #         output[i,:] = np.nan
            #         continue
            #
            #     if savePPGData:
            #         ppg_win = PPG[idx_start:idx_stop+1]
            #
            #     # extract ABP window and fiducial points systolic and diastolic blood pressure
            #     abp_win = ABP[idx_start:idx_stop+1]
            #
            #     # sanity check if enough peak values are present and if the number of SBP peaks matches the number of
            #     # DBP peaks
            #     ABP_sys_idx_win = abp_peaks_idxs[np.logical_and(abp_peaks_idxs >= idx_start, abp_peaks_idxs < idx_stop)].astype(int)
            #     ABP_dia_idx_win = ABP_dia_idx[np.logical_and(ABP_dia_idx >= idx_start, ABP_dia_idx < idx_stop)].astype(int)
            #
            #     if ABP_sys_idx_win.shape[-1] < (win_len/60)*40 or ABP_sys_idx_win.shape[-1] > (win_len/60)*120:
            #         output[i, :] = np.nan
            #         continue
            #
            #     if ABP_dia_idx_win.shape[-1] < (win_len/60)*40 or ABP_dia_idx_win.shape[-1] > (win_len/60)*120:
            #         output[i, :] = np.nan
            #         continue
            #
            #     if len(ABP_sys_idx_win) != len(ABP_dia_idx_win):
            #         if ABP_sys_idx_win[0] > ABP_dia_idx_win[0]:
            #             ABP_dia_idx_win = np.delete(ABP_dia_idx_win,0)
            #         if ABP_sys_idx_win[-1] > ABP_dia_idx_win[-1]:
            #             ABP_sys_idx_win = np.delete(ABP_sys_idx_win,-1)
            #
            #     ABP_sys_win = ABP[ABP_sys_idx_win]
            #     ABP_dia_win = ABP[ABP_dia_idx_win]
            #
            #     # check for NaN in ppg_win and abp_win
            #     if np.any(np.isnan(abp_win)):
            #         output[i, :] = np.nan
            #         continue
            #
            #     if savePPGData:
            #         if np.any(np.isnan(ppg_win)):
            #             output[i, :] = np.nan
            #             continue
            #
            #     NN = np.diff(sys_p_win)/fs
            #     HR = 60/np.mean(NN)
            #     if HR < 50 or HR > 140:
            #         output[i, :] = np.nan
            #         continue
            #
            #     # check for unreasonably large or small RR intervalls
            #     if np.any(NN < 0.3) or np.any(NN > 1.4):
            #         output[i, :] = np.nan
            #         continue
            #
            #     # check if any of the SBP or DBP values exceed reasonable vlaues
            #     if np.any(np.logical_or(ABP_sys_win < SBP_min, ABP_sys_win > SBP_max)):
            #         output[i, :] = np.nan
            #         continue
            #
            #     if np.any(np.logical_or(ABP_dia_win < DBP_min, ABP_dia_win > DBP_max)):
            #         output[i, :] = np.nan
            #         continue
            #
            #     # check for NaN in the detected SBP and DBP peaks
            #     if np.any(np.isnan(ABP_sys_win)) or np.any(np.isnan(ABP_dia_win)):
            #         output[i, :] = np.nan
            #         continue
            #
            #     # calculate the BP ground truth as the median of all SBP and DBP values in the present window
            #     BP_sys = np.median(ABP_sys_win).astype(int)
            #     BP_dia = np.median(ABP_dia_win).astype(int)
            #
            #     # filter the ppg window using a 4th order Butterworth filter
            #     if savePPGData:
            #         ppg_win = filtfilt(b,a, ppg_win)
            #         ppg_win = ppg_win - np.mean(ppg_win)
            #         ppg_win = ppg_win/np.std(ppg_win)
            #         ppg_record[i, :] = ppg_win
            #
            #     output[i,:] = [BP_sys, BP_dia]
            #
            #     # if number of good samples (not NaN) exceeds maximum number of samples, stop extracting data
            #     N_nonNaN = np.count_nonzero(np.isnan(output[0:i+1,0]) == False)
            #     if NsampPerSubMax is not None:
            #         if OUTPUT.shape[0] + N_nonNaN > 20*NsampPerSubMax:
            #             output = np.delete(output,range(i,output.shape[0]), axis=0)
            #
            #             if savePPGData:
            #                 ppg_record = np.delete(ppg_record, range(i,ppg_record.shape[0]), axis=0)
            #
            #             break
            #
            # idx_nans = np.isnan(output[:,0])
            # OUTPUT = np.vstack((OUTPUT, output[np.invert(idx_nans),:]))
            #
            # if savePPGData:
            #     PPG_RECORD = np.vstack((PPG_RECORD, ppg_record[np.invert(idx_nans),:]))
            #
            # # write record name to txt file for reproducibility
            # with open(RecordsFile, 'a') as f:
            #     f.write(file[0:2] + "/" + file[0:-5]+"\n")
            #
            # if NsampPerSubMax is not None:
            #     if OUTPUT.shape[0] >= 20*NsampPerSubMax:
            #         break

    return all_seqs




if __name__ == "__main__":
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
    # all_seqs[:,:-2]=StandardScaler().fit_transform(all_seqs[:,:-2])

    train_seqs, test_seqs = sklearn.model_selection.train_test_split(all_seqs, train_size=0.8)  # 70% for train
    train_seqs, val_seqs = sklearn.model_selection.train_test_split(train_seqs,
                                                                    test_size=0.1)  # 15% for test and 15% for val
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