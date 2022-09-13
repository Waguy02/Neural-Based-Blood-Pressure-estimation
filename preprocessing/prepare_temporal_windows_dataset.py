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

from constants import ROOT_DIR, RAW_DATA_DIR

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


def create_windows(win_len,  N_samp, overlap):
    overlap = np.round(overlap*win_len)
    N_samp = N_samp - win_len + 1

    idx_start = np.round(np.arange(0,N_samp, win_len-overlap)).astype(int)
    idx_stop = np.round(idx_start + win_len - 1)

    return idx_start, idx_stop

def prepare_MIMIC_dataset_temporal_windows(DataPath,  NsampPerSubMax:int=2000, NsampMax:int=None, win_len:int=700, win_overlap:float=0.5):

    all_seqs=[]


    SubjectDirs = scandir(DataPath)
    NumSubjects = sum(1 for x in SubjectDirs)
    SubjectDirs = scandir(DataPath)



    SBP_min = 40;
    SBP_max = 200;
    DBP_min = 40;
    DBP_max = 120;

    # 4th order butterworth filter for PPG preprcessing
    b,a = butter(4,[0.5, 8], 'bandpass', fs=PPG_SAMPLE_RATE)



    # loop over all subjects and their files in the source folder
    all_seqs=[]
    subjectID = 0
    N_samp_total=0
    for idx, dirs in enumerate(SubjectDirs):
        print(f'{datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S")}: Processing subject {idx + 1} of {NumSubjects} ({dirs.name}): {N_samp_total} total samples ')


        PPG_RECORD = np.empty((0, win_len))
        OUTPUT = np.empty((0, 2))

        DataFiles = [f for f in listdir(join(DataPath,dirs)) if isfile(join(DataPath, dirs,f)) and f.endswith('.h5')]
        shuffle(DataFiles)

        n_segments=0
        for file in DataFiles:
            try:
                with h5py.File(join(DataPath, dirs, file), "r") as f:
                    data = {}
                    for key in f.keys():
                        data[key] = np.array(f[key]).transpose()
            except TypeError:
                print("could not read file. Skipping.")

            PPG = data['val'][1, :]

            ABP = data['val'][0, :]

            if 'nB2' not in data:
                continue

            sys_p = data['nA2']


            ABP_sys_idx = data['nA3']-1
            ABP_dia_idx = data['nB3']-1

            # create start and stop indizes for time windows
            N_samples = ABP.shape[0]
            win_start, win_stop = create_windows(win_len,  N_samples, win_overlap)
            N_win = len(win_start)
            n_segments += N_win


            ppg_record = np.zeros((N_win, win_len))

            output = np.zeros((N_win, 2))

            # loop over windows
            for i in range(0, N_win):
                idx_start = win_start[i]
                idx_stop = win_stop[i]

                # extract peak idx of the current windows and the corresponding ABP signal values
                peak_idx = np.where(np.logical_and(sys_p >= idx_start, sys_p < idx_stop))
                sys_p_win = sys_p[peak_idx]
                N_sys_p = len(sys_p_win)

                # check if HR is in a plausible range
                if N_sys_p < (win_len/PPG_SAMPLE_RATE/60)*40 or N_sys_p > (win_len/PPG_SAMPLE_RATE/60)*120:
                    output[i,:] = np.nan
                    continue


                ppg_win = PPG[idx_start:idx_stop+1]

                # extract ABP window and fiducial points systolic and diastolic blood pressure
                abp_win = ABP[idx_start:idx_stop+1]

                # sanity check if enough peak values are present and if the number of SBP peaks matches the number of
                # DBP peaks
                ABP_sys_idx_win = ABP_sys_idx[np.logical_and(ABP_sys_idx >= idx_start, ABP_sys_idx < idx_stop)].astype(int)
                ABP_dia_idx_win = ABP_dia_idx[np.logical_and(ABP_dia_idx >= idx_start, ABP_dia_idx < idx_stop)].astype(int)

                if ABP_sys_idx_win.shape[-1] < (win_len/PPG_SAMPLE_RATE/60)*40 or ABP_sys_idx_win.shape[-1] > (win_len/PPG_SAMPLE_RATE/60)*120:
                    output[i, :] = np.nan
                    continue

                if ABP_dia_idx_win.shape[-1] < (win_len/PPG_SAMPLE_RATE//60)*40 or ABP_dia_idx_win.shape[-1] > (win_len/PPG_SAMPLE_RATE/60)*120:
                    output[i, :] = np.nan
                    continue

                try:
                    if len(ABP_sys_idx_win) != len(ABP_dia_idx_win):
                        if ABP_sys_idx_win[0] > ABP_dia_idx_win[0]:
                            ABP_dia_idx_win = np.delete(ABP_dia_idx_win,0)
                        if ABP_sys_idx_win[-1] > ABP_dia_idx_win[-1]:
                            ABP_sys_idx_win = np.delete(ABP_sys_idx_win,-1)

                    ABP_sys_win = ABP[ABP_sys_idx_win]
                    ABP_dia_win = ABP[ABP_dia_idx_win]

                    # check for NaN in ppg_win and abp_win
                    if np.any(np.isnan(abp_win)):
                        output[i, :] = np.nan
                        continue

                    if np.any(np.isnan(ppg_win)):
                            output[i, :] = np.nan
                            continue

                    NN = np.diff(sys_p_win)/PPG_SAMPLE_RATE
                    HR = 60/np.mean(NN)
                    if HR < 50 or HR > 140:
                        output[i, :] = np.nan
                        continue

                    # check for unreasonably large or small RR intervalls
                    if np.any(NN < 0.3) or np.any(NN > 1.4):
                        output[i, :] = np.nan
                        continue

                    # check if any of the SBP or DBP values exceed reasonable vlaues
                    if np.any(np.logical_or(ABP_sys_win < SBP_min, ABP_sys_win > SBP_max)):
                        output[i, :] = np.nan
                        continue

                    if np.any(np.logical_or(ABP_dia_win < DBP_min, ABP_dia_win > DBP_max)):
                        output[i, :] = np.nan
                        continue

                    # check for NaN in the detected SBP and DBP peaks
                    if np.any(np.isnan(ABP_sys_win)) or np.any(np.isnan(ABP_dia_win)):
                        output[i, :] = np.nan
                        continue

                    # calculate the BP ground truth as the median of all SBP and DBP values in the present window
                    BP_sys = np.median(ABP_sys_win).astype(int)
                    BP_dia = np.median(ABP_dia_win).astype(int)

                    # filter the ppg window using a 4th order Butterworth filter

                    ppg_win = filtfilt(b,a, ppg_win)
                    ppg_win = ppg_win - np.mean(ppg_win)
                    ppg_win = ppg_win/np.std(ppg_win)
                    ppg_record[i, :] = ppg_win

                    output[i,:] = [BP_sys, BP_dia]

                    # if number of good samples (not NaN) exceeds maximum number of samples, stop extracting data
                    N_nonNaN = np.count_nonzero(np.isnan(output[0:i+1,0]) == False)
                    if NsampPerSubMax is not None:
                        if OUTPUT.shape[0] + N_nonNaN > 20*NsampPerSubMax:
                            output = np.delete(output,range(i,output.shape[0]), axis=0)


                            ppg_record = np.delete(ppg_record, range(i,ppg_record.shape[0]), axis=0)

                            break
                except:
                    continue

            idx_nans = np.isnan(output[:,0])
            OUTPUT = np.vstack((OUTPUT, output[np.invert(idx_nans),:]))


            PPG_RECORD = np.vstack((PPG_RECORD, ppg_record[np.invert(idx_nans),:]))



            if NsampPerSubMax is not None:
                if OUTPUT.shape[0] >= 20*NsampPerSubMax:
                    break

        if n_segments == 0:
            print(f'skipping')
            continue

        # save data is at least 100 good samples have been extracted
        if OUTPUT.shape[0] > 100:
            if NsampPerSubMax is not None:
                # if maximum number of samples per subject is defined, draw samples randomly
                if OUTPUT.shape[0] > NsampPerSubMax:
                    idx_select = np.random.choice(OUTPUT.shape[0]-1, size=(int(NsampPerSubMax)), replace=False)
                    PPG_RECORD = PPG_RECORD[idx_select,:]
                    OUTPUT = OUTPUT[idx_select,:]

            segments = np.concatenate([PPG_RECORD, OUTPUT], axis=1).tolist()
            N_samp_total+=len(segments)
            all_seqs+=segments
        else:
            print(f'skipping')

        subjectID += 1




    return all_seqs


if __name__ == "__main__":

    TEMPORAL_WINDOWS=False

    input_data_directory= RAW_DATA_DIR
    dataset_output_path= os.path.join(ROOT_DIR,"data/temporal_windows_data_mmic/")



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


    all_seqs=prepare_MIMIC_dataset_temporal_windows(DataPath=input_data_directory)

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