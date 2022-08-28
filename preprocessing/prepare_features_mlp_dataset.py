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
import datetime
import json
import os
import random
from os import listdir, scandir
from os.path import join, isfile, isdir
from random import shuffle

import h5py
import numpy as np
import scipy
import sklearn
import sklearn.model_selection
from matplotlib import pyplot as plt
from scipy.signal import butter
from sklearn.preprocessing import StandardScaler

""" this function create the window for dividing the ABP and PPG signal
     Arg: sampling frequency, window length, number of samples
     Out: start index and stop index
"""
PPG_SAMPLE_RATE=125
N_FEATURES=21
SBP_min = 40
SBP_max = 200
DBP_min = 40
DBP_max = 120

def plot(wave):
    plt.plot(wave)
    plt.show()
def extract_ppg20(wave,next_wave):
    maxima_index = np.argmax(wave)
    next_maxima_index = np.argmax(next_wave)

    cp = (wave.shape[0] - maxima_index) + next_maxima_index+1
    cp = cp / PPG_SAMPLE_RATE


    features_points = [], []
    sys_min_index = 0
    dias_min_index = len(wave)-1



    features_dict = {}
    features_dict["CP"] = cp
    targets = [0, 10, 25, 33, 50, 66, 75]

    ampli_min=min(wave[dias_min_index],wave[sys_min_index])
    dias_ampli = wave[maxima_index]-ampli_min
    dias_features = ["DT", "DW10", "DW25", "DW33", "DW50", "DW66", "DW75"]
    sys_features = ["SUT", "SW10", "SW25", "SW33", "SW50", "SW66", "SW755"]

    try:
        sys_interpoler=scipy.interpolate.interp1d(wave[sys_min_index:maxima_index],list(range(sys_min_index,maxima_index)))
        dias_interpoler = scipy.interpolate.interp1d(wave[maxima_index:dias_min_index],list(range(maxima_index,dias_min_index)))
        for idx,target in enumerate(targets):
            y=ampli_min+dias_ampli*target/100
            x_dias=dias_interpoler([y])[0] if idx!=0 else dias_min_index
            x_sys=sys_interpoler([y])[0]  if idx!=0 else sys_min_index
            d_sys= (maxima_index - x_sys)/PPG_SAMPLE_RATE
            d_dias=(x_dias-maxima_index)/PPG_SAMPLE_RATE

            if d_sys >d_dias:return None #Bad segment


            features_points[0].append(x_dias)
            features_points[1].append(y)
            features_points[0].append(x_sys)
            features_points[1].append(y)

            features_dict[dias_features[idx]] = d_dias
            if idx == 0:
                features_dict[sys_features[idx]] = d_sys
            else:
                features_dict[f"SW_DW_ADD{target}"] = d_sys + d_dias
                features_dict[f"SW_DW_DIV_{target}"] = d_dias/d_sys
    except:
        return None

    features_values = list(features_dict.values())



    features_points[0].append(maxima_index)
    features_points[1].append(wave[maxima_index])

    # plt.plot(wave)
    # plt.scatter(features_points[0], features_points[1])
    # plt.show()
    # plt.cla()
    return features_values

def extract_segments(record, n_segments=5000, MAX_WIN=60000):
    waves = []
    sbps = []
    dbps = []
    waves_index = []
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

            sbps.append(sbp)
            dbps.append(dbp)
            waves.append(wave)
            waves_index.append(i)
        except:
            continue

    if len(waves) < 2: return []
    segments = []
    for i in range(len(waves) - 1):
        if waves_index[i + 1] != waves_index[i] + 1: continue  # Take contigue sequences only
        wave, next_wave = waves[i], waves[i + 1]
        features = extract_ppg20(wave, next_wave)
        if not features or len(features) < N_FEATURES: continue
        segment = features + [sbps[i], dbps[i]]
        is_safe = True
        for x in segment:
            if x != x :
                is_safe = False
                break
        if not is_safe:
            continue
        segments.append(segment)

    if len(segments) > n_segments: segments = random.sample(segments, n_segments)

    return segments




def prepare_MIMIC_dataset_features(DataPath, segments_per_record=500):


    SubjectDirs = scandir(DataPath)
    NumSubjects = sum(1 for x in SubjectDirs)
    SubjectDirs = scandir(DataPath)

    # 4th order butterworth filter for PPG preprcessing





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
            # PPG = filtfilt(b, a, PPG)
            # PPG = PPG - np.mean(PPG)
            # PPG = PPG / np.std(PPG)


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
    dataset_output_path= "data/features_data_mmic/"
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


    all_seqs=prepare_MIMIC_dataset_features(DataPath=input_data_directory)

    all_seqs=np.stack(all_seqs)
    all_seqs[:,:-2]=StandardScaler().fit_transform(all_seqs[:,:-2])

    train_seqs, test_seqs = sklearn.model_selection.train_test_split(all_seqs, train_size=0.8)  # 70% for train
    train_seqs, val_seqs = sklearn.model_selection.train_test_split(train_seqs,
                                                                    test_size=0.1)  # 15% for test and 15% for val
    datas = {"train": train_seqs, "test": test_seqs, "val": val_seqs}
    directories = {"train": train_path, "test": test_path, "val": eval_path}

    for modus in ["train", "test", "val"]:
        seqs = datas[modus]
        if len(seqs) == 0: continue
        with open(os.path.join(directories[modus], "all_subjects.csv"), "w") as input:
            header = ['CP', 'DT', 'SUT', 'DW10', 'SW_DW_ADD10', 'SW_DW_DIV_10', 'DW25', 'SW_DW_ADD25', 'SW_DW_DIV_25',
                      'DW33', 'SW_DW_ADD33', 'SW_DW_DIV_33', 'DW50', 'SW_DW_ADD50', 'SW_DW_DIV_50', 'DW66',
                      'SW_DW_ADD66', 'SW_DW_DIV_66', 'DW75', 'SW_DW_ADD75', 'SW_DW_DIV_75', 'SBP', 'DBP']
            writer.writerow(header)
            writer = csv.writer(input)
            writer.writerows(seqs)


    with open(os.path.join(dataset_output_path, "dataset_info.json"), "w") as input:
        json.dump({"n_samples": len(all_seqs)}, input)