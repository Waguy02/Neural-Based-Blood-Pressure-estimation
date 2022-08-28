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
import time
from os import listdir, scandir
from os.path import join, isfile, isdir
from random import shuffle

import h5py
import numpy as np
import scipy
import sklearn.model_selection
from joblib import Parallel, delayed
from matplotlib import pyplot as plt
from scipy.signal import butter, filtfilt

from sklearn.preprocessing import StandardScaler

from constants import ROOT_DIR
from my_utils import catchtime

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
    features_dict = {}



    maxima_index = np.argmax(wave)
    next_maxima_index = np.argmax(next_wave)
    cp = (wave.shape[0] - maxima_index) + next_maxima_index+1
    cp = cp / PPG_SAMPLE_RATE
    features_dict["CP"] = cp

    features_points = [], []
    sys_min_index = 0
    dias_min_index = len(wave)-1
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


def extract_ppg20_bench(wave, next_wave):
    features_dict = {}
    computations_times = {}

    with catchtime() as t:
        maxima_index = np.argmax(wave)
        next_maxima_index = np.argmax(next_wave)
        cp = (wave.shape[0] - maxima_index) + next_maxima_index + 1
        cp = cp / PPG_SAMPLE_RATE
    computations_times["cp"] = t()

    features_dict["CP"] = cp

    features_points = [], []
    sys_min_index = 0
    dias_min_index = len(wave) - 1
    targets = [0, 10, 25, 33, 50, 66, 75]

    ampli_min = min(wave[dias_min_index], wave[sys_min_index])
    dias_ampli = wave[maxima_index] - ampli_min
    dias_features = ["DT", "DW10", "DW25", "DW33", "DW50", "DW66", "DW75"]
    sys_features = ["SUT", "SW10", "SW25", "SW33", "SW50", "SW66", "SW755"]

    try:
        with catchtime() as t:
            sys_interpoler = scipy.interpolate.interp1d(wave[sys_min_index:maxima_index],
                                                    list(range(sys_min_index, maxima_index)))
            dias_interpoler = scipy.interpolate.interp1d(wave[maxima_index:dias_min_index],
                                                     list(range(maxima_index, dias_min_index)))

        computations_times["interp"]=t()

        for idx, target in enumerate(targets):
            y = ampli_min + dias_ampli * target / 100

            if idx==0:
                with catchtime() as t:
                    x_dias = dias_interpoler([y])[0] if idx != 0 else dias_min_index
                    d_dias = (x_dias - maxima_index) / PPG_SAMPLE_RATE
                computations_times["st_or_dut"]=t()

                with catchtime() as t:
                    x2 = dias_interpoler([y+0.5*dias_ampli])[0]
                    y2 = (x2 - maxima_index) / PPG_SAMPLE_RATE
                computations_times["sw_or_dw"]=t()


            else:
                x_dias = dias_interpoler([y])[0] if idx != 0 else dias_min_index

            x_sys = sys_interpoler([y])[0] if idx != 0 else sys_min_index
            d_sys = (maxima_index - x_sys) / PPG_SAMPLE_RATE
            d_dias = (x_dias - maxima_index) / PPG_SAMPLE_RATE

            if d_sys > d_dias: return None  # Bad segment

            features_points[0].append(x_dias)
            features_points[1].append(y)
            features_points[0].append(x_sys)
            features_points[1].append(y)

            features_dict[dias_features[idx]] = d_dias
            if idx == 0:
                features_dict[sys_features[idx]] = d_sys
            else:
                features_dict[f"SW_DW_ADD{target}"] = d_sys + d_dias
                features_dict[f"SW_DW_DIV_{target}"] = d_dias / d_sys
    except:
        return None

    features_values = list(features_dict.values())

    features_points[0].append(maxima_index)
    features_points[1].append(wave[maxima_index])

    # plt.plot(wave)
    # plt.scatter(features_points[0], features_points[1])
    # plt.show()
    # plt.cla()
    return features_values,computations_times



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

    #Uncomment to benckmark computation times
    # all_computations_times={"cp":[],"interp":[],"sw_or_dw":[],"st_or_dut":[]}

    for i in range(len(waves) - 1):
        if waves_index[i + 1] != waves_index[i] + 1: continue  # Take contigue sequences only
        wave, next_wave = waves[i], waves[i + 1]
        # Uncomment to benckmark computation times . Use debugger then
        # features=extract_ppg20_bench(wave, next_wave)
        # if not features: continue
        # features, computations_times=features
        # for k in computations_times.keys():
        #     all_computations_times[k].append(computations_times[k])

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




def prepare_MIMIC_dataset_features(DataPath):
    SubjectDirs = scandir(DataPath)
    NumSubjects = sum(1 for x in SubjectDirs)
    SubjectDirs = scandir(DataPath)
    all_seqs=[]
    N_samp_total = 0
    for idx, dirs in enumerate(SubjectDirs):
        print(f'{datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S")}: Processing subject {idx+1} of {NumSubjects} ({dirs.name}): {N_samp_total} total samples ')
        DataFiles = [join(DataPath, dirs, f) for f in listdir(join(DataPath,dirs)) if isfile(join(DataPath, dirs,f)) and f.endswith('.h5')]
        shuffle(DataFiles)

        # seqs_slices=Parallel(n_jobs=2)(delayed(extract_single_file_segments)(file) for i, file in enumerate(DataFiles))
        seqs_slices = [extract_single_file_segments(file) for i, file in enumerate(DataFiles)]

        subjects_samples=[]
        for s in seqs_slices:subjects_samples+=s

        # if len(subjects_samples)>segments_per_subjects:
        #     subjects_samples=random.choices(subjects_samples,k=segments_per_subjects)

        N_samp_total+=len(subjects_samples)
        all_seqs+=subjects_samples
    return all_seqs


def extract_single_file_segments( file, segments_per_record=500):
    try:
        with h5py.File(file, "r") as f:
            data = {}
            for key in f.keys():
                data[key] = np.array(f[key]).transpose()
    except TypeError:
        print("could not read file. Skipping.")
        return []
    PPG = data['val'][1]


    if not 'nB2' in data:
        return []

    ABP = data['val'][0]

    if not 'nB2' in data:
        return []

    sbp_peaks_idxs = data['nA3'] - 1
    dbp_peaks_idxs = data['nB3'] - 1
    ppg_peaks_idxs = data["nB2"] - 1

    record = (PPG, ABP, sbp_peaks_idxs, dbp_peaks_idxs, ppg_peaks_idxs)
    segments = extract_segments(record, n_segments=segments_per_record)
    return  segments


if __name__ == "__main__":
    input_data_directory= r"F:\Projets\Gaby project\NeuralnetworkBPestimation\Rec_mimic"
    dataset_output_path= os.path.join(ROOT_DIR,"data/features_data_mmic/")


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

    if len(all_seqs)>15000:
        all_seqs=random.sample(all_seqs,15000)


    all_seqs=np.stack(all_seqs)
    all_seqs[:,:-2]=StandardScaler().fit_transform(all_seqs[:,:-2])

    train_seqs, test_seqs = sklearn.model_selection.train_test_split(all_seqs, train_size=0.7)  # 70% for train
    test_seqs, val_seqs = sklearn.model_selection.train_test_split(test_seqs,
                                                                    test_size=0.5)  # 15% for test and 15% for val
    datas = {"train": train_seqs, "test": test_seqs, "val": val_seqs}
    directories = {"train": train_path, "test": test_path, "val": eval_path}

    for modus in ["train", "test", "val"]:
        seqs = datas[modus]
        if len(seqs) == 0: continue
        with open(os.path.join(directories[modus], "all_subjects.csv"), "w") as input:
            header = ['CP', 'DT', 'SUT', 'DW10', 'SW_DW_ADD10', 'SW_DW_DIV_10', 'DW25', 'SW_DW_ADD25', 'SW_DW_DIV_25',
                      'DW33', 'SW_DW_ADD33', 'SW_DW_DIV_33', 'DW50', 'SW_DW_ADD50', 'SW_DW_DIV_50', 'DW66',
                      'SW_DW_ADD66', 'SW_DW_DIV_66', 'DW75', 'SW_DW_ADD75', 'SW_DW_DIV_75', 'SBP', 'DBP']
            writer = csv.writer(input)
            writer.writerow(header)
            writer.writerows(seqs)


    with open(os.path.join(dataset_output_path, "dataset_info.json"), "w") as input:
        json.dump({"n_samples": len(all_seqs)}, input)