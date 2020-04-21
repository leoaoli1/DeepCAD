#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Detect arousal
python detect.py detected_arousal deepcad.pth.tar model example_psg.hdf 0.40
"""

# Futures
from __future__ import print_function

# Built-in/Generic Imports
import os
import sys
import argparse
import importlib
import time
import pdb

# Libs
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import robust_scale
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Own modules
import utility


def load_data(path, record):
    """
    Load PSG data from hdf5 files

    Parameters
    ----------
    path: string
        dataset path
    record:
        record ID

    Returns
    ----------
    dictionary:
        each key corresponding to one type of signal. One row corresponding to one PSG record
    """
    # initial empty tensor list
    eeg1 = []
    eeg2 = []
    eeg3 = []
    ecg = []
    target = []
    sleep_stage = []
    sleep_stage_list = ['Wake0', 'Stage1sleep1', 'Stage2sleep2', 'Stage3sleep3', 'REMsleep5', 'Unscored9']
    record_full_path = path + record
    sleep_stage_tmp = []
    with pd.HDFStore(record_full_path) as hdf:
        eeg1_data = hdf.select('EEG1').values
        eeg2_data = hdf.select('EEG2').values
        eeg3_data = hdf.select('EEG3').values
        ecg_data = -hdf.select('EKG').values
        target_tmp = hdf.select('Arousal').values

        eeg1.append(torch.from_numpy(eeg1_data).float())
        eeg2.append(torch.from_numpy(eeg2_data).float())
        eeg3.append(torch.from_numpy(eeg3_data).float())
        ecg.append(torch.from_numpy(ecg_data).float())
        target_tensor = torch.from_numpy(target_tmp)
        target.append(target_tensor)

        for name in sleep_stage_list:
            stage = hdf.select(name).values
            sleep_stage_tmp.append(stage)
        sleep_stage_tmp = np.asarray(sleep_stage_tmp)  # [C, L, N]
        sleep_stage_tensor = torch.from_numpy(sleep_stage_tmp)
        sleep_stage_tensor = torch.transpose(sleep_stage_tensor, 0, 2)  # [C, L, N] => [N, L, C]
        sleep_stage.append(sleep_stage_tensor)
    data_dict = {
        'eeg1': eeg1,
        'eeg2': eeg2,
        'eeg3': eeg3,
        'ecg':  ecg,
        'target': target,
        'stage': sleep_stage
    }
    return data_dict


@torch.no_grad()
def inference(data_dict, t, model, device):
    """
    Parameters
    ----------
    data_dict: dictionary
        data
    t: float
        Decision threshold
    model:
        trained deep learning model. Device: if GPU available use GPU. Otherwise use CPU
    device:
        GPU device id

    Returns
    ----------
    tensor:
        predicted label in 0 or 1 format, predicted probability
    """
    model.eval()
    s_t = time.time()
    local_ecg = data_dict['ecgn']

    model.hidden = model.init_hidden(1)
    local_length = local_ecg.size(2)
    # Reduce memory usage: separate one night record to multiple mini batch.
    segment_length = 60 * 60 * 256  # 1 hr data
    if local_length % segment_length != 0:
        total_local_batch = int(local_length // segment_length) + 1
    else:
        total_local_batch = int(local_length // segment_length)
    for local_batch in range(total_local_batch):
        local_start = local_batch * segment_length
        local_end = (local_batch + 1) * segment_length
        # deal with the last mini batch (if L % segment_L != 0)
        if local_end <= local_length:
            local_segment_ecg = local_ecg[:, :, local_start:local_end]
        else:
            local_segment_ecg = local_ecg[:, :, local_start:]
        single_batch_predict_targets = model(local_segment_ecg.to(device))
        # concatenate one record predict target together
        if local_batch == 0:
            single_predict_targets = single_batch_predict_targets
        else:
            single_predict_targets = torch.cat((single_predict_targets, single_batch_predict_targets), dim=1)
    print("inference time {}s".format(time.time() - s_t))
    # the size of single_predict_targets is (1, L, C)
    # this model use BCEWithLogitsLoss() which include sigmoid function. Here for evaluation,
    # need to use sigmoid to compute probability
    single_predict_target = torch.sigmoid(single_predict_targets)
    single_predict_target_binary = (single_predict_target >= t)

    return single_predict_target_binary.cpu(), single_predict_target.cpu()


def reformat_stage(local_tensor):
    """
    Remove overlapped stage label at the transition from one stage to another stage

    Parameters
    ----------
    local_tensor: tensor
        stage labels

    Returns
    ----------
    tensor:
        cleaned stage labels
    """
    out_tensor = torch.zeros((local_tensor.size(0), local_tensor.size(1), local_tensor.size(2)))
    size = local_tensor.size(3)
    for i in range(size):
        fill_tensor = torch.full((local_tensor.size(0), local_tensor.size(1), local_tensor.size(2)), i)
        out_tensor += torch.where(local_tensor[:, :, :, i] > 0, fill_tensor, local_tensor[:, :, :, i].type(torch.FloatTensor))
        out_tensor = torch.where(out_tensor > i, fill_tensor, out_tensor)  # remove overlap labels
    return out_tensor


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Detect arousal')
    parser.add_argument('save_folder', type=str,
                        help='save folder name')
    parser.add_argument('model', type=str,
                        help='pretrained model name')
    parser.add_argument('pretrained_model', type=str,
                        help='model script')
    parser.add_argument('record', type=str,
                        help='PSG record')
    parser.add_argument('threshold', type=float,
                        help='decision threshold')
    args = parser.parse_args()

    model_lib = importlib.import_module(args.pretrained_model)  # py script lib import

    main_path = './'
    psg_path = './'
    threshold = args.threshold

    save_folder_name = '' + args.save_folder
    save_path = main_path + save_folder_name
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    else:
        sys.exit("Folder exist")

    # load pretrained model
    pretrained = torch.load(main_path + args.model)

    # get GPU device for torch. if GPU available use GPU. Otherwise use CPU
    device_type = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cnn_lstm_model = model_lib.CNNLSTM(device_type).to(device_type)
    cnn_lstm_model.load_weights(device_type, pretrained)
    print(cnn_lstm_model)

    record = args.record

    data_dict = load_data(psg_path, record)
    data_dict = utility.truncate2shortest(data_dict)
    # not necessary to truncate. But it will convert tensor list to one tensor
    data_dict['ecgn'] = torch.from_numpy(robust_scale(torch.unbind(data_dict['ecg'], dim=2)[0].data.numpy(), axis=1)).unsqueeze_(2)
    data_dict = utility.transpose_dict(data_dict, multitensors=False)  # (N, L, C) to (N, C, L)

    test_predict_target, test_predict_value = inference(data_dict, threshold, cnn_lstm_model, device_type)

    # set up list of images for animation
    med = 16
    params = {'legend.fontsize': med,
              'figure.figsize': (18, 14),
              'axes.labelsize': med,
              'axes.titlesize': med,
              'xtick.labelsize': med,
              'ytick.labelsize': med}
    plt.rcParams.update(params)

    # each frame is 31s
    freq = 256
    start_point = 0
    duration = 31
    time = np.arange(0, duration)
    time_256 = np.arange(0, duration * 256)
    subject = 0
    data_dict = utility.transpose_dict(data_dict, multitensors=False)  # (N, C, L) to (N, L, C)
    eeg1 = data_dict['eeg1'].data.numpy()  # shape[N, L, C]
    eeg2 = data_dict['eeg2'].data.numpy()  # shape[N, L, C]
    eeg3 = data_dict['eeg3'].data.numpy()  # shape[N, L, C]
    ecg = data_dict['ecg'].data.numpy()  # shape[N, L, C]
    stage = reformat_stage(data_dict['stage']).data.numpy()
    stage = np.transpose(stage, (0, 2, 1))
    true_target_1 = data_dict['target'].data.numpy()
    predict_value = test_predict_value.data.numpy()
    predict_target = test_predict_target.data.numpy()  # shape[N, L, C]
    for add in np.arange(int(1800/duration)):  # 3600 = 1hr
        end_point = start_point + duration
        eeg1_tmp = eeg1[0, start_point * freq:end_point * freq, 0]
        eeg2_tmp = eeg2[0, start_point * freq:end_point * freq, 0]
        eeg3_tmp = eeg3[0, start_point * freq:end_point * freq, 0]
        ecg_tmp = ecg[0, start_point * freq:end_point * freq, 0]
        stage_tmp = stage[0, start_point:end_point, 0]
        true_target_1_tmp = true_target_1[subject, start_point:end_point, 0]
        predict_value_tmp = predict_value[subject, start_point:end_point, 0]
        predict_target_tmp = predict_target[subject, start_point:end_point, 0]

        # if (1 in predict_target_tmp) or (1 in true_target_1_tmp):
        color = 'blue'

        # set figure
        fig = plt.figure()
        ax1 = fig.add_subplot(6, 1, 1)  # EEG1
        ax2 = fig.add_subplot(6, 1, 2)  # EEG2
        ax3 = fig.add_subplot(6, 1, 3)  # EEG3
        ax4 = fig.add_subplot(6, 1, 4)  # ECG
        ax5 = fig.add_subplot(6, 1, 5)  # probability
        ax6 = fig.add_subplot(6, 1, 6)  # Sleep Stage

        repeated_true = np.repeat(true_target_1_tmp, freq)
        repeated_predict = np.repeat(predict_target_tmp, freq)
        ax1.fill_between(time_256, eeg1_tmp.min(), eeg1_tmp.max(), where=repeated_true > 0,
                         facecolor='blue', alpha=0.2)
        ax1.plot(time_256, eeg1_tmp, color=color)
        ax2.fill_between(time_256, eeg2_tmp.min(), eeg2_tmp.max(), where=repeated_true > 0,
                         facecolor='blue', alpha=0.2)
        ax2.plot(time_256, eeg2_tmp, color=color)
        ax3.fill_between(time_256, eeg3_tmp.min(), eeg3_tmp.max(), where=repeated_true > 0,
                         facecolor='blue', alpha=0.2)
        ax3.plot(time_256, eeg3_tmp, color=color)
        ax4.plot(time, predict_value_tmp, color=color)
        ax5.fill_between(time_256, ecg_tmp.min(), ecg_tmp.max(), where=repeated_predict > 0,
                         facecolor='blue', alpha=0.2)
        ax5.plot(time_256, ecg_tmp, color=color)

        ax6.plot(time, stage_tmp, color=color, drawstyle='steps-pre')

        ax1.set_title('EEG1-Fz/Cz')
        ax2.set_title('EEG2-Cz/Oz')
        ax3.set_title('EEG3-C4/M1')
        ax4.set_title('Arousal Probability')
        ax5.set_title('ECG')
        ax6.set_title('Sleep Stages')

        ax1.get_xaxis().set_visible(False)
        ax2.get_xaxis().set_visible(False)
        ax3.get_xaxis().set_visible(False)
        ax4.get_xaxis().set_visible(False)
        ax5.get_xaxis().set_visible(False)

        ax1.set_ylim(eeg1_tmp.min(), eeg1_tmp.max())
        ax2.set_ylim(eeg2_tmp.min(), eeg2_tmp.max())
        ax3.set_ylim(eeg3_tmp.min(), eeg3_tmp.max())
        ax4.set_ylim(0, 1)
        ax5.set_ylim(ecg_tmp.min(), ecg_tmp.max())

        ax6.set_ylim(-1, 5)
        ax6.set_yticks([0, 1, 2, 3, 4])
        ax6.set_yticklabels(['W', 'N1', 'N2', 'N3', 'R'])
        ax6.grid(axis='y')
        ax6.set_xlabel('time(sec)', fontsize='16')

        plt.savefig(save_path + '/image' + str(add) + '.png', dpi=100)
        plt.close()

        start_point = end_point



