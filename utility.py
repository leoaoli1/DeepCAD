#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Built-in/Generic Imports
import os
import shutil

# Libs
import numpy as np
from sklearn.preprocessing import robust_scale
import torch


def load_list(path):
    """
    Load records from a folder

    Parameters
    ----------
    path: string
        dataset path

    Returns
    ----------
    list:
        record list
    """
    records = []
    for record_name in os.listdir(path):
        records.append(record_name)
    return records


def truncate2shortest(data_dict):
    """
    Truncate PSG records to same length based on the shortest one

    Parameters
    ----------
    data_dict: dict
        PSG data

    Returns
    ----------
    dict:
        truncate PSG data
    """
    list_target = data_dict['target']
    # find the length of the shortest record by target length
    tensor_shortest_target = min(list_target, key=len)
    length = tensor_shortest_target.size(0)
    for key, in_tensor in data_dict.items():
        tmp = []
        for tensor in in_tensor:
            if key == 'ecg':
                freq = 256
            elif 'eeg' in key:
                freq = 256
            else:
                freq = 1
            tmp.append(tensor[:int(freq*length), :])
        data_dict[key] = torch.stack(tmp)  # stack tensors to one tensor
    return data_dict


def truncate_bptt(data_dict, length):
    """
    Truncate tensor to several segment for BPTT

    Parameters
    ----------
    data_dict: dictionary
    length: int
        segment length

    Returns
    ----------
    dictionary:
        truncate tensors
    """
    for key, in_tensor in data_dict.items():
        if key == 'ecg':
            freq = 256
        elif 'eeg' in key:
            freq = 256
        else:
            freq = 1
        chunk_size = int(length) * freq
        data_dict[key] = torch.split(in_tensor, chunk_size, dim=1)
    return data_dict


def scalar(data_dict):
    """
    Normalized the data

    Parameters
    ----------
    data_dict: dict
        raw data

    Returns
    ----------
    dict:
        scaled data
    """
    for key, in_tensor in data_dict.items():
        if key != 'target' and key != 'stage':
            in_tensor = robust_scale(torch.unbind(in_tensor, dim=2)[0].data.numpy(), axis=1)
            data_dict[key] = torch.from_numpy(in_tensor).unsqueeze_(2)
    return data_dict


def transpose_dict(data_dict, multitensors=True):
    """
    Transpose tensors in dictionary from (N, L, C) to (N, C, L) or from (N, C, L) to (N, L, C)

    Parameters
    ----------
    data_dict: dictionary
    multitensors: bool
        For train state, one list included multiple tensors because of BPTT
        For validation/test state, one list includes one tensor. Default True

    Returns
    ----------
    dictionary:
        transposed tensors
    """
    for key, in_tensor in data_dict.items():
        # print(key)
        if multitensors:
            tmp = []
            for tensor in in_tensor:
                tmp.append(torch.transpose(tensor, 1, 2))
            data_dict[key] = tmp
        else:
            data_dict[key] = torch.transpose(in_tensor, 1, 2)
    return data_dict


def save_checkpoint(iteration, state_dict, best, checkpoint):
    """
    Saves model and training parameters at checkpoint + 'last.pth.tar'. If is_best==True, also saves
    checkpoint + 'best.pth.tar'

    Parameters
    ----------
    iteration: int
        iteration number
    state_dict: dict
        contains model's state_dict, may contain other keys such as epoch, optimizer state_dict
    best: bool
        True if it is the best model seen till now
    checkpoint: string
        save folder
    """
    file_path = checkpoint+'/'+str(iteration+1)+'.pth.tar'  # use os.path.join(path, *path) for cross-platform
    torch.save(state_dict, file_path)
    if best:
        shutil.copyfile(file_path, checkpoint+'/best.pth.tar')
    return None