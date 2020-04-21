#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Train model
python train.py project_path dataset_path model_folder n_epochs n_batch lr weight_decay
"""
# Futures
from __future__ import print_function

# Built-in/Generic Imports
import os
import sys
import time
import warnings
import random
import argparse

# Libs
import numpy as np
import pandas as pd
from sklearn import model_selection, utils
from sklearn.metrics import roc_auc_score, average_precision_score
import torch
import torch.nn as nn
import torch.optim as optim

#
import utility
import model


def load_data(path, records):
    """
    Load PSG data from hdf5 files

    Parameters
    ----------
    path: string
        dataset path
    records: list
        record name list

    Returns
    ----------
    dictionary:
        each key is corresponding to one type of signal/label.
        One row is corresponding to one PSG record
    """
    ecg = []
    target = []
    for record in records:
        record_full_path = path + record
        with pd.HDFStore(record_full_path) as hdf:
            ecg_data = -hdf.select('EKG').values
            target_tmp = hdf.select('Arousal').values
            # extract ECG signal from 30s before the first arousal to 30s after the last arousal
            target_index = np.where(target_tmp == 1)[0]
            start_index = target_index[0]
            end_index = target_index[-1]
            if start_index - 30 > 0:
                start_index -= 30
            else:
                start_index = 0
            if end_index + 30 < target_tmp.size:
                end_index += 30
            else:
                end_index = target_tmp.size - 1
            ecg_data = ecg_data[start_index*256:end_index*256]  # 256Hz ECG
            target_tmp = target_tmp[start_index:end_index]
            ecg.append(torch.from_numpy(ecg_data).float())
            target_tensor = torch.from_numpy(target_tmp)
            target.append(target_tensor)
    data_dict = {
        'ecg':  ecg,
        'target': target
    }
    return data_dict


def train(path, records, batch_size, model, optimizer, criterion, device):
    """
    Model training

    Parameters
    ----------
    path: str
        dataset path
    records: list
        record name.
    batch_size: int
    model:
        deep learning model
    optimizer:
        Optimizer algorithm
    criterion:
        Loss function
    device:
        GPU id

    Returns
    ----------
    float:
        training loss, AUPRC, AUROC
    """
    model.train()  # update batch normal weight
    running_loss = 0.0
    running_auprc = 0
    running_auroc = 0
    i = 0

    local_total_batch = int(np.floor(len(records) / batch_size))
    print('Total Batches {}'.format(local_total_batch))
    for batch in range(local_total_batch):
        s_t = time.time()
        train_dict = load_data(path, records[(batch+1) * batch_size - batch_size: (batch+1) * batch_size])
        train_dict = utility.truncate2shortest(train_dict)  # truncate to the same size
        train_dict = utility.scalar(train_dict)
        train_dict = utility.truncate_bptt(train_dict, 90)
        model.hidden = model.init_hidden(batch_size)
        train_dict = utility.transpose_dict(train_dict, multitensors=True)  # (N, L, C) to (N, C, L)
        for (ecg, targets) \
                in zip(train_dict['ecg'], train_dict['target']):
            i += 1
            optimizer.zero_grad()
            model.hidden[0].detach_()
            model.hidden[1].detach_()
            predict_targets = model(ecg.to(device))
            targets = torch.transpose(targets, 1, 2)  # (N, C, L) to (N, L, C)
            loss = criterion(predict_targets, targets.float().to(device))
            loss.backward()
            optimizer.step()

            predict_targets = torch.sigmoid(predict_targets)
            # convert size to (N*L, C)
            flatted_predict_targets = predict_targets.view(-1, predict_targets.size(2)).cpu()
            # convert size to (N*L, C) and contiguous blocks on memory (hardware).
            # The contiguous is only for PyTorch computation because the addresses were blocked in the transpose step.
            flatted_targets = targets.contiguous().view(-1, targets.size(2)).cpu()
            flatted_predict_targets = flatted_predict_targets.data.numpy()
            flatted_targets = flatted_targets.data.numpy()

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    auroc = roc_auc_score(flatted_targets, flatted_predict_targets)
                    auprc = average_precision_score(flatted_targets, flatted_predict_targets)
                except ValueError:
                    auroc = 0
                    auprc = 0
                    i -= 1

            running_auprc += auprc
            running_auroc += auroc
            running_loss += loss.item()
        print("Batch {} training time {}".format(batch, time.time() - s_t))
    return running_loss/i, running_auprc/i, running_auroc/i


@torch.no_grad()
def validation(data_dict, model, criterion, device):
    """
    Parameters
    ----------
    data_dict: dictionary
        validation EEG PSG data Device: On CPU
    model:
        deep learning model. Device: if GPU available use GPU. Otherwise use CPU
    criterion:
        Loss function
    device:
        GPU id

    Returns
    ----------
    float:
        loss, AUPRC, AUROC
    ndarray:
        true label, predicted label
    """
    model.eval()
    i = 0
    # each key points to a (N, C, L) tensor. Iterate based on N
    for ecg in data_dict['ecg']:
        model.hidden = model.init_hidden(1)
        # the output tensor is 2 dimension (C, L). need to unsqueeze to 3 dimension (N, C, L)
        local_ecg = ecg.unsqueeze_(0)
        local_length = local_ecg.size(2)
        # Reduce memory usage: separate one night record to multiple mini batch.
        segment_length = 60*60*256  # 1 hr data
        if local_length % segment_length != 0:
            total_local_batch = int(local_length//segment_length)+1
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
        # if infer the first record, initial the predict_targets. Then keep append to the predict_targets
        # the size of single_predict_targets is (1, L, C)
        if i == 0:
            predict_targets = single_predict_targets

        else:
            predict_targets = torch.cat((predict_targets, single_predict_targets), dim=0)
        i += 1

    targets = torch.transpose(data_dict['target'], 1, 2)  # (N, C, L) to (N, L, C)
    loss = criterion(predict_targets, targets.float().to(device))

    # this model use BCEWithLogitsLoss() which include sigmoid function. Here for evaluation,
    # we need to use sigmoid to compute probability
    predict_targets = torch.sigmoid(predict_targets)

    flatted_predict_targets = predict_targets.view(-1, predict_targets.size(2)).cpu()
    flatted_targets = targets.contiguous().view(-1, targets.size(2)).cpu()
    # convert tensor to numpy array
    flatted_predict_targets = flatted_predict_targets.data.numpy()
    flatted_targets = flatted_targets.data.numpy()
    try:
        auroc = roc_auc_score(flatted_targets, flatted_predict_targets)
        auprc = average_precision_score(flatted_targets, flatted_predict_targets)
    except ValueError:
        auroc = 0
        auprc = 0
    return loss.item(), auroc, auprc, flatted_targets, flatted_predict_targets


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('project_path', type=str,
                        help='project path')
    parser.add_argument('dataset_path', type=str,
                        help='training set path (include validation set)')
    parser.add_argument('model_folder', type=str,
                        help='save model path')
    parser.add_argument('n_epochs', type=int,
                        help='number of training epochs')
    parser.add_argument('n_batch', type=int,
                        help='batch size')
    parser.add_argument('lr', type=float,
                        help='learning rate')
    parser.add_argument('weight_decay', type=float,
                        help='l2 weight decay')
    args = parser.parse_args()

    # set torch print limit to 10000 rows
    torch.set_printoptions(threshold=10000)

    main_path = args.project_path
    dataset_path = args.dataset_path

    # set random seed
    random_seed = 42
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    random.seed(random_seed)
    os.environ['PYTHONHASHSEED'] = str(random_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # The folder will be used to save models
    save_folder_name = args.model_folder
    save_path = main_path + save_folder_name

    # get GPU device for torch. if GPU available use GPU. Otherwise use CPU
    device_type = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # call model class
    cnn_lstm_model = model.CNNLSTM(device_type).to(device_type)
    print(cnn_lstm_model)

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    else:
        pretrained = torch.load(save_path+'/best.pth.tar')
        cnn_lstm_model.load_weights(device_type, pretrained)
        print(cnn_lstm_model)
        save_path = save_path+'continue'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        else:
            sys.exit("Continue folder exist")

    if not os.path.exists(main_path+'train_ID.csv') or not os.path.exists(main_path+'val_ID.csv'):
        # load train records ID
        record_list = utility.load_list(dataset_path)
        arr_records = np.asarray(record_list)
        train_records, val_records = model_selection.train_test_split(arr_records, test_size=0.1, shuffle=True,
                                                                      random_state=random_seed)
        np.savetxt(main_path + 'train_ID.csv', train_records, delimiter=',', fmt='%s')
        np.savetxt(main_path + 'val_ID.csv', val_records, delimiter=',', fmt='%s')
    else:
        train_records = np.loadtxt(main_path + 'train_ID.csv', delimiter=',', dtype=np.unicode_)
        val_records = np.loadtxt(main_path + 'val_ID.csv', delimiter=',', dtype=np.unicode_)

    # initial training parameters
    n_epochs = args.n_epochs  # number of training iterations
    n_batches = args.n_batch
    max_pr_auc = 0
    loss_func = nn.BCEWithLogitsLoss().to(device_type)

    # initial optimizer
    opt = optim.Adam(cnn_lstm_model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08,
                     weight_decay=args.weight_decay, amsgrad=False)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, 'max', patience=4)
    print('-' * 30)
    print('Load validation data')
    print('%d validation records' % val_records.size)
    val_dict = load_data(dataset_path, val_records)
    val_dict = utility.truncate2shortest(val_dict)
    val_dict = utility.scalar(val_dict)
    val_dict = utility.transpose_dict(val_dict, multitensors=False)  # (N, L, C) to (N, C, L)
    print('Start Training')
    print('%d train records' % train_records.size)
    for epoch in range(n_epochs):
        start_time = time.time()
        # shuffle train records in each epoch
        train_records = utils.shuffle(train_records, random_state=random_seed)
        train_loss, train_auprc, train_auroc = train(dataset_path, train_records, n_batches, cnn_lstm_model, opt,
                                                     loss_func, device_type)
        val_loss, roc_auc, pr_auc, val_true_target, val_predict_target \
            = validation(val_dict, cnn_lstm_model, loss_func, device_type)
        scheduler.step(pr_auc)
        print('Epoch {}/{}:\n training loss {:.4f},'
              'val loss {:.4f}, val ROC AUC {:.2f}, val PR AUC {:.2f}'
              .format(epoch, n_epochs - 1, train_loss, val_loss, roc_auc, pr_auc))
        print('--- {:.2f} seconds ---'.format(time.time() - start_time))

        # checkpoint save model
        print('Model Checkpoint')
        state = {'epoch': epoch + 1,
                 'state_dict': cnn_lstm_model.state_dict(),
                 'opt_dict': opt.state_dict()}
        if pr_auc > max_pr_auc:
            max_pr_auc = pr_auc
            is_best = True
        else:
            is_best = False
        utility.save_checkpoint(epoch, state, best=is_best, checkpoint=save_path)
        print('-' * 20)
