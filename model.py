#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Built-in/Generic Imports
from collections import OrderedDict

# Libs
import torch
import torch.nn as nn


class Block(nn.Module):
    def __init__(self, inchannel, outchannel, device, kernel_size=1, stride=1, padding=0, shortcut=None):
        super(Block, self).__init__()
        self.device = device
        self.left = nn.Sequential(OrderedDict([
            ('bn1', nn.BatchNorm1d(inchannel)),
            ('relu1', nn.ReLU()),
            ('conv1', nn.Conv1d(inchannel, outchannel, kernel_size, stride, padding, bias=False)),
            ('bn2', nn.BatchNorm1d(outchannel)),
            ('relu2', nn.ReLU()),
            ('conv2', nn.Conv1d(outchannel, outchannel, 7, 1, 3, bias=False)),
        ]))
        self.right = shortcut

    def forward(self, x):
        out = self.left(x)
        if self.right is None:
            residual = x
        else:
            residual = self.right(x)
            padding = torch.zeros((residual.size(0), 32, residual.size(2)), requires_grad=True).to(self.device)
            residual = torch.cat((residual, padding), 1)
        out += residual
        return out


class CNNLSTM(nn.Module):
    def __init__(self, device):
        super(CNNLSTM, self).__init__()
        self.model_name = 'cnn_lstm'
        self.device = device
        self.lstm_layers = 2
        self.output_classes = 1

        self.layer_1_conv1 = nn.Conv1d(1, 8, 11, 1, 5, bias=False)
        self.layer_1_conv2 = nn.Conv1d(1, 8, 15, 1, 7, bias=False)
        self.layer_1_conv3 = nn.Conv1d(1, 8, 19, 1, 9, bias=False)
        self.layer_1_conv4 = nn.Conv1d(1, 8, 23, 1, 11, bias=False)

        self.block_1 = self._make_layer(32, 64, 2, self.device, kernel_size=2, stride=2, padding=0)
        self.block_2 = self._make_layer(64, 96, 2, self.device, kernel_size=2, stride=2, padding=0)
        self.block_3 = self._make_layer(96, 128, 2, self.device, kernel_size=2, stride=2, padding=0)
        self.block_4 = self._make_layer(128, 160, 2, self.device, kernel_size=2, stride=2, padding=0)
        self.block_5 = self._make_layer(160, 192, 2, self.device, kernel_size=2, stride=2, padding=0)
        self.block_6 = self._make_layer(192, 224, 2, self.device, kernel_size=2, stride=2, padding=0)
        self.block_7 = self._make_layer(224, 256, 2, self.device, kernel_size=2, stride=2, padding=0)
        self.block_8 = self._make_layer(256, 288, 2, self.device, kernel_size=2, stride=2, padding=0)

        self.layer_last_bn = nn.BatchNorm1d(288)
        self.layer_last_relu = nn.ReLU()

        self.lstm = nn.LSTM(input_size=288, hidden_size=256, num_layers=self.lstm_layers, batch_first=True)
        self.dropout1 = nn.Dropout(p=0.2)
        self.fc1 = nn.Linear(256, self.output_classes)

    def init_hidden(self, batches):
        return (torch.zeros(self.lstm_layers, batches, 256, device=self.device),
                torch.zeros(self.lstm_layers, batches, 256, device=self.device))

    def _make_layer(self, inchannel, outchannel, block_num, device, kernel_size=1, stride=1, padding=0):
        layers = []
        for i in range(0, block_num - 1):
            layers.append(Block(inchannel, inchannel, device))

        shortcut = nn.Sequential(OrderedDict([
            ('AvgPool1', nn.AvgPool1d(kernel_size, stride, padding=0, ceil_mode=False, count_include_pad=True))
        ]))
        layers.append(Block(inchannel, outchannel, device, kernel_size, stride, padding, shortcut))
        return nn.Sequential(*layers)

    def load_weights(self, device, pretrained):
        pretrained_dict = pretrained['state_dict']
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)
        self.to(device)

    def forward(self, ecg):
        # (N, C, L)
        output1 = self.layer_1_conv1(ecg)
        output2 = self.layer_1_conv2(ecg)
        output3 = self.layer_1_conv3(ecg)
        output4 = self.layer_1_conv4(ecg)
        output = torch.cat((output1, output2, output3, output4), 1)

        output = self.block_1(output)

        output = self.block_2(output)

        output = self.block_3(output)

        output = self.block_4(output)

        output = self.block_5(output)

        output = self.block_6(output)

        output = self.block_7(output)

        output = self.block_8(output)

        output = self.layer_last_relu(
            self.layer_last_bn(output)
        )

        output = torch.transpose(output, 1, 2)  # (N, C, L) to (N, L, C)
        output, self.hidden = self.lstm(output, self.hidden)
        output = self.dropout1(output)
        output = self.fc1(output)
        return output
