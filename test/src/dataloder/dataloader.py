import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np

import math

import torch


# from hmcn import HMCN
# 数据迭代器
def dataset_split(x_data, input_ids_data, input_ids_train,attention_mask_data, attention_mask_train,nearest_data_eucli,
                  onehot_data, y_data, batch_size, shuffles, poi_graph, split_label, onehot_label_data, config):
    device = config.device
    i = 0
    data_num = len(x_data)
    n = math.ceil(data_num / batch_size)
    print(n)
    indices = np.arange(data_num)
    if shuffles:
        np.random.shuffle(indices)
        print("shuffles")
        print(indices[0])
    for j in range(n):
        if j != n - 1:
            index1 = i
            index2 = i + batch_size
            i = i + batch_size
        if j == n - 1:
            index1 = i
            index2 = data_num
        idx_seq = indices[index1: index2]
        y = y_data[idx_seq]

        #文本数据
        nearest_data_id = nearest_data_eucli[idx_seq]  # 获取这一个batch的近邻的id
        input_ids = input_ids_data[idx_seq]
        input_ids_k = input_ids_train[nearest_data_id]  # 邻居的tokenid 都是训练集的
        attention_mask = attention_mask_data[idx_seq]
        attention_mask_k = attention_mask_train[nearest_data_id]
        #空间数据
        geoinputdata = poi_graph
        if split_label == 'train':
            index_lists = idx_seq
        else:
            a = poi_graph[0].shape[0] - len(y_data)
            index_lists = [ind + a for ind in idx_seq]
            index_lists = np.array(index_lists)
        # 类别信息
        onehot_label_y = onehot_label_data[idx_seq]
        onehot_label = onehot_data[:, -config.hierarchy_classes[2]:][nearest_data_id]  # 邻居的类别向量 都是训练集的
        #输入
        inputs = [x_data[idx_seq], input_ids, input_ids_k, attention_mask,
                  attention_mask_k, onehot_label, index_lists, geoinputdata]
        for k in range(len(inputs)):
            if not isinstance(inputs[k], list):
                inputs[k] = torch.from_numpy(inputs[k]).to(torch.float32).to(device)
        for k in range(len(inputs[-1])):
            inputs[-1][k] = inputs[-1][k].to(device)
        inputs[-2] = inputs[-2].type(torch.long)

        yield j + 1, inputs, torch.from_numpy(y).to(torch.float32).to(device), torch.from_numpy(onehot_label_y).to(
            torch.float32).to(device)
