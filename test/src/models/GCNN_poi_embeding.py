import pandas as pd
import numpy as np
import torch.nn as nn
from torch_geometric.nn import GCNConv
import torch

# 获取图的数据
def get_poi_graph(id_dataset,PATH):
    """this function constructs a poi graph"""
    data = np.load(PATH[:-3] + '/datasets/' + id_dataset + '/newalldata.npz', allow_pickle=True)
    y_train = data['y_train']  # 训练集种类
    y_trianlabel = np.argmax(y_train, axis=1)
    y_test = data['y_test']
    train_l = len(y_trianlabel)
    n = 15
    nearest_geo = data['idx_geo'][:, :n]  # 每个POI的最近的n个邻居序号，取前面训练集个数的POI
    edge_csv = pd.read_csv(PATH[:-3] + "/datasets/" + id_dataset + "/" + "DT_alldata_xy.csv")
    if id_dataset == 'lixia':
        nk = 140  # 类别数
    elif id_dataset == 'haidian':
        nk = 248  # 类别数

    onehot_train = data['onehot_ary'][:train_l, -nk:]
    emb_train = onehot_train

    emb_test = np.zeros((len(y_test), len(emb_train[0])))
    n = 15
    for i in range(len(y_test)):
        ids = nearest_geo[i][:n]
        emb_test[i] = sum(emb_train[ids])/n

    emb = np.vstack((emb_train, emb_test))
    print(len(emb))
    print(emb[2])
    emb = torch.tensor(emb, dtype=torch.float32)
    edge_source = list(edge_csv['source'])
    edge_target = list(edge_csv['target'])
    edge_weight = torch.tensor(list(edge_csv['weight']))
    edge_weight = edge_weight / torch.max(edge_weight)
    edge_index = torch.tensor([edge_source, edge_target], dtype=torch.int64)
    edge_weight = torch.tensor(edge_weight, dtype=torch.float32)
    return [emb, edge_index, edge_weight]

# GCN
class GCN_Encoder(nn.Module):
    """POI GCN encoder"""
    def __init__(self, in_channels, hidden_channels):
        super(GCN_Encoder, self).__init__()
        self.conv = GCNConv(in_channels, hidden_channels, cached=True, bias=True, add_self_loops=False)
        self.soft = nn.Softmax(dim=1)
    def forward(self, x, edge_index, edge_weight):
        x = self.conv(x, edge_index, edge_weight)
        x = self.soft(x)
        return x