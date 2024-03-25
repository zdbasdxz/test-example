import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np


import copy
import datetime
from tqdm import tqdm
from torch import nn
import torch

from torch.utils.data import Dataset

from .GCNN_poi_embeding import get_poi_graph, GCN_Encoder





class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)


# 注意力层
class Attention(nn.Module):

    def __init__(self,
                 num_nearest,
                 shape_input_phenomenon,
                 num_features_extras,
                 label=None,
                 graph_label=None,
                 suffix_mean=None,
                 **kwargs):
        super().__init__()
        self.label = label
        self.num_nearest = num_nearest,  # 最近邻居个数
        self.shape_input_phenomenon = shape_input_phenomenon,  # 输入向量维度
        self.num_features_extras = num_features_extras,
        self.graph_label = graph_label,  # softmax层维度
        self.suffix_mean = suffix_mean

        self.kernel = nn.Parameter(torch.randn(self.num_nearest[0], self.num_nearest[0]))

        self.bias = nn.Parameter(torch.randn(self.num_nearest[0]))

    # 这是定义层功能的方法
    def forward(self, inputs):
        source_distance = inputs[0]  # Node features (N x F)
        context = inputs[1]
        source_distance.to(torch.float32)
        context.to(torch.float32)
        ######################## Attention data ########################
        self.distance = source_distance

        if self.label == 'text':

            self.simi = self.distance
        if self.label == 'geo':
            self.simi = Lambda(lambda x: torch.exp(-x * (10 ** 2 / 2)))(self.distance)
        self.simi = self.simi.to(torch.float32)
        self.weight = torch.mm(self.simi, self.kernel) + self.bias
        self.weight = torch.softmax(self.weight, dim=1)

        # 重复为了给每一个向量对应位置相乘
        repeattimes = self.shape_input_phenomenon[0] + self.num_features_extras[0]
        prob_repeat = torch.repeat_interleave(self.weight, repeattimes, dim=1).reshape(self.weight.shape[0],
                                                                                       self.weight.shape[1],
                                                                                       repeattimes)
        relevance = torch.multiply(prob_repeat, context)

        self.mean = torch.sum(relevance, axis=1)

        return self.mean


# 文本语义信息增强
class semanticmodelembeding(nn.Module):

    def __init__(self, base_encoder, num_nearest_text, num_features_extras_text):
        super().__init__()
        self.base_encoder = base_encoder
        self.base_encoder_k = copy.deepcopy(base_encoder)  # base_encoder_k
        
        #momentum_param
        self.m = 1.0
        for param_q, param_k in zip(self.base_encoder.parameters(), self.base_encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False
        self.num_nearest_text = num_nearest_text
        self.attention = Attention(num_nearest=num_nearest_text,
                                   shape_input_phenomenon=0,
                                   num_features_extras=num_features_extras_text,
                                   label='text',
                                   graph_label='semantic_weight_text',  # label
                                   suffix_mean='semantic_mean_text'  # label
                                   )

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        % key encoder的Momentum update
        """
        for param_q, param_k in zip(self.base_encoder.parameters(), self.base_encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    def forward(self, inputs):
        idx = inputs[0].to(torch.int32)
        idx_k = inputs[1].to(torch.int32)
        att = inputs[2].to(torch.int32)
        att_k = inputs[3].to(torch.int32)
        label_onehot = inputs[4]
        idx_k = torch.reshape(idx_k, (-1, 32))
        att_k = torch.reshape(att_k, (-1, 32))
        x1 = self.base_encoder(input_ids=idx, attention_mask=att)[1]
        x2 = self.base_encoder_k(input_ids=idx_k, attention_mask=att_k)[1]
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()
        xf2 = x1
        xfk2 = x2
        # 现在都是二维矩阵，先正则化
        normalize_a = torch.nn.functional.normalize(xfk2, p=2, dim=1)
        normalize_b = torch.nn.functional.normalize(xf2, p=2, dim=1)
        # 将自身转化为三维矩阵，方便之后计算
        normalize_b = torch.reshape(normalize_b, (-1, 1, 384))

        # 将邻居转化为三维矩阵
        xfk2 = torch.reshape(xfk2, (-1, self.num_nearest_text, 384))

        normalize_a = torch.reshape(normalize_a, (-1, self.num_nearest_text, 384))

        distance = 1 - torch.matmul(normalize_a, normalize_b.permute(0, 2, 1))
        # 最后转化为二维矩阵
        distance = torch.reshape(distance, (-1, self.num_nearest_text))

        # 每个poi的邻居 + type
        # context = torch.cat((xfk2, label_onehot), dim=2)
        context = label_onehot
        mean = self.attention([distance, context])
        return mean

class self_Attention(nn.Module):

    def __init__(self, input_dinm, hidden_dim):
        super(self_Attention, self).__init__()

        self.hidden_dim = hidden_dim
        # 下面使用nn的Linear层来定义Q，K，V矩阵
        self.Q_linear = nn.Linear(input_dinm, hidden_dim)
        self.K_linear = nn.Linear(input_dinm, hidden_dim)
        self.V_linear = nn.Linear(input_dinm, hidden_dim)

    def forward(self, inputs):
        # 计算生成QKV矩阵
        Q = self.Q_linear(inputs)
        K = self.K_linear(inputs)
        V = self.V_linear(inputs)
        device = inputs.device
        attention = torch.matmul(Q, K.transpose(-1, -2))
        d_k = torch.from_numpy(np.array(K.shape[-1])).to(torch.float32).to(device)
        attention = torch.divide(attention, torch.sqrt(d_k))
        attention = torch.softmax(attention, axis=-1)
        output = torch.matmul(attention, V)
        return output


class mv_attn(nn.Module):

    def __init__(self, inputdim):
        super(mv_attn, self).__init__()
        self.inputdim = inputdim,  # 最近邻居个数
        # Layer kernel 论文中的H = WD+b 的W
        # self.kernel = nn.Parameter(torch.randn(self.inputdim[0], self.inputdim[0]))
        # Layer bias 论文中的H = WD+b的b
        # self.bias = nn.Parameter(torch.randn(self.inputdim[0]))
        self.L = nn.Linear(inputdim, inputdim)

    # 这是定义层功能的方法
    def forward(self, view):
        # weights1 = torch.sigmoid(torch.matmul(view[0], self.kernel) + self.bias)
        # weights2 = torch.sigmoid(torch.matmul(view[1], self.kernel) + self.bias)
        weights1 = torch.sigmoid(self.L(view[0]))
        weights2 = torch.sigmoid(self.L(view[1]))
        attns1 = torch.multiply(view[0], weights1)
        attns2 = torch.multiply(view[1], weights2)
        attn = torch.add(attns1, attns2)
        return attn


class mul_attn(nn.Module):

    def __init__(self, inputdim):
        super(mul_attn, self).__init__()
        self.inputdim = inputdim,  # 最近邻居个数
        # Layer kernel 论文中的H = WD+b 的W
        self.kernel = nn.Linear(self.inputdim[0], self.inputdim[0])
        self.kernel1 = nn.Linear(self.inputdim[0], 1, bias=False)

    # 这是定义层功能的方法
    def forward(self, view):
        weights_a1 = self.kernel1(torch.tanh(self.kernel(view[0])))
        weights_a2 = self.kernel1(torch.tanh(self.kernel(view[1])))
        weights_a = torch.cat((weights_a1, weights_a2), dim=1)
        weights_b = torch.softmax(weights_a, dim=1)

        attns1 = torch.multiply(view[0], weights_b[:, :1])
        attns2 = torch.multiply(view[1], weights_b[:, 1:])
        attn = torch.add(attns1, attns2)
        # attn = torch.cat((attns1, attns2), dim=1)
        return attn


class DeepAndWide(nn.Module):
    def __init__(self, inputdim):
        super(DeepAndWide, self).__init__()
        self.inputdim = inputdim
        self.L = nn.Linear(2 * inputdim, 2 * inputdim)
        self.L2 = nn.Linear(inputdim * inputdim, 2 * inputdim)

    # 这是定义层功能的方法
    def forward(self, view):
        hspatial = view[0]
        htext = view[1]
        g = self.L(torch.cat((hspatial, htext), dim=1))
        for i in range(hspatial.shape[0]):
            if i == 0:
                v = torch.flatten(torch.outer(hspatial[i], htext[i]), 0).unsqueeze(0)
            else:
                newv = torch.flatten(torch.outer(hspatial[i], htext[i]), 0).unsqueeze(0)
                v = torch.cat((v, newv), dim=0)
        u = self.L2(v)
        c = torch.cat((g, u), dim=1)
        # c = torch.add(g, u)
        return c








class ITPA(nn.Module):
    def __init__(self, shape_input_phe,
                 shape_input_phenomenon_text,
                 shape_input_phenomenon_geo,
                 num_features_extras_text,
                 num_features_extras_geo,
                 num_nearest_text,
                 base_encoder,
                 num_classes,
                 config):
        super(ITPA, self).__init__()
        # create model
        self.shape_input_phe = shape_input_phe
        self.shape_input_phenomenon_text = shape_input_phenomenon_text
        self.shape_input_phenomenon_geo = shape_input_phenomenon_geo
        self.num_features_extras_text = num_features_extras_text
        self.num_features_extras_geo = num_features_extras_geo
        self.num_nearest_text = num_nearest_text
        self.base_encoder = base_encoder
        self.num_classes = num_classes
        self.inputsize = 0


        ######################## hiden layer ########################

        self.textattention_b = semanticmodelembeding(self.base_encoder, self.num_nearest_text,
                                                     self.num_features_extras_text)

        self.GCN = GCN_Encoder(config.hierarchy_classes[2], config.hierarchy_classes[2])
        self.inputsize = shape_input_phe + self.num_features_extras_geo + num_features_extras_text  # +384

        print(self.inputsize)

        self.L1 = nn.Linear(self.inputsize, 256)
        self.ELU1 = nn.ELU()
        self.L2 = nn.Linear(256, 128)
        self.ELU2 = nn.ELU()  # inplace = True
        self.Drop = nn.Dropout(p=0.5)
        # Output
        self.L3 = nn.Linear(128, self.num_classes)


    def forward(self, x):
        spatialconcate = self.GCN(x[-1][0], x[-1][1], x[-1][2])[x[-2]]
        semanticconcate = self.textattention_b([x[1], x[2], x[3], x[4], x[5]])

        ######################## spatialconcate ########################

        self.concate = x[0]
        self.concate = torch.cat((self.concate, spatialconcate), dim=1)
        self.concate = torch.cat((self.concate, semanticconcate), dim=1)

        feedforward = self.L1(self.concate)
        feedforward = self.ELU1(feedforward)
        feedforward = self.L2(feedforward)
        feedforward = self.ELU2(feedforward)
        dropout = self.Drop(feedforward)

        # Output
        main_output = self.L3(dropout)
        return main_output