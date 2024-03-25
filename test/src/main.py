import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
import numpy as np


import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score

import datetime
from tqdm import tqdm
from torch import nn
import torch
from torch.optim.lr_scheduler import MultiStepLR, ReduceLROnPlateau
from transformers import BertTokenizer, AlbertModel, AlbertConfig, AutoTokenizer, AutoModel  # 需要安装transformers库

from torch.utils.data import Dataset

from models.GCNN_poi_embeding import get_poi_graph
from models.ITPA import ITPA
from dataloder.dataloader import dataset_split
from dataloder.get_data import getdata

import csv

torch.cuda.device_count()


# loss图像
def plt_epoch(tainlist, vallist, label, path, id_dataset):
    rcParams['figure.figsize'] = (8, 4)
    rcParams['figure.dpi'] = 100
    rcParams['font.size'] = 8
    rcParams['font.family'] = 'sans-serif'
    rcParams['axes.facecolor'] = '#ffffff'
    rcParams['lines.linewidth'] = 2.0
    rcParams['figure.figsize'] = (16, 6)
    plt.figure()
    # train
    plt.plot(tainlist, 'g', label='train_' + label)
    # val
    plt.plot(vallist, 'b', label='val_' + label)
    plt.title('model ' + label)
    plt.ylabel(label)
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper right')
    plt.savefig(
        path +".png")
    plt.close()



#    plt.show()
def plt_batch(lists, path, id_dataset):
    rcParams['figure.figsize'] = (8, 4)
    rcParams['figure.dpi'] = 100
    rcParams['font.size'] = 8
    rcParams['font.family'] = 'sans-serif'
    rcParams['axes.facecolor'] = '#ffffff'
    rcParams['lines.linewidth'] = 2.0
    rcParams['figure.figsize'] = (16, 6)
    plt.figure()
    # loss
    plt.plot(lists, 'g', label='train loss')
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper right')
    plt.savefig(
        path +".png")
    plt.close()


#    plt.show()
def plt_batchph(lists, path, id_dataset):
    rcParams['figure.figsize'] = (8, 4)
    rcParams['figure.dpi'] = 100
    rcParams['font.size'] = 8
    rcParams['font.family'] = 'sans-serif'
    rcParams['axes.facecolor'] = '#ffffff'
    rcParams['lines.linewidth'] = 2.0
    rcParams['figure.figsize'] = (16, 6)
    plt.figure()
    # loss
    plt.plot(lists, 'g', label='train loss')
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper right')
    plt.savefig(
        path + "_pinghua.png")
    plt.close()


# mrr指标
def compute_mrr(true_labels, machine_preds):
    """Compute the MRR """
    rr_total = 0.0
    for i in range(len(true_labels)):
        ranklist = list(np.argsort(machine_preds[i])[::-1])  # 概率从大到小排序，返回index值
        rank = ranklist.index(true_labels[i]) + 1  # 获取真实值的rank
        rr_total = rr_total + 1.0 / rank
    mrr = rr_total / len(true_labels)
    return mrr



def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

# 模型训练
def train(config):
    device = config.device
    model_path = config.model_path
    # 加载预训练模型
    tokenizer = BertTokenizer.from_pretrained(model_path)
    al_bert = AlbertModel.from_pretrained(model_path)

    # ----------------------------------------------------------------------------
    # getdata
    # ----------------------------------------------------------------------------

    input_ids_train, input_ids_test, \
    attention_mask_train, attention_mask_test, \
    nearest_train_text, nearest_test_text, \
    dist_text_train, dist_text_test, \
    X_train, X_test, y_train, y_test, \
    onehot_train, onehot_test = getdata(PATH=config.PATH,
                                        id_dataset=config.id_dataset,
                                        num_nearest_text=config.num_nearest_text,
                                        tokenizer=tokenizer,
                                        config=config)

    poi_graph = get_poi_graph(config.id_dataset, config.PATH,)

    # 定义维度
    config.num_classes = y_train.shape[1]

    num_features_extras_geo = config.hierarchy_classes[2]
    num_features_extras_text = config.hierarchy_classes[2]

    shape_input_phenomenon = X_train.shape[1]
    shape_input_phenomenon_text = X_train.shape[1]
    shape_input_phenomenon_geo = 0

    #    summary(model)
    # ----------------------------------------------------------------------------
    # Training loop.
    # ----------------------------------------------------------------------------
    train_losses_iter = []
    train_losses_iterph = []
    train_losses = []
    val_losses = []
    train_acc = []
    val_acc = []
    y_label = np.argmax(y_test, axis=1)
    y_trianlabel = np.argmax(y_train, axis=1)
    y_testlabel = np.argmax(y_test, axis=1)

    model = ITPA(shape_input_phe=shape_input_phenomenon,
                 shape_input_phenomenon_text=shape_input_phenomenon_text,
                 shape_input_phenomenon_geo=shape_input_phenomenon_geo,
                 num_features_extras_text=num_features_extras_text,
                 num_features_extras_geo=num_features_extras_geo,
                 num_nearest_text=config.num_nearest_text,
                 base_encoder=al_bert,
                 num_classes=config.num_classes,
                 config=config).to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(optimizer, 'max', factor=0.35, verbose=1, min_lr=0.0001, patience=2)

    header = ('l', 'epoch', 'Accuracy', 'F1-score', 'MRR')
    l = 0
    result_filename = "result_optimer"
    if not os.path.exists(config.base_result_outdir):
        os.makedirs(config.base_result_outdir)
    if not os.path.exists(config.base_result_image_outdir):
        os.makedirs(config.base_result_image_outdir)
    with open(config.base_result_outdir + result_filename +'.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
    for epoch in tqdm(range(config.number_of_epochs)):

        # 训练
        model.train()
        train_losses_batch = []
        accuracy_train = 0

        minibatches_train = dataset_split(x_data=X_train,
                                          input_ids_data=input_ids_train,
                                          input_ids_train=input_ids_train,
                                          attention_mask_data=attention_mask_train,
                                          attention_mask_train=attention_mask_train,
                                          nearest_data_eucli=nearest_train_text,
                                          onehot_data=onehot_train,
                                          y_data=y_trianlabel,
                                          batch_size=config.batch_size,
                                          shuffles=True,
                                          poi_graph=poi_graph,
                                          split_label='train',
                                          onehot_label_data=onehot_train,
                                          config=config)

        starttime = datetime.datetime.now()
        config.train = True

        for (step, x_train_inputs, label, onehot_label) in minibatches_train:
            optimizer.zero_grad()
            label = label.to(torch.long)


            outputs = model(x_train_inputs)
            loss = loss_fn(outputs, label)

            loss.backward()
            # 参数更新
            optimizer.step()

            # 统计预测信息
            _, predicted = torch.max(outputs.data, 1)
            labels = label
            # _, labels = torch.max(label, 1)
            correct = (predicted.cpu() == labels.cpu()).squeeze().sum().numpy()
            loss_data = loss.item()
            train_losses_batch.append(loss_data)  # 保存这个batch的loss
            train_losses_iter.append(loss_data)
            train_losses_iterph.append(sum(train_losses_batch) / len(train_losses_batch))
            accuracy_train = accuracy_train + correct / len(labels.cpu())

            # 输出loss 和 acc
            if step % 100 == 0:
                print(step, 'loss:', sum(train_losses_batch) / len(train_losses_batch), 'acc:',
                      accuracy_train / len(train_losses_batch))

        print("第%d个epoch的学习率：%f" % (epoch, optimizer.param_groups[0]['lr']))
        model.eval()
        config.train = False
        with torch.no_grad():
            minibatches_test = dataset_split(x_data=X_test,
                                             input_ids_data=input_ids_test, input_ids_train=input_ids_train,
                                             attention_mask_data=attention_mask_test,
                                             attention_mask_train=attention_mask_train,
                                             nearest_data_eucli=nearest_test_text,
                                             onehot_data=onehot_train,
                                             y_data=y_testlabel,
                                             batch_size=config.batch_size,
                                             shuffles=False,
                                             poi_graph=poi_graph,
                                             split_label='test',
                                             onehot_label_data=onehot_test,
                                             config=config)

            val_accuracy = 0
            val_loss = 0
            for (step, x_test_inputs, label, onehot_label) in minibatches_test:

                outputs = model(x_test_inputs)
                label = label.to(torch.long)
                labels = label
                val_loss += loss_fn(outputs, label).item()
                _, predicted = torch.max(outputs.data, 1)

                val_accuracy += (predicted.cpu() == labels.cpu()).squeeze().sum().numpy() / len(labels.cpu())
            val_loss = val_loss / step
            val_accuracy = val_accuracy / step

        # 一个epoch结束
        scheduler.step(val_accuracy)

        print('epoch:', epoch, 'train_loss:', sum(train_losses_batch) / len(train_losses_batch), 'train_acc:',
              accuracy_train / len(train_losses_batch), 'val_loss:', val_loss, 'val_acc:', val_accuracy)
        endtime = datetime.datetime.now()
        print("traintime:")
        print((endtime - starttime).seconds)

        train_losses.append(sum(train_losses_batch) / len(train_losses_batch))  # 保存这个epoch的loss

        train_acc.append(accuracy_train / len(train_losses_batch))  # 保存这个epoch的acc
        plt_batch(train_losses_iter, config.base_result_image_outdir+result_filename, config.id_dataset)

        val_losses.append(val_loss)
        val_acc.append(val_accuracy)  # 保存这个epoch的acc

        plt_batchph(train_losses_iterph, config.base_result_image_outdir+result_filename, config.id_dataset)

        plt_epoch(train_losses, val_losses, 'loss', config.base_result_image_outdir+result_filename, config.id_dataset)
        plt_epoch(train_acc, val_acc, 'acc', config.base_result_image_outdir+result_filename, config.id_dataset)

        with torch.no_grad():
            if epoch in [0, 29, 34]:
                minibatches_test_data = dataset_split(x_data=X_test,
                                                      input_ids_data=input_ids_test,
                                                      input_ids_train=input_ids_train,
                                                      attention_mask_data=attention_mask_test,
                                                      attention_mask_train=attention_mask_train,
                                                      nearest_data_eucli=nearest_test_text,
                                                      onehot_data=onehot_train,
                                                      y_data=y_testlabel,
                                                      batch_size=config.batch_size,
                                                      shuffles=False,
                                                      poi_graph=poi_graph,
                                                      split_label='test',
                                                      onehot_label_data=onehot_test,
                                                      config=config)
                for (step, x_test_inputs, label, onehot_label) in minibatches_test_data:

                    outputs = model(x_test_inputs)
                    if step == 1:
                        predictions_test = outputs.data.cpu()
                    else:
                        predictions_test = np.concatenate((predictions_test, outputs.data.cpu()), axis=0)

                predictions_test_dim = np.argmax(predictions_test, axis=1)
                accuracy_score_test = accuracy_score(y_label, predictions_test_dim)
                f1_score_test = f1_score(y_label, predictions_test_dim, average="macro")
                mrr_test = compute_mrr(y_label, predictions_test)


                with open(config.base_result_outdir + result_filename +'.csv', 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([l, epoch, accuracy_score_test, f1_score_test, mrr_test])


# 设置参数
class Config(object):
    PATH = os.path.dirname(os.path.realpath(__file__))
    PATH = PATH.replace('\\', '/')
    device = torch.device('cuda:3')
    batch_size = 64
    number_of_epochs = 35
    id_dataset = 'haidian'
    if id_dataset == 'lixia':
        num_classes = 140  # 类别数
    elif id_dataset == 'haidian':
        num_classes = 248  # 类别数
    num_nearest_geo = 15
    num_nearest_text = 15

    num_features_extras_geo = 0
    num_features_extras_text = 0
    max_length = 32
    n = 15  # 邻居个数

    if id_dataset == 'lixia':
        hierarchy_classes = [20, 79, 140]
    elif id_dataset == 'haidian':
        hierarchy_classes = [21, 108, 248]

    base_result_outdir = PATH[:-3] + "result/resulttxt/" + id_dataset + "/"
    base_result_image_outdir = PATH[:-3] + "result/image/" + id_dataset + "/"
    model_path = PATH[:-3] + "/albert_chinese_small/"



if __name__ == "__main__":
    config = Config()



    print(config.PATH)
    print(config.model_path)
    train(config)


