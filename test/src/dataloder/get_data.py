
import pandas as pd
import numpy as np

import re
from pandas.core.frame import DataFrame








# 文本转化为tokenid
def convert_example_to_feature(review, tokenizer):
    # combine step for tokenization, WordPiece vector mapping, adding special tokens as well as truncating reviews longer than the max length
    return tokenizer.encode_plus(review,
                                 add_special_tokens=True,  # add [CLS], [SEP]
                                 max_length=32,  # max length of the text that can go to BERT
                                 pad_to_max_length=True,  # add [PAD] tokens
                                 return_attention_mask=True,  # add attention mask to not focus on pad tokens
                                 truncation=True
                                 )


# 返回所有的tokenid
def encode_data(ds, tokenizer, limit=-1):
    input_ids_list = []
    attention_mask_list = []
    # prepare list, so that we can build up final TensorFlow datasets from slices.
    if (limit > 0):
        ds = ds.take(limit)
    for index, row in ds.iterrows():
        review = row["text"]
        bert_input = convert_example_to_feature(review, tokenizer)
        input_ids_list.append(bert_input['input_ids'])
        attention_mask_list.append(bert_input['attention_mask'])
    input_ids_ary = np.array(input_ids_list)
    attention_mask_ary = np.array(attention_mask_list)
    return input_ids_ary, attention_mask_ary
# 获取csv的数据
def get_csv(PATH, id_dataset):
    F_Data = pd.read_csv(PATH[:-3] + "/datasets/" + id_dataset + "/" + id_dataset + "F_data.csv")

    # 训练的每个POI 的种类

    namelist = []
    for i in range(len(F_Data)):
        poiname = re.sub('\(.*?\)', '', F_Data['name'][i])  # 去掉poiname中的括号以及括号内的内容
        p1 = re.compile(r'[(](.*?)[)]', re.S)
        bracketscontent = re.findall(p1, F_Data['name'][i])
        if len(bracketscontent) == 1:
            poiname = poiname + bracketscontent[0]  #
        namelist.append(poiname)

    df_text = DataFrame(namelist)
    df_raw = pd.concat([df_text], axis=1)
    df_raw.columns = ['text']
    return df_raw


# 获取所有数据
def getdata(PATH, id_dataset, num_nearest_text, tokenizer, config):
    data = np.load(PATH[:-3] + '/datasets/' + id_dataset + '/newalldata.npz', allow_pickle=True)
    # original data
    X_train = data['X_train']  # 文本特征
    X_test = data['X_test']
    y_train = data['y_train']  # 训练集种类
    y_test = data['y_test']  # 测试集种类
    train_l = y_train.shape[0]

    onehot_train = data['onehot_ary'][:train_l, :]
    onehot_test = data['onehot_ary'][train_l:, :]

    # original data
    df_raw = get_csv(PATH, config.id_dataset)
    input_ids_ary, attention_mask_ary = encode_data(df_raw, tokenizer)
    input_ids_train = input_ids_ary[:train_l]
    input_ids_test = input_ids_ary[train_l:]
    attention_mask_train = attention_mask_ary[:train_l]
    attention_mask_test = attention_mask_ary[train_l:]
    nearest_train_text = data['idx_eucli'][:train_l, :num_nearest_text]  # 每个POI的最近的n个邻居序号，取前面训练集个数
    dist_text_train = data['dist_eucli'][:train_l, :num_nearest_text]  # 对应每个POI与最近n个邻居的文本距离
    nearest_test_text = data['idx_eucli'][train_l:, :num_nearest_text]  # 每个POI的最近的n个邻居序号，与测试集个数相同
    dist_text_test = data['dist_eucli'][train_l:, :num_nearest_text]  # 对应每个POI与最近n个邻居的文本距离



    return input_ids_train, input_ids_test, \
           attention_mask_train,attention_mask_test,\
           nearest_train_text, nearest_test_text, \
           dist_text_train, dist_text_test, \
           X_train, X_test, y_train, y_test, \
           onehot_train, onehot_test