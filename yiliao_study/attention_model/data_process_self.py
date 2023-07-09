import os
import json
import math
import pickle

import jieba
import dataclasses
import numpy as np
import logging
from dataclasses import field, dataclass
import pandas as pd
from gensim.models import KeyedVectors
from typing import List, Union, Dict, Any, Mapping, Optional

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

jieba.setLogLevel(logging.INFO)


class MyDataset(Dataset):
    def __init__(self, data_path,vocab_mapping,max_length,label_list):
        super().__init__()
        with open(data_path, 'r', encoding='utf-8') as f:
            data = []
            for sample in json.load(f):
                data.append([sample.get('id',None),sample.get('query1',None),sample.get('query2',None),sample.get('label',None)])
        self.data = pd.DataFrame(data,columns=['id', 'query1','query2','label'])
        self.features = self.data[['id', 'query1','query2']].values  # [[]]是返回结果是一个df，[]可能返回一个Serial；
        self.label = self.data[['label']].values
        self.vocab_mapping = vocab_mapping
        self.max_length = max_length
        self.label2id = {label: idx for idx, label in enumerate(label_list)}
        self.id2label = {idx: label for idx, label in enumerate(label_list)}


    def _pad_truncate(self, token_ids: List[int]):
        """
        # 如果文本长度（以词为单位）大于设定的阈值，则截断末尾部分；
        # 如果文本长度小于设定的阈值，则填充0
        """
        attention_mask = None # attention_mask作为标识文本填充情况
        # 如果文本长度（以词为单位）大于设定的阈值，则截断末尾部分；
        # 如果文本长度小于设定的阈值，则填充0
        if len(token_ids) > self.max_length:
            token_ids = token_ids[:self.max_length],
            attention_mask = [1] * self.max_length
        else:
            attention_mask = [1] * len(token_ids)
            diff = self.max_length - len(token_ids)
            token_ids.extend([0] * diff)
            attention_mask.extend([0] * diff)
        return token_ids, attention_mask

    def _tokenize(self, text):
        """
        OOV情况逻辑
        如果该词为多字的词，将其拆分多个字，分别将这些字转换为相应的ID；
        如果该词为单字，则从词表中随机采样一个词，其ID作为该词的ID
        """
        # 文本分词
        tokens = list(jieba.cut(text))
        token_ids = []
        for token in tokens:
            if token in self.vocab_mapping:
                token_id = self.vocab_mapping[token] # 如果当前词存在于词表，将词转换为词的ID.
                token_ids.append(token_id)
            else:
                if len(token) > 1:
                    for t in list(token):
                        if t in self.vocab_mapping:
                            token_ids.append(self.vocab_mapping[t])
                        else:
                            token_ids.append(np.random.choice(len(self.vocab_mapping), 1)[0])
                else:
                    token_ids.append(np.random.choice(len(self.vocab_mapping), 1)[0])

        # 对文本进行填充或者截断
        token_ids, attention_mask = self._pad_truncate(token_ids)
        return token_ids, attention_mask


    def __getitem__(self, index):  # 参数index必写
        # return self.features[index], self.label[index]
        label = self.label[index]
        text_a = self.features[index][1]
        text_b = self.features[index][2]
        # tokenize
        text_a_token_ids, text_a_attention_mask = self._tokenize(text_a)
        text_b_token_ids, text_b_attention_mask = self._tokenize(text_b)


        return {'text_a_input_ids': text_a_token_ids,
                'text_b_input_ids': text_b_token_ids,
                'text_a_attention_mask': text_a_attention_mask,
                'text_b_attention_mask': text_b_attention_mask,
                'label': label}

    def __len__(self):
        return len(self.data)

# w2v_file = r'E:\python_study\nlp_classification\yiliao_study\data_original\tencent-ailab-embedding-zh-d100-v0.2.0-s\tencent-ailab-embedding-zh-d100-v0.2.0-s\tencent-ailab-embedding-zh-d100-v0.2.0-s.txt'
# w2v_model = KeyedVectors.load_word2vec_format(w2v_file, binary=False)
#第一次保存模型
# import pickle
# with open(r'E:\python_study\nlp_classification\yiliao_study\data_original\w2v_model.pickle', 'wb') as f:
#    pickle.dump(w2v_model,f)
#后续加载模型
# with open(r'E:\python_study\nlp_classification\yiliao_study\data_original\w2v_model.pickle', 'rb') as f:
#    w2v_model = pickle.load(f)
# mydataset = MyDataset(r'E:\python_study\nlp_classification\yiliao_study\data_original\KUAKE-QQR_train.json',
#                       vocab_mapping = w2v_model.key_to_index, max_length = 64, label_list = ['0','1','2'])


class DataCollator:
    def __call__(self, features: List[Dict[str, Any]]):
        text_a_input_ids = []
        text_b_input_ids = []
        text_a_attention_mask = []
        text_b_attention_mask = []
        labels = []
        for item in features:
            text_a_input_ids.append(item['text_a_input_ids'])
            text_b_input_ids.append(item['text_b_input_ids'])
            text_a_attention_mask.append(item['text_a_attention_mask'])
            text_b_attention_mask.append(item['text_b_attention_mask'])
            if item['label'] is not None:
                labels.append([int(item['label'])])
        text_a_input_ids = torch.tensor(text_a_input_ids, dtype=torch.long)
        text_b_input_ids = torch.tensor(text_b_input_ids, dtype=torch.long)
        text_a_attention_mask = torch.tensor(text_a_attention_mask, dtype=torch.bool)
        text_b_attention_mask = torch.tensor(text_b_attention_mask, dtype=torch.bool)
        if len(labels) > 0:
            labels = torch.tensor(labels, dtype=torch.long)
        else:
            labels = None

        return {'text_a_input_ids': text_a_input_ids, 'text_b_input_ids': text_b_input_ids,
                'text_a_attention_mask': text_a_attention_mask, 'text_b_attention_mask': text_b_attention_mask,
                'labels': labels}
# data_collator = DataCollator()
# dataloader = DataLoader(dataset=mydataset,batch_size=128,shuffle=False,collate_fn=data_collator)
# iterator = iter(dataloader)
# data = next(iterator)