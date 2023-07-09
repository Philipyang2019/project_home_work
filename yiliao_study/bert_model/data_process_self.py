import jieba
import os
import json
import math
import numpy as np
import dataclasses
from dataclasses import dataclass, field
from typing import Optional, List, Union, Any
from collections.abc import Mapping

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import transformers
from transformers import set_seed
from transformers.data.processors.utils import InputExample, InputFeatures
from transformers.data import DefaultDataCollator
from transformers import PreTrainedTokenizer, AutoTokenizer, PreTrainedModel, BertForSequenceClassification
from transformers.trainer_pt_utils import get_parameter_names
from transformers.optimization import get_linear_schedule_with_warmup, AdamW

jieba.setLogLevel(logging.INFO)


class ClassificationDataset(Dataset):

    def __init__(self,examples: List[InputExample],label_list: List[Union[str, int]],tokenizer: PreTrainedTokenizer,max_length: int = 128,processor=None):
        super().__init__()
        self.examples = examples
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.processor = processor
        self.label2id = {label: idx for idx, label in enumerate(label_list)}
        self.id2label = {idx: label for idx, label in enumerate(label_list)}

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index) -> InputFeatures:
        example = self.examples[index]
        label = None
        if example.label is not None:
            label = self.label2id[example.label]

        inputs = self.tokenizer(text=example.text_a,text_pair=example.text_b,padding='max_length',truncation=True,max_length=self.max_length)
        feature = InputFeatures(**inputs, label=label)
        return feature


class ClassificationDataset1(Dataset):

    def __init__(self,examples: List[InputExample],label_list: List[Union[str, int]],tokenizer: PreTrainedTokenizer,max_length: int = 128,processor=None):
        super().__init__()
        self.examples = examples
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.processor = processor
        self.label2id = {label: idx for idx, label in enumerate(label_list)}
        self.id2label = {idx: label for idx, label in enumerate(label_list)}

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index) -> InputFeatures:
        example = self.examples[index]
        label = None
        if example.label is not None:
            label = self.label2id[example.label]
        inputs = self.tokenizer(text=example.text_a,text_pair=example.text_b,padding='max_length',truncation=True,max_length=self.max_length)
        feature = InputFeatures(**inputs, label=label)
        return feature

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