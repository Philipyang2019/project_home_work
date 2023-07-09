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




in_feat = 100
dropout_prob = 0.1
num_labels = ['0','1','2']
with open(r'E:\python_study\nlp_classification\yiliao_study\data_original\w2v_model.pickle', 'rb') as f:
   w2v_model = pickle.load(f)

import sys
sys.path.append(r'E:\python_study\nlp_classification\yiliao_study\lstm_model')
from yiliao_study.attention_model.data_process_self import MyDataset,DataCollator
mydataset = MyDataset(r'E:\python_study\nlp_classification\yiliao_study\data_original\KUAKE-QQR_train.json',
                      vocab_mapping = w2v_model.key_to_index, max_length = 64, label_list = ['0','1','2'])
data_collator = DataCollator()
dataloader = DataLoader(dataset=mydataset,batch_size=128,shuffle=False,collate_fn=data_collator)
iterator = iter(dataloader)
data = next(iterator)
model = model = BertForSequenceClassification.from_pretrained('hfl/chinese-bert-wwm-ext', num_labels=3)
model.to('cpu')
a = model(**data)
print(a)