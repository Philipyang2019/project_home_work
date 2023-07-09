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


@dataclass
class TrainingArguments:
    output_dir: str = field(default='output_data/',metadata={'help': 'The output_data directory where the model predictions and checkpoints will be written.'})
    train_batch_size: int = field(default=64,metadata={'help': 'batch size for training'})
    eval_batch_size: int = field(default=64,metadata={'help': 'batch size for evaluation'})
    num_train_epochs: int = field(default=27,metadata={"help": "The total number of training epochs"})
    learning_rate: float = field(default=0.001,metadata={'help': '"The initial learning rate for AdamW.'})
    weight_decay: float = field(default=5e-4,metadata={"help": "Weight decay for AdamW"})
    logging_steps: int = field(default=50, metadata={'help': 'logging states every X updates steps.'})
    eval_steps: int = field(default=100,metadata={'help': 'Run an evaluation every X steps.'})
    device: str = field(default='cpu',metadata={"help": 'The device used for training'})
    def __str__(self):
        self_as_dict = dataclasses.asdict(self)
        attrs_as_str = [f"{k}={v},\n" for k, v in sorted(self_as_dict.items())]
        return f"{self.__class__.__name__}(\n{''.join(attrs_as_str)})"
    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(dataclasses.asdict(self), indent=2) + "\n"

# a = TrainingArguments()
# print(a)

#定义模型
class Encoder(nn.Module):
    def __init__(self, in_feat: int = 100, dropout_prob: float = 0.1):
        super().__init__()
        self.dense1 = nn.Linear(in_feat, in_feat)
        self.dense2 = nn.Linear(in_feat, in_feat)
        self.dropout = nn.Dropout(dropout_prob)
        self.act = nn.Tanh()
    def forward(self, token_embeds, attention_mask):
        batch_size = token_embeds.size(0)
        # 进行Mean pooling操作，即简单地将文本的词向量求和
        x = torch.stack([token_embeds[i, attention_mask[i, :], :].sum(dim=0) for i in range(batch_size)], dim=0)
        x = self.act(self.dense1(x))
        x = self.act(self.dense2(self.dropout(x)))
        return x

class Classifier(nn.Module):
    def __init__(self, in_feat, num_labels: int, dropout_prob: float = 0.1):
        super().__init__()
        self.dense1 = nn.Linear(in_feat, in_feat // 2)
        self.dense2 = nn.Linear(in_feat // 2, num_labels)
        self.act = nn.Tanh()
        self.dropout = nn.Dropout(dropout_prob)
    def forward(self, x):
        x = self.act(self.dense1(self.dropout(x)))
        x = self.dense2(self.dropout(x))
        return x


class SemNN(nn.Module):
    def __init__(self,in_feat: int = 100,num_labels: int = 3,dropout_prob: float = 0.1,w2v_state_dict: torch.Tensor = None,
                    vocab_size: int = None,word_embedding_dim: int = None):
        super().__init__()
        self.num_labels = num_labels
        self._init_word_embedding(w2v_state_dict,vocab_size,word_embedding_dim)
        self.encoder = Encoder(in_feat=in_feat)
        self.classifier = Classifier(in_feat=2 * in_feat, num_labels=num_labels, dropout_prob=dropout_prob)

    def _init_word_embedding(self, state_dict=None, vocab_size=None, word_embedding_dim=None):
        if state_dict is None:
            self.word_embedding = nn.Embedding(vocab_size, word_embedding_dim, padding_idx=0)
        else:
            # 默认载入预训练好的词向量（且固定词向量），并将其第一个词作为填充词（以及其对应向量设为零向量）
            state_dict = torch.tensor(state_dict.vectors, dtype=torch.float32)
            state_dict[0] = torch.zeros(state_dict.size(-1))
            self.word_embedding = nn.Embedding.from_pretrained(state_dict, freeze=True, padding_idx=0)

    def forward(self,text_a_input_ids,text_b_input_ids,text_a_attention_mask,text_b_attention_mask,labels=None):
        # 将两个query的词ID转换为其相应的词向量
        text_a_vecs = self.word_embedding(text_a_input_ids)
        text_b_vecs = self.word_embedding(text_b_input_ids)
        # 通过Encoder得到两个query的向量表示
        text_a_vec = self.encoder(text_a_vecs, text_a_attention_mask)
        text_b_vec = self.encoder(text_b_vecs, text_b_attention_mask)
        # 拼接两个Query的表示，再输入到分类器中
        pooler_output = torch.cat([text_a_vec, text_b_vec], dim=-1)
        logits = self.classifier(pooler_output)#[batch_size,num_labels]
        # 训练过程中的Loss计算
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        return (loss, logits) if loss is not None else logits



# in_feat = 100
# dropout_prob = 0.1
# num_labels = ['0','1','2']
# with open(r'E:\python_study\nlp_classification\yiliao_study\data_original\w2v_model.pickle', 'rb') as f:
#    w2v_model = pickle.load(f)
#
# from data_process_self import MyDataset,DataCollator
# mydataset = MyDataset(r'E:\python_study\nlp_classification\yiliao_study\data_original\KUAKE-QQR_train.json',
#                       vocab_mapping = w2v_model.key_to_index, max_length = 64, label_list = ['0','1','2'])
# data_collator = DataCollator()
# dataloader = DataLoader(dataset=mydataset,batch_size=128,shuffle=False,collate_fn=data_collator)
# iterator = iter(dataloader)
# data = next(iterator)
#
# model = SemNN(in_feat=in_feat,num_labels=len(num_labels),dropout_prob=dropout_prob,w2v_state_dict=w2v_model)
# model.to('cpu')
#
#
# a = model(**data)
# print(a)