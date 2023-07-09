#引入相关模块
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
import sys
sys.path.append(r'E:\python_study\nlp_classification\yiliao_study\nn_model')
from data_process_self import MyDataset,DataCollator
from model_process_self import SemNN


#处理数据，引入dataset
with open(r'E:\python_study\nlp_classification\yiliao_study\data_original\w2v_model.pickle', 'rb') as f:
   w2v_model = pickle.load(f)
train_dataset = MyDataset(r'E:\python_study\nlp_classification\yiliao_study\data_original\KUAKE-QQR_train.json',
                      vocab_mapping = w2v_model.key_to_index, max_length = 64, label_list = ['0','1','2'])

dev_dataset = MyDataset(r'E:\python_study\nlp_classification\yiliao_study\data_original\KUAKE-QQR_dev.json',
                      vocab_mapping = w2v_model.key_to_index, max_length = 64, label_list = ['0','1','2'])
data_collator = DataCollator()
train_dataloader = DataLoader(dataset=train_dataset,batch_size=64,shuffle=False,collate_fn=data_collator)
dev_dataloader = DataLoader(dataset=dev_dataset,batch_size=64,shuffle=False,collate_fn=data_collator)
#定义模型参数,引入model
device: str = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
in_feat = 100
dropout_prob = 0.1
num_labels = ['0','1','2']

model = SemNN(in_feat=in_feat,num_labels=len(num_labels),dropout_prob=dropout_prob,w2v_state_dict=w2v_model)
# 定义优化器和学习率相关内容
learning_rate: float = 0.001
weight_decay: float = 5e-4

optimizer = AdamW(model.parameters(),lr=learning_rate,weight_decay=weight_decay,)     #构建优化器
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=3, T_mult=2, eta_min=1e-5) # 构建学习率调度器
#定义训练参数和评价参数
output_dir: str = r'E:\python_study\nlp_classification\yiliao_study\nn_model\output_data'
train_batch_size: int = 64
eval_batch_size: int = 64
num_train_epochs: int = 50
logging_steps: int = 50
eval_steps: int = 100
num_examples = len(train_dataloader.dataset)                            #样本个数
num_update_steps_per_epoch = len(train_dataloader)                      #batch_size的个数
max_steps = math.ceil(num_train_epochs * num_update_steps_per_epoch)    #迭代的个数
num_train_samples = len(train_dataset) * num_train_epochs               #全量循环训练的样本数

def _prepare_input(data: Union[torch.Tensor, Any], device: str = 'cuda'):
    # 将准备输入模型中的数据转到GPU上
    if isinstance(data, Mapping):
        return type(data)({k: _prepare_input(v, device) for k, v in data.items()})
    elif isinstance(data, (tuple, list)):
        return type(data)(_prepare_input(v, device) for v in data)
    elif isinstance(data, torch.Tensor):
        kwargs = dict(device=device)
        return data.to(**kwargs)
    return data
#评估函数
def evaluate(device,model: nn.Module,eval_dataloader):
    model.eval()
    loss_list = []
    preds_list = []
    labels_list = []

    for item in eval_dataloader:
        inputs = _prepare_input(item, device=device)

        with torch.no_grad():
            outputs = model(**inputs)
            loss = outputs[0]
            loss_list.append(loss.detach().cpu().item())
            preds = torch.argmax(outputs[1].cpu(), dim=-1).numpy()
            preds_list.append(preds)
            labels_list.append(inputs['labels'].cpu().numpy())
    preds = np.concatenate(preds_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)
    loss = np.mean(loss_list)
    accuracy = (preds == labels).mean()
    model.train()
    return loss, accuracy

#开始训练模型
model.zero_grad()
model.train()
global_steps = 0
best_metric = 0.0
best_steps = -1
for epoch in range(num_train_epochs):
    for step, item in enumerate(train_dataloader):
        inputs = _prepare_input(item, device=device)
        outputs = model(**inputs)
        loss = outputs[0]
        loss.backward()
        optimizer.step()
        lr_scheduler.step(epoch + step / num_update_steps_per_epoch)
        optimizer.zero_grad()
        global_steps += 1

        if global_steps % logging_steps == 0:
            print(f'Training: Epoch {epoch + 1}/{num_train_epochs} - Step {(step + 1)} - Loss {loss}')

        if global_steps % eval_steps == 0:
            loss, acc = evaluate(device, model, dev_dataloader)
            print(f'Evaluation: Epoch {epoch + 1}/{num_train_epochs} - Step {(global_steps + 1)} - Loss {loss} - Accuracy {acc}')
            if acc > best_metric:
                best_metric = acc
                best_steps = global_steps
                saved_path = os.path.join(output_dir, f'checkpoint-{best_steps}.pt')
                torch.save(model.state_dict(), saved_path)

