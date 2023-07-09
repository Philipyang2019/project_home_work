import os
import json
import math
import numpy as np
import dataclasses
from dataclasses import dataclass, field
from typing import Optional, List, Union, Any,Dict
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

set_seed(2022)


@dataclass
class DataTrainingArguments:
    model_dir: str = field(default=r'C:\Users\admin\.cache\huggingface\hub\models--hfl--chinese-bert-wwm-ext',metadata={'help': 'The pretrained model directory'})
    data_dir: str = field(default=r'E:\python_study\nlp_classification\yiliao_study\data_original',metadata={'help': 'The data directory'})
    max_length: int = field(default=64,metadata={'help': 'Maximum sequence length allowed to input'})

    def __str__(self):
        self_as_dict = dataclasses.asdict(self)
        attrs_as_str = [f"{k}={v},\n" for k, v in sorted(self_as_dict.items())]
        return f"{self.__class__.__name__}(\n{''.join(attrs_as_str)})"

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(dataclasses.asdict(self), indent=2) + "\n"

class QQRProcessor:
    TASK = 'KUAKE-QQR'

    def __init__(self, data_dir):
        self.task_dir = os.path.join(data_dir)

    def get_train_examples(self):
        return self._create_examples(os.path.join(self.task_dir, f'KUAKE-QQR_train.json'))

    def get_dev_examples(self):
        return self._create_examples(os.path.join(self.task_dir, f'KUAKE-QQR_dec.json'))

    def get_test_examples(self):
        return self._create_examples(os.path.join(self.task_dir, f'KUAKE-QQR_test.json'))

    def get_labels(self):
        return ["0", "1", "2"]

    def _create_examples(self, data_path):

        with open(data_path, 'r', encoding='utf-8') as f:
            samples = json.load(f)

        examples = []
        for sample in samples:
            guid = sample['id']
            text_a = sample['query1']
            text_b = sample['query2']
            label = sample.get('label', None)

            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))

        return examples


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

class DataCollator:
    def __call__(self, features):
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

data_collator = DataCollator()
import time
data_args = DataTrainingArguments()
processor = QQRProcessor(data_args.data_dir)
print(data_args)
tokenizer = AutoTokenizer.from_pretrained('hfl/chinese-bert-wwm-ext')
processor = QQRProcessor(data_args.data_dir)

import torch
from einops import rearrange,reduce,repeat

x = torch.randn(2,3,4,4) #W 4D tenaor: bu*1c*h*w
# 1，转置
out1 =  x.transpose(1, 2)
out2 = rearrange(x, 'b i h w -> b h i w')
# 2.变形
out1 = x.reshape(6, 4,4)
out2 = rearrange(x, 'b i h w -> (b i) h w')
out3 = rearrange(out2,'(b i) h w-> b i h w', b=2)
flag  =  torch.allclose(out1, out2)
flag  = torch.allclose(out3,x)
# 3.imago2patch
out1 = rearrange(x,'b i (h1 p1) (w1 p2)->b i (h1 w1) (p1 p2)',p1=2,p2=2) #p1.p2Mpaten的高度机宽度
out2 = rearrange(out1, 'b i n a -> b n (a i)') #out2 ahape:(batchaize, num_patch, patch_depth)
# 4、次平均池化
out1 = reduce(x, 'b i h w -> b i h', 'mean') #mean, min, max, num, prod
out2 = reduce(x, 'b i h w->b i h 1','sum') #keep dimension
out3 = reduce(x,'b i h w -> b i', 'max')
# 5，维叠张量
tensor_list = [x, x, x]
out1  = rearrange(tensor_list, 'n b i h w -> n b i h w')
# 6.扩维
out1 = rearrange(x,'b i h w ->b i h w 1') #类似ftorch.unnqueeze
#7，复制
out2 = repeat(out1,'b i h w 1 -> b i h w 2') #类似ftorch.tile
out3 =repeat(x,'b i h w -> b i (2 h) (2 w)')
print(out1.shape)
print(out2.shape)
print(out3.shape)