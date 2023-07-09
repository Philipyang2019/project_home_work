import os
import json
import math
import jieba
import dataclasses
import numpy as np
import logging
from dataclasses import field, dataclass
from gensim.models import KeyedVectors
from typing import List, Union, Dict, Any, Mapping, Optional

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

jieba.setLogLevel(logging.INFO)

@dataclass
class InputExample:

    guid: str
    text_a: str
    text_b: Optional[str] = None
    label: Optional[str] = None

    def to_json_string(self):
        return json.dumps(dataclasses.asdict(self), indent=2) + "\n"

class QQRProcessor:
    TASK = 'KUAKE-QQR'

    def __init__(self, data_dir):
        self.task_dir = os.path.join(data_dir)

    def get_train_examples(self):
        return self._create_examples(os.path.join(self.task_dir, f'{self.TASK}_train.json'))

    def get_dev_examples(self):
        return self._create_examples(os.path.join(self.task_dir, f'{self.TASK}_dev.json'))

    def get_test_examples(self):
        return self._create_examples(os.path.join(self.task_dir, f'{self.TASK}_test.json'))

    def get_labels(self):
        return ["0", "1", "2"]

    def _create_examples(self, data_path):

        # 读入文件
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

processor = QQRProcessor(data_dir=r'E:\python_study\nlp_classification\yiliao_study\data_original')

processor.get_test_examples()




processor = QQRProcessor(data_dir=data_args.data_dir)

train_dataset = QQRDataset(
    processor.get_train_examples(),
    processor.get_labels(),
    vocab_mapping=w2v_model.key_to_index,
    max_length=32
)


class QQRDataset(Dataset):

    def __init__(self,examples: List[InputExample],label_list: List[Union[str, int]],vocab_mapping: Dict,max_length: int = 64,):
        super().__init__()
        self.examples = examples
        self.vocab_mapping = vocab_mapping
        self.max_length = max_length
        self.label2id = {label: idx for idx, label in enumerate(label_list)}
        self.id2label = {idx: label for idx, label in enumerate(label_list)}

    def __len__(self):
        return len(self.examples)

    def _tokenize(self, text):
        # 文本分词
        tokens = list(jieba.cut(text))

        token_ids = []
        for token in tokens:
            if token in self.vocab_mapping:
                # 如果当前词存在于词表，将词转换为词的ID.
                token_id = self.vocab_mapping[token]
                token_ids.append(token_id)
            else:
                # OOV情况处理
                # 如果该词为多字的词，将其拆分多个字，分别将这些字转换为相应的ID；
                # 如果该词为单字，则从词表中随机采样一个词，其ID作为该词的ID
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

    def _pad_truncate(self, token_ids: List[int]):
        # attention_mask作为标识文本填充情况
        attention_mask = None
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

    def __getitem__(self, index):
        example = self.examples[index]
        label = None
        if example.label is not None:
            label = self.label2id[example.label]
        # tokenize
        text_a_token_ids, text_a_attention_mask = self._tokenize(example.text_a)
        text_b_token_ids, text_b_attention_mask = self._tokenize(example.text_b)
        return {'text_a_input_ids': text_a_token_ids, 'text_b_input_ids': text_b_token_ids,
                'text_a_attention_mask': text_a_attention_mask, 'text_b_attention_mask': text_b_attention_mask,
                'label': label}
