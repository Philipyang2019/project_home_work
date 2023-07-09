import jieba
import logging
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import pickle
from torch.utils.data import DataLoader

jieba.setLogLevel(logging.INFO)


class Encoder(nn.Module):

    def __init__(self, in_feat: int = 100, dropout_prob: float = 0.1):
        super().__init__()

        self.lstm = nn.LSTM(input_size=in_feat, hidden_size=in_feat, bidirectional=True, batch_first=True)

    def forward(self, token_embeds, attention_mask):
        batch_size = attention_mask.size(0)
        output, (h, c) = self.lstm(token_embeds)
        output, lens_output = pad_packed_sequence(output, batch_first=True)  # [B, L, H]

        return output, lens_output


class Decoder(nn.Module):

    def __init__(self, in_feat: int = 100, dropout_prob: float = 0.1):
        super().__init__()

        self.lstm = nn.LSTM(input_size=in_feat, hidden_size=in_feat, bidirectional=True, batch_first=True)

    def forward(self, token_embeds, attention_mask):
        batch_size = attention_mask.size(0)
        output, (h, c) = self.lstm(token_embeds)
        output, lens_output = pad_packed_sequence(output, batch_first=True)  # [B, L, H]
        output = torch.stack([torch.mean(output[i][:lens_output[i]], dim=0) for i in range(batch_size)], dim=0)

        return output


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


class CrossAttention(nn.Module):

    def __init__(self, in_feat, dropout):
        super().__init__()

        self.dense = nn.Linear(4 * in_feat, in_feat // 2)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, a, b, mask_a, mask_b):
        in_feat = a.size(-1)
        # a: [B, L1, H], b: [B, L2, H]
        # 计算query A和query B之间的attention score，即 Key * Value
        # a [128, 11, 200],b[128, 14, 200]   cross_attn [128,11,14]
        cross_attn = torch.matmul(a, b.transpose(1, 2))  # [B, L1, L2]

        # 将填充位置的score设为-1e9，即不考虑填充位置的信息
        row_attn = cross_attn.masked_fill((mask_b == False).unsqueeze(1), -1e9)
        row_attn = row_attn.softmax(dim=2)  # [B, L1, L2]

        #col_attn从[128,11,14]变成[128,14,11]
        col_attn = cross_attn.permute(0, 2, 1).contiguous()  # [B, L2, L1]
        col_attn = col_attn.masked_fill((mask_a == False).unsqueeze(1), -1e9)
        col_attn = col_attn.softmax(dim=2)

        # attention score * value
        # row_attn [128, 11, 14] * b [128, 14, 200] -> [128,11,200] 实际含义是计算11个中的每个词对14个词的加权值
        attn_a = torch.matmul(row_attn, b)  # [B, L1, H]
        attn_b = torch.matmul(col_attn, a)  # [B, L2, H]

        diff_a = a - attn_a
        diff_b = b - attn_b
        prod_a = a * attn_a
        prod_b = b * attn_b

        # 将原本的hidden state和attention得到的hidden state拼接，并经过线性变换降维
        a = torch.cat([a, attn_a, diff_a, prod_a], dim=-1)  # [B, L1, 2*H]
        b = torch.cat([b, attn_b, diff_b, prod_b], dim=-1)  # [B, L2, 2*H]

        a = self.act(self.dense(self.dropout(a)))  # [B, L1, H/2]
        b = self.act(self.dense(self.dropout(b)))  # [B, L2, H/2]

        return a, b


class SemAttn(nn.Module):

    def __init__(self,in_feat: int = 100,num_labels: int = 3,dropout_prob: float = 0.1
                 ,w2v_state_dict: torch.Tensor = None,vocab_size: int = None,word_embedding_dim: int = None):
        super().__init__()
        self.num_labels = num_labels
        self._init_word_embedding(w2v_state_dict, vocab_size, word_embedding_dim)
        self.encoder = Encoder(in_feat=in_feat, dropout_prob=dropout_prob)
        self.decoder = Decoder(in_feat=in_feat, dropout_prob=dropout_prob)
        self.cross_attn = CrossAttention(in_feat=in_feat * 2, dropout=dropout_prob)
        self.classifier = Classifier(in_feat=4 * in_feat, num_labels=num_labels, dropout_prob=dropout_prob)

    def _init_word_embedding(self, state_dict=None, vocab_size=None, word_embedding_dim=None):
        if state_dict is None:
            self.word_embedding = nn.Embedding(vocab_size, word_embedding_dim, padding_idx=0)
        else:
            # 默认载入预训练好的词向量（且固定词向量），并将其第一个词作为填充词（以及其对应向量设为零向量）
            state_dict = torch.tensor(state_dict.vectors, dtype=torch.float32)
            state_dict[0] = torch.zeros(state_dict.size(-1))
            self.word_embedding = nn.Embedding.from_pretrained(state_dict, freeze=True, padding_idx=0)

    def forward(self,text_a_input_ids: torch.Tensor,text_b_input_ids: torch.Tensor,text_a_attention_mask: torch.Tensor,text_b_attention_mask: torch.Tensor,labels=None):

        text_a_vecs = self.word_embedding(text_a_input_ids)#[128, 64] ->[128, 64, 100]
        text_b_vecs = self.word_embedding(text_b_input_ids)#[128, 64] ->[128, 64, 100]

        #text_a_attention_mask 从[128, 64]->[128]
        #text_a_vecs从[128, 64, 100]->[775, 100],pack_padded_sequence将张量变成一个PackedSequence，这个类的第一个属性是将[128,64,100]中不为0的数据进行重排得到[775,100]
        #这个类的第二个属性是一个长度为11的一维张量  [128, 128, 128,  90,  70,  57,  57,  57,  20,  20,  20] 代表batch中128个样本都有第一位，128都有第二位，。。。，20个样本有十位，20个样本有十一位
        text_a_vecs = pack_padded_sequence(text_a_vecs, text_a_attention_mask.cpu().long().sum(dim=-1),enforce_sorted=False, batch_first=True)
        text_b_vecs = pack_padded_sequence(text_b_vecs, text_b_attention_mask.cpu().long().sum(dim=-1),enforce_sorted=False, batch_first=True)
        # pack_padded_sequence 将text_a_attention_mask 从[128, 64]->[775]，然后pad_packed_sequence将将text_a_attention_mask从[775]->[128, 11]
        text_a_attention_mask = pack_padded_sequence(text_a_attention_mask,text_a_attention_mask.cpu().long().sum(dim=-1),enforce_sorted=False, batch_first=True)
        text_b_attention_mask = pack_padded_sequence(text_b_attention_mask,text_b_attention_mask.cpu().long().sum(dim=-1),enforce_sorted=False, batch_first=True)
        text_a_attention_mask, _ = pad_packed_sequence(text_a_attention_mask, batch_first=True)
        text_b_attention_mask, _ = pad_packed_sequence(text_b_attention_mask, batch_first=True)

        # 两个query先独自经过encoder
        #text_a_vecs从[775,100]->[128,11,200],其中11代表了128中最长的序列，text_a_lens[128]代表每个序列的长度
        text_a_vecs, text_a_lens = self.encoder(text_a_vecs, text_a_attention_mask)
        text_b_vecs, text_b_lens = self.encoder(text_b_vecs, text_b_attention_mask)

        # 两个query通过Attention进行交互
        #text_a_vecs [128,11,200],text_a_attention_mask [128,11]
        text_a_vecs, text_b_vecs = self.cross_attn(text_a_vecs, text_b_vecs, text_a_attention_mask,text_b_attention_mask)
        text_a_vecs = pack_padded_sequence(text_a_vecs, text_a_lens, enforce_sorted=False, batch_first=True)
        text_b_vecs = pack_padded_sequence(text_b_vecs, text_b_lens, enforce_sorted=False, batch_first=True)

        # 融合当前query的hidden states和attention后的hidden states的信息
        text_a_vec = self.decoder(text_a_vecs, text_a_attention_mask)
        text_b_vec = self.decoder(text_b_vecs, text_b_attention_mask)

        # 拼接两个query的表示，输入到分类器
        pooler_output = torch.cat([text_a_vec, text_b_vec], dim=-1)
        logits = self.classifier(pooler_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        return (loss, logits) if loss is not None else logits



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
model = SemAttn(in_feat=in_feat,num_labels=len(num_labels),dropout_prob=dropout_prob,w2v_state_dict=w2v_model)
model.to('cpu')
a = model(**data)
print(a)