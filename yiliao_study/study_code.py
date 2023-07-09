import torch

batch_size = 3   # 这个batch有3个序列
max_len = 6       # 最长序列的长度是6
embedding_size = 8 # 嵌入向量大小8
hidden_size = 16   # 隐藏向量大小16
vocab_size = 20    # 词汇表大小20

input_seq = [[3, 5, 12, 7, 2, ], [4, 11, 14, ], [18, 7, 3, 8, 5, 4]]
lengths = [5, 3, 6]   # batch中每个seq的有效长度。
# embedding
embedding = torch.nn.Embedding(vocab_size, embedding_size, padding_idx=0)
# LSTM的RNN循环神经网络
lstm = torch.nn.LSTM(embedding_size, hidden_size)


#由大到小排序
input_seq = sorted(input_seq, key = lambda tp: len(tp), reverse=True)
#input_seq从[[3, 5, 12, 7, 2, ], [4, 11, 14, ], [18, 7, 3, 8, 5, 4]] 变到  [[18, 7, 3, 8, 5, 4], [3, 5, 12, 7, 2], [4, 11, 14]]
lengths = sorted(lengths, key = lambda tp: tp, reverse=True)
#lengths从[5, 3, 6] 变到  [6, 5, 3]


PAD_token = 0 # 填充下标是0
def pad_seq(seq, seq_len, max_length):
    seq = seq
    seq += [PAD_token for _ in range(max_length - seq_len)]
    return seq

pad_seqs = []  # 填充后的数据
for i,j in zip(input_seq, lengths):
    pad_seqs.append(pad_seq(i, j, max_len))

#input_seq从[[18, 7, 3, 8, 5, 4], [3, 5, 12, 7, 2], [4, 11, 14]]变到[[18, 7, 3, 8, 5, 4], [3, 5, 12, 7, 2, 0], [4, 11, 14, 0, 0, 0]]
pad_seqs = torch.tensor(pad_seqs)
embeded = embedding(pad_seqs)
#input_seq从[3, 6]变成[3, 6, 8]

# 压缩，设置batch_first为true
pack = torch.nn.utils.rnn.pack_padded_sequence(embeded, lengths, batch_first=True)
#这里如果不写batch_first,你的数据必须是[sequence_length,batch_size,embedding]，不然会报错lenghth错误
# pack[0].shape 为 [14, 8] ，可以理解为1,6,8 + 1,5,8 + 1,3,8 -> 14,8
# pack[1].shape 为 [6],可以裂解为[18, 7, 3, 8, 5, 4], [3, 5, 12, 7, 2, 0], [4, 11, 14, 0, 0, 0] 变成[3,3,3,2,2,1]


#　利用lstm循环神经网络测试结果
state = None
pade_outputs, _ = lstm(pack, state)
# pack[0].shape 为 [14,16] ，
# pack[1].shape 为 [6]

# 设置batch_first为true;你可以不设置为true,为false时候只影响结构不影响结果
pade_outputs, others = torch.nn.utils.rnn.pad_packed_sequence(pade_outputs, batch_first=True)
# pade_outputs[0].shape 为 [3,6,16]
# pade_outputs[1].shape 为 [6,5,3]

pade_outputs1, _ = lstm(embeded, state)

pade_outputs.shape#[3, 6, 16],bs,sequence,hidden_size
pade_outputs1.shape
pade_outputs[0]
pade_outputs1[0]