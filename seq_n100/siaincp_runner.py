import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import random
import h5py

import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np

from siacnn_models_gpu2 import *

#################parameters of d1, d2################
d1 = 5
d2 = 6

dataset = data_reader1(35, d1, d2, '~/seq-n100-ED35-1.txt', 0, 10000)
dataset2 = data_reader2(35, d1, d2, '~/seq-n100-ED35-2.txt', 0, 10000)

random.shuffle(dataset)
df = pd.DataFrame(data=dataset[0:11000])

df1 = pd.DataFrame(data=dataset[0:10000])
df2 = pd.DataFrame(data=dataset2)

seq_a, seq_b, labels_, labels = aby_sep(df)
test_a, test_b, test_t, test_y = aby_sep(df2)

train_a = seq_a[0:10000]
train_b = seq_b[0:10000]
train_t = torch.tensor(labels_[0:10000])
train_y = torch.tensor(labels[0:10000])

valid_a = seq_a[10000:]
valid_b = seq_b[10000:]
valid_t = torch.tensor(labels_[10000:11000])
valid_y = torch.tensor(labels[10000:11000])

test_t = torch.tensor(test_t)
test_y = torch.tensor(test_y)

print('training samples number:'+str(len(train_a)))
print('testing samples number:'+str(len(test_a)))

#################parameters of k, m################
out_dim = 5000
num_b = 50
m_dim = 100
flat_dim = 7488
batch_size = 5000

device = torch.device('cuda:0')
inpm = Inp_Model_2().to(device)
siacnn = SiameseCNN(inpm, flat_dim, out_dim).to(device)

trainer1 = Trainer1(train_a, train_b, train_t, siacnn, hash_loss0, batch_size)

lr = 0.001
num_epo = 30
loss_t, loss_v = trainer1.run(num_epo, lr, valid_a, valid_b, valid_t, m_dim, num_b, device)
for i in range(3):
    lr *= 0.5
    loss1_, loss11_ = trainer1.run(num_epo, lr, valid_a, valid_b, valid_t, m_dim, num_b, device)
    loss_t += loss1_
    loss_v += loss11_

num_epo = 50
for i in range(4):
    lr *= 0.5
    loss1_, loss11_ = trainer1.run(num_epo, lr, valid_a, valid_b, valid_t, m_dim, num_b, device)
    loss_t += loss1_
    loss_v += loss11_


torch.save(inpm, '~/incp12x2_'+str(num_b)+'k_'+str(m_dim)+'m_('+str(d1)+'-'+str(d2)+')s.pt')
torch.save(siacnn, '~/siacnn12x2_'+str(num_b)+'k_'+str(m_dim)+'m_('+str(d1)+'-'+str(d2)+')s.pt')


