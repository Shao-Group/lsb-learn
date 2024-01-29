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

#################parameters of k, m and others################
out_dim = 5000
num_b = 50
m_dim = 100
flat_dim = 7488
batch_size = 5000
th_x = 0.5
######################################

cnnk = torch.load('~/incp12x2'+str(num_b)+'k_'+str(m_dim)+'m_('+str(d1)+'-'+str(d2)+')s.pt')
siacnn = torch.load('~/siacnn12x2_'+str(num_b)+'k_'+str(m_dim)+'m_('+str(d1)+'-'+str(d2)+')s.pt')

#############################################
print('Results n = 100:')

acc = acc_test_batch(train_a, train_b, train_y, batch_size, siacnn, th_x, m_dim, num_b, device)
acc_t = acc_test_batch(test_a, test_b, test_y, batch_size, siacnn, th_x, m_dim, num_b, device)

print('acc'+str(d1)+'-'+str(d2)+'eds_'+str(num_b)+'k_'+str(m_dim)+'m')

print('acc_train:', acc)
print('acc_test:', acc_t)

print('###################BD output###############################')

print('bd_acc_('+str(d1)+','+str(d2)+')-eds_'+str(num_b)+'k_'+str(m_dim)+'m'+'_train')
eds = ed_sp(df1)
print('edits number')
ed_num_train = []
for i in sorted(eds.keys()):
    print('ed = ', i, ': ', len(eds[i]))
    ed_num_train.append([i, len(eds[i])])

res = breakdown_acc(eds, d1, d2, siacnn, batch_size, th_x, m_dim, num_b, device)

####
print('bd_acc_('+str(d1)+','+str(d2)+')-eds_'+str(num_b)+'k_'+str(m_dim)+'m'+'_test')
eds = ed_sp(df2)
print('edits number')
ed_num_test = []
for i in sorted(eds.keys()):
    print('ed = ', i, ': ', len(eds[i]))
    ed_num_test.append([i, len(eds[i])])

res = breakdown_acc(eds, d1, d2, siacnn, batch_size, th_x, m_dim, num_b, device)
