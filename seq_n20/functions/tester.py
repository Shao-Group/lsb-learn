import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import random
import h5py

import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import editdistance as ed
from siacnn_models_gpu import *

#################parameters of d1, d2################
d1 = 1
d2 = 2

dataset = data_reader1(15, d1, d2, '~/seq-n20-ED15-1.txt', 0, 10000)
dataset2 = data_reader1(15, d1, d2, '~/seq-n20-ED15-2.txt', 0, 10000)

random.shuffle(dataset)
df = pd.DataFrame(data=dataset[0:1100000])

df1 = pd.DataFrame(data=dataset[0:1000000])
df2 = pd.DataFrame(data=dataset2)

seq_a, seq_b, labels_, labels = aby_sep(df)
test_a, test_b, test_t, test_y = aby_sep(df2)

train_a = seq_a[0:1000000]
train_b = seq_b[0:1000000]
train_t = torch.tensor(labels_[0:1000000])
train_y = torch.tensor(labels[0:1000000])

valid_a = seq_a[1000000:]
valid_b = seq_b[1000000:]
valid_t = torch.tensor(labels_[1000000:1100000])
valid_y = torch.tensor(labels[1000000:1100000])

test_t = torch.tensor(test_t)
test_y = torch.tensor(test_y)

#################parameters of k, m and others################
out_dim = 800
num_b = 20
m_dim = 40
flat_dim = 2320
batch_size = 20000
th_x = 0.5

######################################

cnnk = torch.load('~/cnn2_'+str(num_b)+'k_'+str(m_dim)+'m_('+str(d1)+'-'+str(d2)+')s.pt')
siacnn = torch.load('~/siacnn2_'+str(num_b)+'k_'+str(m_dim)+'m_('+str(d1)+'-'+str(d2)+')s.pt')

#######################################TESTING############################################################
print('Results n = 20:')

acc = acc_test_batch(train_a, train_b, train_y, 30000, siacnn, th_x, m_dim, num_b, device)
acc_t = acc_test_batch(test_a, test_b, test_y, 30000, siacnn, th_x, m_dim, num_b, device)

print('acc_'+str(d1)+'-'+str(d2)+'eds_'+str(num_b)+'k_'+str(m_dim)+'m_bl')

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

