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

d1 = 2
d2 = 4

dataset1 = data_reader1(15, d1, d2, '/storage/home/xvy5180/work/seqhash/seq_20n/seq20n/seq-n20-ED15-1.txt', 0, 100000)
dataset3 = data_reader1(15, d1, d2, '/storage/home/xvy5180/work/seqhash/seq_20n/seq20n/seq-n20-ED15-3.txt', 0, 10000)
#dataset4 = data_reader1(15, d1, d2, '/storage/home/xvy5180/work/seqhash/seq_20n/seq20n/seq-n20-ED15-4.txt ', 0, 20000)

dataset2 = data_reader1(15, d1, d2, '/storage/home/xvy5180/work/seqhash/seq_20n/seq20n/seq-n20-ED15-2.txt', 0, 50000)

dataset = dataset1+dataset3
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

#################model_loading################
#hidden_dim = 1200
out_dim = 800
num_b = 20
m_dim = 40
flat_dim = 2320
batch_size = 20000
th_x = 0.5

######################################

device = torch.device('cuda:0')

cnnk = CNN2_kmer_mp_8().to(device)
siacnn = SiameseCNN_s(cnnk, flat_dim, out_dim).to(device)

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
for i in range(5):
    lr *= 0.5
    loss1_, loss11_ = trainer1.run(num_epo, lr, valid_a, valid_b, valid_t, m_dim, num_b, device)
    loss_t += loss1_
    loss_v += loss11_

torch.save(cnnk, '/storage/work/xvy5180/seqhash/seq_20n/cnn2_mp8_model/cnn2_'+str(num_b)+'k_'+str(m_dim)+'m_('+str(d1)+'-'+str(d2)+')s_10xs_bnsig.pt')
torch.save(siacnn, '/storage/work/xvy5180/seqhash/seq_20n/cnn2_mp8_model/siacnn2_'+str(num_b)+'k_'+str(m_dim)+'m_('+str(d1)+'-'+str(d2)+')s_10xs_bnsig.pt')

#######################################TESTING############################################################

acc = acc_test_batch(train_a, train_b, train_y, 30000, siacnn, th_x, m_dim, num_b, device)
acc_t = acc_test_batch(test_a, test_b, test_y, 30000, siacnn, th_x, m_dim, num_b, device)

print('acc_CNN2_8mp_'+str(d1)+'-'+str(d2)+'eds_'+str(num_b)+'k_'+str(m_dim)+'m_bl')

print('acc_train:', acc)
print('acc_test:', acc_t)

f = h5py.File('/storage/work/xvy5180/seqhash/seq_20n/cnn2_mp8_model/loss_acc_'+str(num_b)+'k_'+str(m_dim)+'m_('+str(d1)+'-'+str(d2)+')s_10xs_bnsig.hdf5', 'w')

f.create_dataset('loss_t', data=np.array(loss_t))
f.create_dataset('loss_v', data=np.array(loss_v))

f.create_dataset('acc', data=np.array([acc, acc_t]))

print('###################BD output###############################')

print('CNN2_8mp_'+str(d1)+'-'+str(d2)+'eds_'+str(num_b)+'k_'+str(m_dim)+'m_bl'+'_train')
eds = ed_sp(df1)
print('edits number')
ed_num_train = []
for i in sorted(eds.keys()):
    print('ed = ', i, ': ', len(eds[i]))
    ed_num_train.append([i, len(eds[i])])

res = breakdown_acc(eds, d1, d2, siacnn, batch_size, th_x, m_dim, num_b, device)

bd_res = []
for i in sorted(res.keys()):
    bd_res.append([i, res[i]])

f.create_dataset('bd_numb_train', data=np.array(np.array(ed_num_train)))
f.create_dataset('bd_train', data=np.array(np.array(bd_res)))

####
print('CNN2_8mp_'+str(d1)+'-'+str(d2)+'eds_'+str(num_b)+'k_'+str(m_dim)+'m_bl'+'_test')
eds = ed_sp(df2)
print('edits number')
ed_num_test = []
for i in sorted(eds.keys()):
    print('ed = ', i, ': ', len(eds[i]))
    ed_num_test.append([i, len(eds[i])])

res = breakdown_acc(eds, d1, d2, siacnn, batch_size, th_x, m_dim, num_b, device)

bd_res = []
for i in sorted(res.keys()):
    bd_res.append([i, res[i]])

f.create_dataset('bd_numb_test', data=np.array(np.array(ed_num_test)))
f.create_dataset('bd_test', data=np.array(np.array(bd_res)))

f.close()

