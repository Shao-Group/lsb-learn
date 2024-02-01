import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import random
import h5py

import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np

from siacnn_models_gpu import *

#setting d1, d2
d1 = 5
d2 = 6

dataset2 = data_reader2(35, d1, d2, '~/seq-n100-ED35-2.txt', 0, 50000)


df2 = pd.DataFrame(data=dataset2)

test_a, test_b, test_t, test_y = aby_sep(df2)

test_t = torch.tensor(test_t)
test_y = torch.tensor(test_y)

#setting of k, m, batchsize
num_b = 50
m_dim = 100
batch_size = 5000
th_x = 0.5
##################### Model uploading #################

cnnk = torch.load('~/incp8x3'+str(num_b)+'k_'+str(m_dim)+'m_('+str(d1)+'-'+str(d2)+')s.pt')
siacnn = torch.load('~/siacnn8x3_'+str(num_b)+'k_'+str(m_dim)+'m_('+str(d1)+'-'+str(d2)+')s.pt')

#######################################OUTPUT############################################################
print('Results n = 100:')

acc_t = acc_test_batch(test_a, test_b, test_y, batch_size, siacnn, th_x, m_dim, num_b, device)

print('acc'+str(d1)+'-'+str(d2)+'eds_'+str(num_b)+'k_'+str(m_dim)+'m')

print('acc_test:', acc_t)

print('###################BreakDown Acc###############################')

####
print('bd_acc_('+str(d1)+','+str(d2)+')-eds_'+str(num_b)+'k_'+str(m_dim)+'m'+'_test')
eds = ed_sp(df2)
print('edits number')
ed_num_test = []
for i in sorted(eds.keys()):
    print('ed = ', i, ': ', len(eds[i]))
    ed_num_test.append([i, len(eds[i])])

res = breakdown_acc(eds, d1, d2, siacnn, batch_size, th_x, m_dim, num_b, device)
