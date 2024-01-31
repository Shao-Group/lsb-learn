import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import random

import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np

#######################################DATA READER######################################################

def data_reader1(eds, d1, d2, id_, i_ranga, i_rangb):
    with open(id_, 'r') as f:
        lines = f.readlines()
    for i in range(len(lines)):
        lines[i] = lines[i].strip('\n').split(' ')
    ed = [i+1 for i in range(eds)]
    set_all = []
    for j in range(len(ed)):
        set_ = []
        for i in range(len(lines)):
            if int(lines[i][2]) == ed[j]:
                set_.append([lines[i][0], lines[i][1], lines[i][2]])
        set_all.append(set_)
    Po = []
    Ne = []
    for i in range(len(set_all)):
        if i<=d1-1:
            Po += set_all[i][i_ranga:i_rangb]
        elif i>=d2-1:
            Ne += set_all[i][i_ranga:i_rangb]
    for i in range(len(Po)):
        Po[i].append(-1)
    for i in range(len(Ne)):
        Ne[i].append(1)

    dataset = Po+Ne
    random.shuffle(dataset)
    #df = pd.DataFrame(data=dataset)
    return dataset

def leng_fea(seq):
    leng = []
    A = torch.zeros(4)
    C = torch.zeros(4)
    G = torch.zeros(4)
    T = torch.zeros(4)
    A[0] += 1
    C[1] += 1
    G[2] += 1
    T[3] += 1
    fea = [A, C, G, T]
    word = ['A', 'C', 'G', 'T']

    for i in range(len(seq)):
        for j in range(len(word)):
            if seq[i] == word[j]:
                leng.append(fea[j])

    leng = torch.stack(leng)

    return leng

def aby_sep(df):

    labels_ = list(df[3])
    labels = []
    for i in range(len(labels_)):
        if labels_[i] == 1:
            labels.append(0)
        else:
            labels.append(1)
    #labels = torch.tensor(labels)

    samples_a = df[0]
    samples_b = df[1]
    seq_a = []
    for i in range(len(samples_a)):
        seq_a.append(leng_fea(samples_a[i]))

    seq_b = []
    for i in range(len(samples_b)):
        seq_b.append(leng_fea(samples_b[i]))

    return seq_a, seq_b, labels_, labels 

#######################################################MODELS##################################################################

#one-layer Inp_module
class CNN2_kmer_mp_8(nn.Module):
    def __init__(self):#out_size,
        super(CNN2_kmer_mp_8, self).__init__()

        self.cnn1 = nn.Conv2d(1, 20, (4, 2), stride=1)
        self.cnn2 = nn.Conv2d(1, 20, (4, 3), stride=1)
        self.cnn3 = nn.Conv2d(1, 20, (4, 4), stride=1)
        self.cnn4 = nn.Conv2d(1, 20, (4, 5), stride=1)
        self.cnn5 = nn.Conv2d(1, 20, (4, 6), stride=1)
        self.cnn6 = nn.Conv2d(1, 20, (4, 7), stride=1)
        self.cnn7 = nn.Conv2d(1, 20, (4, 8), stride=1)
        self.cnn8 = nn.Conv2d(1, 20, (4, 9), stride=1)

        self.maxpool = nn.MaxPool2d((1, 2), stride=(1, 1))
        self.flat = nn.Flatten()

    def forward(self, x):
        out1 = F.relu(self.cnn1(x))
        out1 = self.maxpool(out1)
        out2 = F.relu(self.cnn2(x))
        out2 = self.maxpool(out2)
        out3 = F.relu(self.cnn3(x))
        out3 = self.maxpool(out3)
        out4 = F.relu(self.cnn4(x))
        out4 = self.maxpool(out4)
        out5 = F.relu(self.cnn5(x))
        out5 = self.maxpool(out5)
        out6 = F.relu(self.cnn6(x))
        out6 = self.maxpool(out6)
        out7 = F.relu(self.cnn7(x))
        out7 = self.maxpool(out7)
        out8 = F.relu(self.cnn8(x))
        out8 = self.maxpool(out8)

        out1 = self.flat(out1)
        out2 = self.flat(out2)
        out3 = self.flat(out3)
        out4 = self.flat(out4)

        out5 = self.flat(out5)
        out6 = self.flat(out6)
        out7 = self.flat(out7)
        out8 = self.flat(out8)

        out = F.normalize(torch.cat((out1, out2, out3, out4, out5, out6, out7, out8), 1))
        #out = self.flat(out)

        return out

#Multi-layer Inp_module

class Inp_Layer(nn.Module):
    def __init__(self, in_ch, out_ch, in_dim, out_max):#out_size,
        super(Inp_Layer, self).__init__()

        self.cnn1 = nn.Conv2d(in_ch, out_ch, (in_dim, 2), stride=1, padding="same")
        self.cnn2 = nn.Conv2d(in_ch, out_ch, (in_dim, 3), stride=1, padding="same")
        self.cnn3 = nn.Conv2d(in_ch, out_ch, (in_dim, 4), stride=1, padding="same")
        self.cnn4 = nn.Conv2d(in_ch, out_ch, (in_dim, 5), stride=1, padding="same")
        self.maxpool = nn.MaxPool2d((1, out_max), stride=(1, 1))
        self.flat = nn.Flatten()

    def forward(self, x):

        out1 = F.relu(self.cnn1(x))
        out1 = self.maxpool(out1)
        out2 = F.relu(self.cnn2(x))
        out2 = self.maxpool(out2)
        out3 = F.relu(self.cnn3(x))
        out3 = self.maxpool(out3)
        out4 = F.relu(self.cnn4(x))
        out4 = self.maxpool(out4)

        out = F.normalize(torch.cat((out1, out2, out3, out4), 1))

        return out  

class Inp_Layer2(nn.Module):
    def __init__(self, in_ch, out_ch, in_dim):#out_size,
        super(Inp_Layer2, self).__init__()

        self.cnn1 = nn.Conv2d(in_ch, out_ch, (in_dim, 2), stride=1, padding="same")
        self.cnn2 = nn.Conv2d(in_ch, out_ch, (in_dim, 3), stride=1, padding="same")
        self.cnn3 = nn.Conv2d(in_ch, out_ch, (in_dim, 4), stride=1, padding="same")
        self.cnn4 = nn.Conv2d(in_ch, out_ch, (in_dim, 5), stride=1, padding="same")
        #self.maxpool = nn.MaxPool2d((1, out_max), stride=(1, 1))
        self.flat = nn.Flatten()

    def forward(self, x):

        out1 = F.relu(self.cnn1(x))
        out2 = F.relu(self.cnn2(x))
        out3 = F.relu(self.cnn3(x))
        out4 = F.relu(self.cnn4(x))

        out = F.normalize(torch.cat((out1, out2, out3, out4), 1))

        return out

class Inp_Model_4(nn.Module):
    def __init__(self):#out_size,
        super(Inp_Model_4, self).__init__()

        self.inp_layer1 = Inp_Layer(1, 5, 4, 2)
        self.inp_layer2 = Inp_Layer(20, 10, 4, 2)
        self.inp_layer3 = Inp_Layer(40, 10, 4, 2)
        self.inp_layer4 = Inp_Layer(40, 10, 4, 2)

        self.flat = nn.Flatten()

    def forward(self, x):

        out = self.inp_layer1(x)
        out = self.inp_layer2(out)
        out = self.inp_layer3(out)
        out = self.inp_layer4(out)
        out = self.flat(out)

        return out

class Inp_Model_2(nn.Module):
    def __init__(self):#out_size,
        super(Inp_Model_2, self).__init__()

        self.inp_layer1 = Inp_Layer(1, 10, 4, 5)
        self.inp_layer2 = Inp_Layer(40, 10, 4, 3)
        #self.inp_layer3 = Inp_Layer(20, 5, 1, 2)
        #self.inp_layer4 = Inp_Layer(20, 5, 1, 2)

        self.flat = nn.Flatten()

    def forward(self, x):

        out = self.inp_layer1(x)
        out = self.inp_layer2(out)
        #out = self.inp_layer3(out)
        #out = self.inp_layer4(out)
        out = self.flat(out)

        return out

###########################################Siamese model##############################################################
def ste_round(x):
    return torch.round(x) - x.detach() + x

class SiameseCNN(nn.Module):
    def __init__(self, cnn, flat_dim, out_size):
        super(SiameseCNN, self).__init__()
        self.cnn = cnn
        self.fc1 = nn.Sequential(
            nn.Linear(flat_dim, out_size),
            nn.Sigmoid()
            #nn.Tanh()
        )

    def forward_once(self, x):
        out = self.cnn(x)
        out = self.fc1(out)

        return out

    def forward(self, x1, x2):
        out1 = self.forward_once(x1)
        out2 = self.forward_once(x2)

        return out1, out2

class SiameseCNN_s(nn.Module):
    def __init__(self, cnn, flat_dim, out_size):
        super(SiameseCNN_s, self).__init__()
        self.cnn = cnn
        self.fc1 = nn.Sequential(
            nn.Linear(flat_dim, out_size),
            #nn.Sigmoid()
            #nn.Tanh()
        )
        self.bn = nn.BatchNorm1d(out_size)

    def forward_once(self, x):
        out = self.cnn(x)
        out = self.bn(self.fc1(out))
        out = F.sigmoid(out)        
        return out

    def forward(self, x1, x2):
        out1 = ste_round(self.forward_once(x1))
        out2 = ste_round(self.forward_once(x2))

        return out1, out2


class SiameseCNN_r(nn.Module):
    def __init__(self, cnn, flat_dim, out_size):
        super(SiameseCNN_r, self).__init__()
        self.cnn = cnn
        self.fc1 = nn.Sequential(
            nn.Linear(flat_dim, out_size),
            #nn.Sigmoid()
            #nn.Tanh()
        )
        self.bn = nn.BatchNorm1d(out_size)

    def forward_once(self, x):
        out = self.cnn(x)
        #out = self.bn(self.fc1(out))
        out = self.fc1(out)
        out = ste_round(out)
        return out

    def forward(self, x1, x2):
        out1 = ste_round(self.forward_once(x1))
        out2 = ste_round(self.forward_once(x2))

        return out1, out2

class SiameseCNN_b(nn.Module):
    def __init__(self, cnn, flat_dim, out_size):
        super(SiameseCNN_b, self).__init__()
        self.cnn = cnn
        self.fc1 = nn.Sequential(
            nn.Linear(flat_dim, out_size),
            #nn.Sigmoid()
            #nn.Tanh()
        )

    def forward_once(self, x):
        out = self.cnn(x)
        out = self.fc1(out)

        return out

    def forward(self, x1, x2):
        out1 = self.forward_once(x1)
        out2 = self.forward_once(x2)

        return out1, out2


class SiameseCNN_sr(nn.Module):
    def __init__(self, cnn, flat_dim, out_size):
        super(SiameseCNN_sr2, self).__init__()
        self.cnn = cnn
        self.fc1 = nn.Sequential(
            nn.Linear(flat_dim, out_size),
            nn.Sigmoid()
            #nn.Tanh()
        )
        self.bn = nn.BatchNorm1d(out_size)

    def forward_once(self, x):
        out = self.cnn(x)
        out = self.bn(self.fc1(out))
        out = ste_round(out)       
        return out

    def forward(self, x1, x2):
        out1 = self.forward_once(x1)
        out2 = self.forward_once(x2)

        return out1, out2

#############################loss_function#########################################################################
#2norm
def hash_loss0(a, b, t, x_0, m_dim, batch_size, num_b):

    hash_a = a.reshape([batch_size, m_dim, num_b])
    hash_b = b.reshape([batch_size, m_dim, num_b])
    ed = torch.norm(hash_a - hash_b, p=2, dim=1)
    d = torch.min(ed, 1).values
    loss = torch.max(x_0, 1-2*(d-0.5)*t)
    loss_ = torch.mean(loss)
    return loss_, ed, d, loss

#max-norm
def hash_loss1(a, b, t, x_0, m_dim, batch_size, num_b):

    hash_a = a.reshape([batch_size, m_dim, num_b])
    hash_b = b.reshape([batch_size, m_dim, num_b])
    ed = torch.linalg.norm(hash_a - hash_b, dim=1, ord=float('inf'))
    d = torch.min(ed, 1).values
    loss = torch.max(x_0, 1-2*(d-0.5)*t)
    loss_ = torch.mean(loss)
    return loss_, ed, d, loss

def hash_loss0_l(a, b, t, x_0, batch_size, num_b):

    ed = torch.norm(a - b, p=2, dim=0)
    d = torch.min(ed, 1).values-0.5
    loss = torch.max(x_0, 1-2*(d)*t)
    loss_ = torch.mean(loss)
    return loss_, ed, d, loss

#################################################Mini-batch###############################################################

def mini_batch_cnn1(x, i, batch_size):
    x_range = x[i*batch_size : (i+1)*batch_size]
    batch_x = torch.cat([torch.unsqueeze(i.t(), 0) for i in x_range], 0)
    batch_x = torch.unsqueeze(batch_x, 1)
    return batch_x


#################################################TRAINER###############################################################
class Trainer1:
    def __init__(self, x_a, x_b, t, model, loss_f, batchsize):
        super(Trainer1, self).__init__()
        self.x_a = x_a
        self.x_b = x_b
        self.t = t
        self.model = model
        self.loss_f = loss_f
        self.batch_size = batchsize        

    def run(self, epo, lr, v_a, v_b, v_t, m_dim, num_b, device):#, state1

        loss_ = []
        loss1_ = []
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=0.0001)
        for epoch in range(epo):
            optimizer.zero_grad()
            #D_t = 0
            final_loss_t = 0
            for i in range(int(len(self.x_a)/self.batch_size)):
                mini_inputa = mini_batch_cnn1(self.x_a, i, self.batch_size).to(device)
                mini_inputb = mini_batch_cnn1(self.x_b, i, self.batch_size).to(device)
                batch_t = self.t[i*self.batch_size : (i+1)*self.batch_size].to(device)
                #batch_y = self.y[i:i+batch_size]
           
                out1, out2 = self.model(mini_inputa, mini_inputb)
                #loss = hash_loss(out1, out2, batch_y)
                loss, _, d_t, _ = self.loss_f(out1, out2, batch_t, torch.zeros(self.batch_size).to(device), m_dim, self.batch_size, num_b)
                loss.backward()
                optimizer.step()
                final_loss_t += float(loss)
            #print('epoch:', epoch, ' loss:', float(loss_all))
            #D_v = 0
            final_loss_v = 0
            for i in range(int(len(v_a)/self.batch_size)):
                mini_inputa = mini_batch_cnn1(v_a, i, self.batch_size).to(device)
                mini_inputb = mini_batch_cnn1(v_b, i, self.batch_size).to(device)
                batch_t = v_t[i*self.batch_size : (i+1)*self.batch_size].to(device)
                #batch_y = v_y[i:i+batch_size]
            
                out1, out2 = self.model(mini_inputa, mini_inputb)
                #loss = hash_loss(out1, out2, batch_y)
                loss1, _, d_v, _ = self.loss_f(out1, out2, batch_t, torch.zeros(self.batch_size).to(device), m_dim, self.batch_size, num_b)
                final_loss_v += float(loss1)

            loss_.append(final_loss_t/int(len(self.x_a)/self.batch_size))
            loss1_.append(final_loss_v/int(len(v_a)/self.batch_size))
            print('epoch:', epoch, 'loss_t:', float(final_loss_t/int(len(self.x_a)/self.batch_size)), 'loss_v:', float(final_loss_v/int(len(v_a)/self.batch_size)))

        return loss_, loss1_


##########################################Accuracy######################################################################################################

def round_0(x, th, device):
    x_n = torch.where(x < th, torch.tensor(0).to(device), torch.tensor(1).to(device)) 
    return x_n

def acc_count0(a, b, th, m_dim, num_b, device):
    
    a = a.reshape([m_dim, num_b]).T
    b = b.reshape([m_dim, num_b]).T
    
    equal_tensor = torch.eq(round_0(a, th, device), round_0(b, th, device))
    for tensor in equal_tensor:
        all_true = torch.all(tensor)
        if all_true:
            return 1
    return 0

def acc_fun0(a, b, y, th, m_dim, num_b, device):
    score = [acc_count0(a[i], b[i], th, m_dim, num_b, device) for i in range(a.shape[0])]
    eq_sc = torch.eq(torch.tensor(score).to(device), y)
    equal_to_T = (eq_sc == True)
    count = torch.sum(equal_to_T).tolist()

    return score, count/int(len(y))


def hash_coding(a_set, th, m_dim, num_b, device):

    a_set = a_set.reshape([len(a_set), m_dim, num_b])
    a_set = a_set.transpose(-2, -1)
    hash_a = round_0(a_set, th, device)
    return hash_a

###########Breakdown Acc##########
import editdistance as ed

def hamming_distance(chaine1, chaine2):
    return sum(c1 != c2 for c1, c2 in zip(chaine1, chaine2))

def ed_sp(df):
    ed_dict = {}
    for i in range(len(df[1])):
        if int(df[2][i]) not in ed_dict.keys():
            ed_dict[int(df[2][i])] = [[df[0][i], df[1][i]]]
        else:
            ed_dict[int(df[2][i])].append([df[0][i], df[1][i]])

    return ed_dict

#############evaluate1########## 
def acc_test(test_a, test_b, test_y, siacnn, th_x, m_dim, num_b, device):

    minit_inputa1 = mini_batch_cnn1(test_a, 0, len(test_a))
    minit_inputb1 = mini_batch_cnn1(test_b, 0, len(test_a))

    out1_t, out2_t = siacnn(minit_inputa1.to(device), minit_inputb1.to(device))

    _, acc = acc_fun0(out1_t, out2_t, test_y.to(device), th_x, m_dim, num_b, device)

    return acc

def acc_test_batch(test_a, test_b, test_y, batchsize, siacnn, th_x, m_dim, num_b, device):
    scores = []
    for i in range(int(len(test_a)/batchsize)):
        minit_inputa1 = mini_batch_cnn1(test_a, i, batchsize)
        minit_inputb1 = mini_batch_cnn1(test_b, i, batchsize)
        minit_y = test_y[i*batchsize : (i+1)*batchsize]

        out1_, out2_ = siacnn(minit_inputa1.to(device), minit_inputb1.to(device))
        score = [acc_count0(out1_[i], out2_[i], th_x, m_dim, num_b, device) for i in range(out1_.shape[0])]
        scores += score
    num1 = int(len(test_a)/batchsize)*batchsize
    num2 = len(test_a) - num1
    if num2 != 0:
        out1_, out2_ = siacnn(mini_batch_cnn1(test_a[num1:], 0, num2).to(device), mini_batch_cnn1(test_b[num1:], 0, num2).to(device))
        score = [acc_count0(out1_[i], out2_[i], th_x, m_dim, num_b, device) for i in range(out1_.shape[0])]
        #score, _ = acc_fun0(out1_, out2_, test_y[num1:].to(device), th_x, m_dim, num_b, device)
        scores += score
    equal_to_T = (torch.eq(torch.tensor(scores), test_y) == True)
    count = torch.sum(equal_to_T).tolist()
    
    return count/int(len(scores))


def breakdown_acc(eds, d1, d2, siacnn, batchsize, th_x, m_dim, num_b, device):
    res = {}
    for key in eds.keys():
        seq_a = []
        seq_b = []
        for j in range(len(eds[key])):
            seq_a.append(leng_fea(eds[key][j][0]))
            seq_b.append(leng_fea(eds[key][j][1]))
        if key<=d1:
            test_y = torch.ones(len(eds[key]))
        elif key>=d2:
            test_y = torch.zeros(len(eds[key]))
        if len(seq_a)>batchsize:
            acc = acc_test_batch(seq_a, seq_b, test_y, batchsize, siacnn, th_x, m_dim, num_b, device)
        else:
            acc = acc_test(seq_a, seq_b, test_y, siacnn, th_x, m_dim, num_b, device)
        res[key] = acc

    for i in sorted(res.keys()):
        print('ed = ', i, ' acc = ', res[i])

    return res 

