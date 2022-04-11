#!/usr/bin/env python
# coding: utf-8

# In[1]:

import pandas as pd
import os
import numpy as np
import torch
from torch.autograd import Variable
from sklearn import preprocessing
import copy
# https://blog.csdn.net/Muzi_Water/article/details/103921115


# In[2]:


sequence_length = 20
batch_size = 16
print(torch.cuda.is_available())


# In[3]:


x_train_path = r'data/HQZX/train'
x_train_path_txt = r'./washed_data/good_data_list.txt'

with open(x_train_path_txt,'r') as f:
    train_csv_pathes = f.read().split('\n')
# print(train_csv_pathes)


import math
def geo2xyz(lat, lng, r=6400):
    '''
    将地理经纬度转换成笛卡尔坐标系
    :param lat: 纬度
    :param lng: 经度
    :param r: 地球半径
    :return: 返回笛卡尔坐标系
    '''
    thera = (math.pi * lat) / 180
    fie = (math.pi * lng) / 180
    x = r * math.cos(thera) * math.cos(fie)
    y = r * math.cos(thera) * math.sin(fie)
    z = r * math.sin(thera)
    return [x,y,z]

def get_angle(l1, l2, l3):
    '''
    :param l1: 经纬度
    :param l2: 顶点经纬度
    :param l3: 经纬度
    :return: 线段l2-l1 与 l2-l3之间的角度
    '''
    p1 = geo2xyz(l1[0], l1[1])
    p2 = geo2xyz(l2[0], l2[1])
    p3 = geo2xyz(l3[0], l3[1])

    _P1P2 = math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2 + (p2[2] - p1[2]) ** 2)
    _P2P3 = math.sqrt((p3[0] - p2[0]) ** 2 + (p3[1] - p2[1]) ** 2 + (p3[2] - p2[2]) ** 2)
    if _P1P2==0 or _P2P3 == 0:
        return 180.0
    P = (p1[0] - p2[0]) * (p3[0] - p2[0]) + (p1[1] - p2[1]) * (p3[1] - p2[1]) + (p1[2] - p2[2]) * (p3[2] - p2[2])
#     print(P / (_P1P2 * _P2P3))
    P_ = P / (_P1P2 * _P2P3)
    if P_ > 1 or P_ < -1:
        if P_>1:
            P_=1
        else:
            P_=-1
    angle = (math.acos(P_) / math.pi) * 180.0
    return angle


# In[6]:


def wash_data(csv):
    data0 = csv.loc[0]
    for i in range(len(csv))[1:-1]:
#         print(i)
        data = csv.loc[i]
        l1 = [(data0)['经度'],(data0)['纬度'] ]
        l2 = [(data)['经度'],(data)['纬度'] ]
        l3 = [(data)['经度2'],(data)['纬度2'] ]
        if get_angle(l1,l2,l3) < 90:
            csv = csv.drop(i)
        data0 = data
    return csv

train_data_list = []
print("开始读取数据！")
tot = 0

for i,path in enumerate(train_csv_pathes):
#     print(os.path.join(x_train_path,path))
    s_data = pd.read_csv(os.path.join(x_train_path,path))
#     if s_data.std()['经度'] < 1e-4 :
#         continue

    s_data = s_data.drop(s_data[s_data['间隔时间'] == 0 ].index)
    s_data = s_data.drop(s_data[s_data['间隔距离'] == 0 ].index)
#     s_data.reset_index(drop=True,inplace=True)
#     s_data = wash_data(s_data)
    save_length = int(int(len(s_data)/(batch_size*sequence_length)) * (batch_size*sequence_length))
    if save_length ==0 :
        continue
    s_data = s_data[:save_length]
    tot += len(s_data)
    train_data_list.append(s_data)


print("读取数据完成！")
# In[10]:


print("总共有{}条数据，{}个csv文件".format(tot,len(train_data_list)))


# In[11]:


# import os
# if not os.path.isdir('washed_data'):
#         os.mkdir('washed_data')



# In[19]:




# In[14]:


x_train_list = []
y_train_list = []
for train_data in train_data_list:
    x_train_list.append(pd.DataFrame(train_data,columns=['纬度','经度','间隔时间']))
    y_train_list.append(pd.DataFrame(train_data,columns=['纬度2','经度2']))


# In[15]:


x_dim = len(x_train_list[0].columns)
y_dim = len(y_train_list[0].columns)
# print(x_dim,y_dim)



Scaler = preprocessing.MinMaxScaler()
for i in range(len(train_data_list)):
#     print(x_train_list[i].shape)
    x_train_list[i] = Scaler.fit_transform(x_train_list[i])
    y_train_list[i] = Scaler.fit_transform(y_train_list[i])


# In[ ]:


for i in range(len(x_train_list)):
    x_train_list[i] = torch.tensor(np.array(x_train_list[i]))                        .view(-1,sequence_length,batch_size,x_dim).to(torch.float32).cuda()
    y_train_list[i] = torch.tensor(np.array(y_train_list[i]))                        .view(-1,sequence_length,batch_size,y_dim).to(torch.float32).cuda()
x_train_list[0].shape


# In[ ]:


import torch
import torch.nn as nn


# In[ ]:


input_size = x_dim
hidden_size = 10
output_size = 2


# In[ ]:


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=2
        )
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, h_state = self.rnn(x)

        out = out[-1, :, :]
#         print(out)
        out = self.linear(out) #=> out[batch_size*seq,hidden_size] --> [batch_size*seq,output_size]
        out = out.unsqueeze(dim=0) # [1,,]
        return out




def test(epoch):
    # In[ ]:

    model.eval()


    global best_score

    import os
    import pandas as pd
    x_test_path = r'data/HQZX/test/'
    test_csv_pathes = os.listdir(x_test_path)


    # In[ ]:


    for path in test_csv_pathes:
        if path.split('.')[1] != 'csv':
            continue
        single_csv_data = pd.read_csv(os.path.join(x_test_path,path),engine='python')
        if path == test_csv_pathes[1]:
            test_data = single_csv_data
        else:
            test_data = pd.concat([test_data,single_csv_data])


    # In[ ]:


    x_test0 = (pd.DataFrame(test_data,columns=['纬度','经度','间隔时间']))
    y_test0 = (pd.DataFrame(test_data,columns=['纬度2','经度2']))

    x_dim = len(x_test0.columns)
    y_dim = len(y_test0.columns)

    x_test0 = Scaler.fit_transform(x_test0)
    y_test0 = Scaler.fit_transform(y_test0)


    # In[ ]:


    save_length = int(int(len(x_test0)/(sequence_length*batch_size)) * (sequence_length*batch_size))
    x_test = x_test0[:save_length]
    y_test = y_test0[:save_length]

    x_test = torch.tensor(np.array(x_test)).view(-1,sequence_length,batch_size,x_dim).to(torch.float32).cuda()
    y_test = torch.tensor(np.array(y_test)).view(-1,sequence_length,batch_size,y_dim).to(torch.float32).cuda()
#     print(x_test.size())
    y_test.size()


    # In[ ]:


    from math import radians, cos, sin, asin, sqrt

    def haversine(lon1, lat1, lon2, lat2): # 经度1，纬度1，经度2，纬度2 （十进制度数）
        """
        Calculate the great circle distance between two points 
        on the earth (specified in decimal degrees)
        """
        # 将十进制度数转化为弧度
        lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

        # haversine公式
        dlon = lon2 - lon1 
        dlat = lat2 - lat1 
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * asin(sqrt(a)) 
        r = 6371 # 地球平均半径，单位为公里
        return c * r * 1000 # 单位为m


    # In[ ]:

    #predict
#     print("start predict!")
    # torch.set_printoptions(precision=8)
    haversine_history = []
    for indexx in range(len(x_test)):
        with torch.no_grad():
            target = y_test[indexx][-1].cpu().numpy()
            out = model(x_test[indexx]).cpu().numpy()
            lon1, lat1 = Scaler.inverse_transform(target)[0][0],Scaler.inverse_transform(target)[0][1]
            lon2, lat2 = Scaler.inverse_transform(out[0])[0][1-1],Scaler.inverse_transform(out[0])[0][2-1]
            haversine_history.append(haversine(lon1, lat1, lon2, lat2 ))
    #         print(haversine(lon1, lat1, lon2, lat2 ))
    average_loss = sum(haversine_history)/len(haversine_history)
    print("测试集上的平均距离误差",average_loss)
    
    if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
    score = 1.0/average_loss
    if score > best_score:
        print('Saving..')
        state = {
            'model': model.state_dict(),
            'score': score,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        best_score = score
    
# In[ ]:
# 开始训练
print("开始训练\n")

model = Net()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
criterion = nn.MSELoss()
resume = True #是否在模型的基础上训练
if resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    model.load_state_dict(checkpoint['model'])
    best_score = checkpoint['score']
    start_epoch = checkpoint['epoch']
else:
    start_epoch = 0
    best_score = 0
optimizer = torch.optim.Adam(model.parameters(), 1e-2)
def train(epoch):
    print("epoch {}:   ".format(epoch),end=' ')
    model.train()
    for j in range(len(x_train_list)):
        x_train = x_train_list[j]
        y_train = y_train_list[j]
        for iter in range(len(x_train)):
            output = model(x_train[iter])
#             print(output.shape)
            loss = criterion(output, y_train[iter][-1].view(1,batch_size,2))
            model.zero_grad()
            loss.backward() 
            optimizer.step()
    print("loss: {} ".format(loss))
    
for epoch in range(start_epoch, start_epoch+2000):
    train(epoch)
    test(epoch)