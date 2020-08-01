import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

#------------data preprocessing part----------------

#read csv
data=pd.read_csv("./train.csv", encoding = 'big5')

# Extract data , replace 'NR' with 0
data=data.iloc[:,3:]
data=data.replace('NR',0.0)

raw_data=data.to_numpy()
#raw_data.shape=(4320, 24)

#將原始 4320 * 18 的資料依照每個月分重組成 12 個 18 (features) * 480 (hours) 的資料。
month_data={}
for month in range(12):
    # 18个feature list, 包含480小时的数据
    sample = np.empty([18,480])
    for day in range(20):
        sample[:,day*24:(day+1)*24]=raw_data[18*day+month*20*18:18*(day+1)+month*20*18]
    month_data[month] = sample

#month_data[0].shape = (18,480)

#每個月會有 480hrs，每 9 小時形成一個 data，每個月會有 471 個 data，故總資料數為 471 * 12 筆，而每筆 data 有 9 * 18 的 features (一小時 18 個 features * 9 小時)。
#對應的 target 則有 471 * 12 個(第 10 個小時的 PM2.5)
data= np.empty([12*471,18*9],dtype=float)
data_3d= np.empty([12*471,18,9],dtype=float)
target=np.empty([12*471,1],dtype = float)


for month in range(12):
        for time in range(471):
            data[month*471+time]=month_data[month][:,time:time+9].reshape(1,-1)
            target[month*471+time]=month_data[month][9,time+9]

#data (5652, 18, 9)
#target (5652, 1)
# print(data[0])

#------------Normalize part----------------
mean_x = np.mean(data, axis = 0) #18 * 9 
std_x = np.std(data, axis = 0) #18 * 9 
# print(data.shape[0])
# print(data.shape[1])
for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        if std_x[j]!=0:
            data[i][j]= (data[i][j]-mean_x[j])/std_x[j]

#分開90%的trainning 和 10%的 validation set
x_train = data[: math.floor(len(data) * 0.9), :]
y_train = target[: math.floor(len(target) * 0.9), :]
x_val = data[math.floor(len(data) * 0.9): , :]
y_val = target[math.floor(len(target) * 0.9): , :]

#tensor preparation
x= torch.from_numpy(x_train).float()
y=torch.from_numpy(y_train).float()
x_v= torch.from_numpy(x_val).float()
y_v=torch.from_numpy(y_val).float()

#Trainning- DIY way - Adagrad
x_t=x_train
y_t=y_train
dim = 18 * 9 + 1
w = np.zeros([dim, 1])
x_t = np.concatenate((np.ones([len(x_t), 1]), x_t), axis = 1).astype(float)
learning_rate = 1
iter_time = 5000
adagrad = np.zeros([dim, 1])
eps = 0.0000000001
ada_loss=[]
for t in range(iter_time):
    loss = np.sum(np.power(np.dot(x_t, w) - y_t, 2)/len(x_t))#mse
    ada_loss.append(loss)
    if((t+1)%1000==0):
        print(str(t+1) + ":" + str(loss))
    gradient = 2 * np.dot(x_t.transpose(), np.dot(x_t, w) - y_t) #dim*1
    adagrad += gradient ** 2
    w = w - learning_rate * gradient / np.sqrt(adagrad + eps)
x_val = np.concatenate((np.ones([len(x_val), 1]), x_val), axis = 1).astype(float)
predict_y=np.dot(x_val,w)
diy_loss = np.sum(np.power(predict_y - y_val, 2))/len(x_val)


#Trainning- modern way - Adam
adam_iter=7000
linear_module = nn.Linear(18*9, 1)
optim = torch.optim.Adam(linear_module.parameters(), lr=0.01)
loss_func = nn.MSELoss()
print('iter,\tloss')
adam_loss=[]
for i in range(adam_iter):
    y_hat = linear_module(x)
    loss = loss_func(y_hat, y)
    adam_loss.append(loss.item())
    optim.zero_grad()
    loss.backward()
    optim.step()
    if ((i+1)%1000==0):
        print('{},\t{:.2f}'.format(i+1, loss.item()))



#Model Validation 

print(x_v.shape)
y_hat = linear_module(x_v)
loss = loss_func(y_hat, y_v)
print("\nAdagrad Loss: ",diy_loss)
print("Adam Loss: ",loss.item())

#Plot Loss function & Predict vs Validation 
f1=plt.figure(1)
plt.plot(np.arange(1,adam_iter+1,1), adam_loss,5)
plt.title('Adam Loss, lr=0.01, iter=7000')
plt.xlabel('$iter$')
plt.ylabel('$loss$')

f2=plt.figure(2)
plt.plot(np.arange(1,len(y_val)+1,1), np.absolute(y_hat.detach().numpy()-y_val),5)
# plt.plot(np.arange(1,len(y_val)+1,1), y_val,5, label="validation data")
plt.title('Adam Predict absolute error')
plt.xlabel('$id$')
plt.ylabel('$PM2.5 predict error$')

f3=plt.figure(3)
plt.plot(np.arange(1,iter_time+1-6,1), ada_loss[6:],5)
plt.title('Adagrad Predict, lr=1, iter=5000')
plt.xlabel('$iter$')
plt.ylabel('$loss$')


f4=plt.figure(4)
plt.plot(np.arange(1,len(y_val)+1,1), np.absolute(predict_y-y_val),5)
# plt.plot(np.arange(1,len(y_val)+1,1), y_val,5, label="validation data")
plt.title('Adagrad Predict absolute error')
plt.xlabel('$id$')
plt.ylabel('$PM2.5 predict error$')

plt.show()