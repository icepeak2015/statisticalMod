import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import HmmClass as hmm
import os


#对原始数据划分为9个观测类别
min = 2400       #区间的长度 10*60*4=2400
def DivideObservionData(fileName):   
    rawData = []
    with open(fileName) as f:       #从文件中提取时间信息
        for line in f.readlines():
            spans = line.strip().split('\t')
            # timeIndex.append(spans[0])
            rawData.append(float(spans[-1]))
       
    rawData = np.array(rawData)
    
    #计算阈值
    thresh = (rawData.mean()+rawData.std())/2
    
    ret_value = []
    #计算每个区间的均值和标准差
    for i in range(0, len(rawData)-min, min):
        scope_mean = rawData[i:i+min].mean()
        scope_std = rawData[i:i+min].std()
        
        if scope_std <= 0.1:           
            if scope_mean <= thresh:
                ret_value.append(0)
            elif scope_mean > thresh and scope_mean <= 2*thresh:
                ret_value.append(1)
            else:
                ret_value.append(2)               
        elif scope_std > 0.1 and scope_std <= 0.3:
            if scope_mean <= thresh:
                ret_value.append(3)
            elif scope_mean > thresh and scope_mean <= 2*thresh:
                ret_value.append(4)
            else:
                ret_value.append(5)
        else:
            if scope_mean <= thresh:
                ret_value.append(6)
            elif scope_mean > thresh and scope_mean <= 2*thresh:
                ret_value.append(7)
            else:
                ret_value.append(8)              
        
    return ret_value, rawData

 
#绘制曲线图
fig = plt.figure()
ax = fig.add_subplot(111)

train_data = []
files = os.listdir()
for file in files:
    if file.endswith('.log'):
        single_data, _ = DivideObservionData(file)
        train_data.extend(single_data)

# fileName = 'sleep_20170105_2245_ref.log'
# train_data, _ = DivideObservionData(fileName)
train_data = np.mat(train_data).flatten().A[0]
   

pi = np.random.rand(3)
pi = pi/np.sum(pi)
A = np.random.rand(3,3)
A = A/np.sum(A, 1)
B = np.random.rand(3,9)

newB = []
for row in B:
    row = row/np.sum(row)
    newB.append(list(row))
B = newB.copy()
B = np.mat(B).A          # .A是为了去除中间的逗号



# 调用HMM进行训练
# 初始化起始概率pi，转移矩阵A，生成矩阵B
hmmguess = hmm.HMM()
hmmguess.pi = pi
hmmguess.A = A
hmmguess.B = B
# hmmguess.pi = np.array([0.3, 0.6, 0.1])
# hmmguess.A = np.array([[0.5, 0.3, 0.2],
                       # [0.2, 0.6, 0.2],
                       # [0.3, 0.4, 0.3]])
# hmmguess.B = np.array([[0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                       # [0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                       # [0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]])

hmmguess.train(train_data,0.0001,graphics=True)    #训练给定的数据, train_data需要为纯数据，中间没有逗号


#判断预测的结果
states = ('0', '1', '2')
obs_states = [0, 1, 2, 3, 4, 5, 6, 7, 8]
hmmguess.hid_states = states
hmmguess.obs_states = obs_states


#用于分类的数据
observations, originData = DivideObservionData('sleep_20170102_2248_ref.log')   
ax.plot(originData, label='original')
prob, path = hmmguess.viterbi(observations)
print('path: ', path)


#显示hmm
aweak_data = np.mat([np.ones(min)*10 if v=='0' else np.zeros(min) for v in path])
aweak_data = aweak_data.flatten().A[0]
ax.plot(aweak_data, color='red', linewidth=2, label='my_deep')

light_data = np.mat([np.ones(min)*10 if v=='1' else np.zeros(min) for v in path])
light_data = light_data.flatten().A[0]
ax.plot(light_data, color='green', linewidth=2, label='my_light')

aweak_data = np.mat([np.ones(min)*10 if v=='2' else np.zeros(min) for v in path])
aweak_data = aweak_data.flatten().A[0]
ax.plot(aweak_data, color='cyan', linewidth=2, label='my_aweak')

plt.legend(frameon=False)
plt.show()
