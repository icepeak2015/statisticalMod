import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

#获取各种手环与传感器之间对应的数据点
#refTime: 手环各种睡眠状态的起始时间
#refFileName：处理后的数据文件名称
def CorTimeSpan(refTime, refFileName):
    timeIndex = []
    ret_magnitude = []
    with open(refFileName) as f:       #从文件中提取时间信息
        for line in f.readlines():
            spans = line.strip().split('\t')
            timeIndex.append(spans[0])
            ret_magnitude.append(float(spans[-1]))
    
    
    ret_index = []
    length = len(timeIndex)
    findIndex = 0
    for i in range(len(refTime)):
        for j in range(findIndex, length):          # 不断更新内循环的起始位置
            if refTime[i] in timeIndex[j]:      # 时间点是否包含于timeIndex
                # print(timeIndex[j][0], ' contains: ', refTime[i])
                ret_index.append(j)
                findIndex = j
                break
    ret_index.append(length)
    return ret_index, ret_magnitude
    

#对原始数据，根据峰值的交替次数判断
# 算法流程：
# step1：计算每个区间的均值和标准差
# step2：在区间内标准差 > 0.3, 均值 > thresh  则判定为aweak
# step3：标准差 > 0.1 判定为 light-sleep
# step4：均值 < 3*thresh  则判定为  deep-sleep
# step5: 否则 判定为light-sleep

# 扩展状态 1
# 计算绝对的状态 0 
# 处理状态 2
# 合并离散的状态 1

min = 2400       #区间的长度 10*60*4=2400
def StatisticalMethodMean(rawData):   
    #计算阈值
    thresh = (rawData.mean()+rawData.std())/2
    
    ret_value = []
    #计算每个区间的均值和标准差
    for i in range(0, len(rawData)-min, min):
        scoph_mean = rawData[i:i+min].mean()
        scoph_std = rawData[i:i+min].std()
        # print('scoph_mean: ',scoph_mean, '  scoph_std: ', scoph_std)
        if scoph_std >= 0.3 and scoph_mean > thresh:
            ret_value.append(2)
            # print('std > 0.3   scoph_mean: ',scoph_mean, '  scoph_std: ', scoph_std)
        elif scoph_std >= 0.1:
            ret_value.append(1)
            # print('std > 0.1   scoph_mean: ',scoph_mean, '  scoph_std: ', scoph_std)
        elif scoph_mean <= 3*thresh:
            ret_value.append(0)
            # print('mean < 3*thresh   scoph_mean: ',scoph_mean, '  scoph_std: ', scoph_std)
        else:
            ret_value.append(1)
            # print('else    scoph_mean: ',scoph_mean, '  scoph_std: ', scoph_std)
                
                
    #将状态 1 前后相邻的状态也置为 1
    # light_sleep = ret_value.copy()
    # gap_flag = True
    # light_count = 1
    # for i in range(1,len(light_sleep)-1):
        # if gap_flag and ret_value[i] == 1:
            # light_sleep[i-1:i+2] = [1,1,1]      #list[a:b] 不包含b， 低级错误
            # light_count += 1
            # gap_flag = False

        # if not gap_flag:     #跳过后面的一个点
            # light_count -= 1
            # if light_count <= 0:
                # light_count = 1
                # gap_flag = True
                  
    
    #对连续的 0 状态进行处理，确定其为绝对的deep-sleep状态
    mod_deep = ret_value.copy()
    deep_length = 100
    deep_count = 0
    flag = True
    for i in range(len(mod_deep)):      #确定所有的deep-sleep
        if flag:
            for j in range(deep_length):
                if i+j<len(mod_deep) and mod_deep[i+j] == 0:     #统计连续的 状态0的数目
                    deep_count += 1
                else:
                    break
            if deep_count >= 5:          #超过5个连续的状态
                flag = False               
                deep_std = rawData[i*min:(i+deep_count)*min].std()               
                if deep_std < 0.1:
                    mod_deep[i:i+deep_count] = np.zeros(deep_count)
                    # print('i: ', i, '  deep_std: ', deep_std, ' deep_count: ', deep_count, ' classify: ',0)
                else:
                    mod_deep[i:i+deep_count] = np.ones(deep_count)
                    # print('i: ', i, '  deep_std: ', deep_std, ' deep_count: ', deep_count, ' classify: ',1)
            else:
                deep_count = 0
    
        if not flag:         #跳过连续的0状态
            deep_count -= 1
            if deep_count == 0:
                flag = True
    
    #对状态2进行处理， 处理后的结果放到mod_aweak中
    mod_aweak = mod_deep.copy()
    tp_count = 0
    aweak_length = 5
    aweak_flag = True
    for i in range(len(mod_aweak)-aweak_length):
        if aweak_flag:
            tp_count = aweak_length
            aweak_count = mod_aweak[i:i+aweak_length].count(2)
            if aweak_count >= (int(aweak_length/2) + 1):            #超过aweak_length一半的状态为2 则该区间判定为2
                mod_aweak[i:i+aweak_length] = np.ones(aweak_length, int)*2
                aweak_flag = False
            else:                  #少于一半，则将该区间中的 2 都修改为状态 1
                for j in range(aweak_length):
                    if mod_aweak[i+j] == 2:
                        mod_aweak[i+j] = 1
                aweak_flag = False
        if not aweak_flag:
            tp_count -= 1
            if tp_count == 0:            #跳过aweak_length次的处理
                aweak_flag = True

                
      
    #对距离较近的 离散的light-sleep状态进行合并        
    mod_light = mod_aweak.copy()
    light_length = 100               #内部循环区间的大小
    light_flag = True
    deep_length = 4
    deep_start_point = 0            #记录符合连续 0 长度的起始点
    deep_count = 0
    for i in range(len(mod_light)):
        if light_flag and mod_light[i]==1:
            for j in range(1,light_length):      #在一个较大的区间内，搜索0和1的出现次数
                if i+j < len(mod_light):       #防止数组越界
                    if mod_light[i+j]==1:
                        deep_count += 1
                        continue
                    elif mod_light[i+j]==0 or mod_light[i+j]==2:
                        tp_count_0 = mod_light[i+j:i+j+deep_length].count(0)
                        tp_count_2 = mod_light[i+j:i+j+deep_length].count(2)
                        if tp_count_0 >= deep_length or tp_count_2 >= deep_length:       #连续的多个状态0或2，停止内循环
                            deep_count += deep_length
                            deep_start_point = i+j
                            light_flag = False       #修改flag ，跳过前面确定的点
                            break             #退出内部 for 循环
                        else:
                            deep_count += 1        #连续状态0的个数太少
                            continue
                else:
                    break
            #将从i到deep_start_point 之间的状态都置为 1
            if deep_start_point - i > 1:
                mod_light[i:deep_start_point] = np.ones(deep_start_point-i, int)   
                
        if not light_flag:
            deep_count -= 1
            if deep_count == 0:
                light_flag = True
        
    return ret_value, mod_light

    
 
#贯众手环睡眠分类结果
# 12.26的数据
# refTime = ['22:10:00', '23:18:00', '23:32:00', '23:48:00', '00:28:00', '01:28:00', '01:46:00', '02:08:00', 
# '02:28:00', '03:48:00', '04:06:00', '05:14:00', '05:36:00', '05:52:00', '06:02:00', '06:04:00', 
# '06:14:00', '06:16:00', '06:32:00'] 

# 12.29的数据
# refTime = ['23:24:00', '23:36:00', '00:10:00', '00:34:00', '01:14:00', '02:46:00', '03:36:00', 
# '04:28:00', '04:42:00', '05:00:00', '05:10:00']

# 12.30的数据
# refTime = ['00:46:00', '00:58:00', '01:10:00', '02:00:00', '02:16:00', '02:18:00', '03:08:00',
# '03:38:00', '03:48:00', '03:50:00', '04:50:00', '05:14:00', '05:24:00', '06:00:00', '06:14:00', '06:16:00',
# '06:34:00', '07:19:00']

# 01.02的数据
# refTime = ['22:54:00', '23:06:00', '23:40:00', '00:26:00', '01:44:00', '01:46:00', '02:00:00', '02:04:00', 
# '02:24:00', '02:26:00', '02:52:00', '03:10:00', '03:20:00', '04:02:00', '04:16:00', '04:24:00', '04:44:00', 
# '05:08:00', '05:28:00']

# 01.03的数据
refTime = ['22:33:00', '22:43:00', '23:26:00', '00:48:00', '01:04:00', '01:26:00', '01:36:00', '01:59:00', '02:24:00', '02:50:00', '03:10:00', 
'03:34:00', '03:48:00', '04:00:00', '04:19:00', '04:30:00', '05:00:00', '05:58:00', '06:10:00', '06:21:00', '06:33:00', '06:34:00']

# 01.04的数据  贯众手环
# refTime = ['23:04:00', '23:42:00', '00:26:00', '00:28:00', '00:42:00', '00:44:00', '02:00:00', '02:58:00', '03:08:00', 
# '03:10:00', '03:20:00', '03:22:00', '03:34:00', '03:36:00', '04:00:00', '05:02:00', '05:24:00', '06:22:00', '06:34:00', '07:02:00'] 

# 01.04的数据  bong手环
# refTime = ['23:19:00', '23:48:00', '00:25:00', '00:54:00', '01:16:00', '01:29:00', '02:08:00', '02:45:00', 
# '03:07:00', '03:20:00', '04:01:00', '04:31:00', '04:35:00', '04:47:00', '05:39:00', '06:27:00'] 

# 01.04的数据  小米手环
# refTime = ['23:06:00', '23:19:00', '23:29:00', '23:40:00', '00:20:00', '00:49:00', '01:11:00', '02:33:00', '02:45:00', '03:35:00', 
# '03:57:00', '05:08:00', '05:30:00', '06:15:00', '06:32:00', '06:49:00', '06:59:00', '07:15:00']
  

#绘制曲线图
fig = plt.figure()
ax = fig.add_subplot(111)
fileName = 'sleep_20170103_2224_ref.log'

time, rawData = CorTimeSpan(refTime, fileName)

ori_value, statis_value = StatisticalMethodMean(np.array(rawData))
ax.plot(rawData)

# print('total deep-sleep minites: ', statis_value.count(0)*4)
# print('total light-sleep minites: ', statis_value.count(1)*4)


#原始分类结果
# aweak_data = np.mat([np.ones(min)*8 if v==2 else np.zeros(min) for v in ori_value])
# aweak_data = aweak_data.flatten().A[0]
# ax.plot(aweak_data, color='cyan', linewidth=2, label='aweak epoch')

# light_data = np.mat([np.ones(min)*8 if v==1 else np.zeros(min) for v in ori_value])
# light_data = light_data.flatten().A[0]
# ax.plot(light_data, color='magenta', linewidth=2, label='light-sleep epoch')


#合并后结果
aweak_data = np.mat([np.ones(min)*10 if v==0 else np.zeros(min) for v in statis_value])
aweak_data = aweak_data.flatten().A[0]
ax.plot(aweak_data, color='red', linewidth=2, label='my_deep')

light_data = np.mat([np.ones(min)*10 if v==1 else np.zeros(min) for v in statis_value])
light_data = light_data.flatten().A[0]
ax.plot(light_data, color='green', linewidth=2, label='my_light')


aweak_data = np.mat([np.ones(min)*10 if v==2 else np.zeros(min) for v in statis_value])
aweak_data = aweak_data.flatten().A[0]
ax.plot(aweak_data, color='black', linewidth=2, label='my_aweak')


guanzhong_light = []
guanzhong_deep = []
guanzhong_light.extend(np.zeros(time[0]))
guanzhong_deep.extend(np.zeros(time[0]))
for i in range(len(time)-1):
    if i%2 == 0:
        guanzhong_light.extend(np.ones(time[i+1]-time[i], int)*15)
        guanzhong_deep.extend(np.zeros(time[i+1]-time[i]))
    else:
        guanzhong_deep.extend(np.ones(time[i+1]-time[i], int)*15)
        guanzhong_light.extend(np.zeros(time[i+1]-time[i]))
        
ax.plot(guanzhong_light, color='green', linewidth=3, label='ref_light')
ax.plot(guanzhong_deep, color='red', linewidth=3, label='ref_deep')


plt.legend(frameon=False)
plt.show()
