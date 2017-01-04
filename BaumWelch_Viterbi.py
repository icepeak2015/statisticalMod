# -*- coding:utf-8 -*-
import numpy as np
# functions and classes go here

# 前向-后向算法计算概率
# A_mat: 转换矩阵
# O_mat: 观测矩阵(生成矩阵)
# observer：观测序列
def fb_alg(A_mat, O_mat, observ):
    # set up
    k = observ.size 
    (n,m) = O_mat.shape
    prob_mat = np.zeros( (n,k) )    #概率矩阵
    fw = np.zeros( (n,k+1) )        #前向概率
    bw = np.zeros( (n,k+1) )
    # forward part    计算前向概率
    # fw[:, 0] = 1.0/n              #初始化前向概率
    
    # 随机初始化前向概率
    init_fw = np.random.rand(n)
    init_fw = init_fw/np.sum(init_fw)   
    fw[:, 0] = init_fw.copy()

    
    for obs_ind in range(k):
        f_row_vec = np.matrix(fw[:,obs_ind])
        fw[:, obs_ind+1] = f_row_vec * \
                           np.matrix(A_mat) * \
                           np.matrix(np.diag(O_mat[:,observ[obs_ind]]))  #将向量变成对角矩阵
        fw[:,obs_ind+1] = fw[:,obs_ind+1]/np.sum(fw[:,obs_ind+1])      #向量归一化
    
    # backward part    计算后向概率
    bw[:,-1] = 1.0           #设置bw[:, k] = 1.0
    
    # 随机初始化后向概率
    init_bw = np.random.rand(n)
    init_bw = init_bw/np.sum(init_fw)   
    bw[:,-1] = init_bw.copy()
    
    
    for obs_ind in range(k, 0, -1):      # [K, 0)
        b_col_vec = np.matrix(bw[:,obs_ind]).transpose()
        bw[:, obs_ind-1] = (np.matrix(A_mat) * \
                            np.matrix(np.diag(O_mat[:,observ[obs_ind-1]])) * \
                            b_col_vec).transpose()
        bw[:,obs_ind-1] = bw[:,obs_ind-1]/np.sum(bw[:,obs_ind-1])
    
    # combine it
    prob_mat = np.array(fw)*np.array(bw)
    prob_mat = prob_mat/np.sum(prob_mat, 0)
    # get out
    return prob_mat, fw, bw

# num_states: 隐状态数目
# num_obs: 观察值的数目
# observ: 观察值的序列
def baum_welch( num_states, num_obs, observ ):
    # allocate
    A_mat = np.ones( (num_states, num_states) )
    A_mat = A_mat / np.sum(A_mat,1)
    # print(A_mat)
    # print('sum: ', A_mat.sum())
    O_mat = np.ones( (num_states, num_obs) )
    O_mat = O_mat / np.sum(O_mat,1)
    # print(O_mat)
    # print('sum: ', O_mat.sum())
    theta = np.zeros( (num_states, num_states, observ.size) )
    while True:
        old_A = A_mat
        old_O = O_mat
        A_mat = np.ones( (num_states, num_states) )
        O_mat = np.ones( (num_states, num_obs) )
        # expectation step, forward and backward probs
        P,F,B = fb_alg( old_A, old_O, observ)
        # need to get transitional probabilities at each time step too
        for a_ind in range(num_states):
            for b_ind in range(num_states):
                for t_ind in range(observ.size):
                    theta[a_ind,b_ind,t_ind] = \
                    F[a_ind,t_ind] * \
                    B[b_ind,t_ind+1] * \
                    old_A[a_ind,b_ind] * \
                    old_O[b_ind, observ[t_ind]]
        # update A_mat
        for a_ind in range(num_states):
            for b_ind in range(num_states):
                A_mat[a_ind, b_ind] = np.sum( theta[a_ind, b_ind, :] )/ \
                                      np.sum(P[a_ind,:])
        A_mat = A_mat / np.sum(A_mat,1)
        # print(A_mat)
        # print('sum: ', A_mat.sum())

        # update O_mat
        for a_ind in range(num_states):
            for o_ind in range(num_obs):
                right_obs_ind = np.array(np.where(observ == o_ind))+1
                O_mat[a_ind, o_ind] = np.sum(P[a_ind,right_obs_ind])/ \
                                      np.sum( P[a_ind,1:])
        O_mat = O_mat / np.sum(O_mat,1)

        print('A_mat: ', A_mat)
        print('O_mat: ', O_mat)

        # compare
        if np.linalg.norm(old_A-A_mat) < .00001 and np.linalg.norm(old_O-O_mat) < .00001:
            break
    # get out
    return A_mat, O_mat

#测试Demo
num_obs = 10000
observations1 = np.random.randn( num_obs )    #范围无限制的标准正态分布
observations1[observations1>0] = 1
observations1[observations1<=0] = 0
A_mat, O_mat = baum_welch(2,2,observations1)
# print('A_mat: ', A_mat)
# print('O_mat: ', O_mat)


#Viterbi 算法解决HMM的问题之一：
#已知模型参数和观测数据，估计最优的音状态序列
 
states = ('Rainy', 'Sunny')
observations = ('walk', 'shop', 'clean')
start_probability = {'Rainy': 0.6, 'Sunny': 0.4}
 
transition_probability = {
    'Rainy' : {'Rainy': 0.7, 'Sunny': 0.3},
    'Sunny' : {'Rainy': 0.4, 'Sunny': 0.6},
    }
 
emission_probability = {
    'Rainy' : {'walk': 0.1, 'shop': 0.4, 'clean': 0.5},
    'Sunny' : {'walk': 0.6, 'shop': 0.3, 'clean': 0.1},
}
 
# 打印路径概率表
def print_dptable(V):
    s = "    " + " ".join(("%7d" % i) for i in range(len(V))) + "\n"
    for y in V[0]:      #显示每个状态对应观察值的概率
        s += "%.5s: " % y
        s += " ".join("%.7s" % ("%f" % v[y]) for v in V)
        s += "\n"
    print(s)
 
def viterbi(obs, states, start_p, trans_p, emit_p):
    """
    :param obs:观测序列
    :param states:隐状态
    :param start_p:初始概率（隐状态）
    :param trans_p:转移概率（隐状态）
    :param emit_p: 发射概率 （隐状态表现为显状态的概率）
    :return:
    """
    # 路径概率表 V[时间][隐状态] = 概率
    V = [{}]
    # 一个中间变量，代表当前状态是哪个隐状态
    path = {}
 
    # 初始化初始状态 (t == 0)
    # V的形式 [{key:value, key:value},{},..]
    for y in states:
        # print('y: ', y)
        # print(emit_p[y])
        V[0][y] = start_p[y] * emit_p[y][obs[0]]
        path[y] = [y]

    # 对 t > 0 跑一遍维特比算法
    for t in range(1, len(obs)):
        V.append({})
        newpath = {}
 
        for y in states:
            # 概率 隐状态 =    前状态是y0的概率 * y0转移到y的概率 * y表现为当前状态的概率
            # max(a,b)以a为比较对象
            (prob, state) = max([(V[t - 1][y0] * trans_p[y0][y] * emit_p[y][obs[t]], y0) for y0 in states])
            # 记录最大概率
            V[t][y] = prob
            # 记录路径 = 上次最佳路径 + 本次状态(而不是最大概率对应的状态state)
            # 最大概率 prob 对应的的state 只是本次的局部最优
            newpath[y] = path[state] + [y]


        # 保存最新路径
        path = newpath
        # print(V)
    print_dptable(V)

    #找出最后一天的最大概率及其对应的状态
    prob, state = max([(V[len(obs) - 1][y], y) for y in states])
    # print(prob, path[state])
    return (prob, path[state])
 
#测试Viterbi算法
# viterbi(observations,
         # states,
         # start_probability,
         # transition_probability,
         # emission_probability)

 