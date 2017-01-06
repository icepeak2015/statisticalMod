import numpy as np
from copy import copy
import matplotlib.pyplot as plt
from collections import defaultdict

class HMM:
    def __init__(self):
        pass

    def simulate(self,nSteps):

        def drawFrom(probs):
            return np.where(np.random.multinomial(1,probs) == 1)[0][0]

        observations = np.zeros(nSteps)
        states = np.zeros(nSteps)
        states[0] = drawFrom(self.pi)
        observations[0] = drawFrom(self.B[states[0],:])
        for t in range(1,nSteps):
            states[t] = drawFrom(self.A[states[t-1],:])
            observations[t] = drawFrom(self.B[states[t],:])
        return observations,states


    def train(self,observations,criterion,graphics=False):
        nStates = self.A.shape[0]
        nSamples = len(observations)

        A = self.A             #转移矩阵
        B = self.B             #生成矩阵
        pi = copy(self.pi)
       
        done = False
        while not done:

            # alpha_t(i) = P(O_1 O_2 ... O_t, q_t = S_i | hmm)
            # Initialize alpha  初始化前向概率
            alpha = np.zeros((nStates,nSamples))
            c = np.zeros(nSamples) #scale factors
            alpha[:,0] = pi.T * self.B[:,observations[0]]
            c[0] = 1.0/np.sum(alpha[:,0])
            alpha[:,0] = c[0] * alpha[:,0]
            # Update alpha for each observation step
            for t in range(1,nSamples):
                alpha[:,t] = np.dot(alpha[:,t-1].T, self.A).T * self.B[:,observations[t]]
                c[t] = 1.0/np.sum(alpha[:,t])
                alpha[:,t] = c[t] * alpha[:,t]

            # beta_t(i) = P(O_t+1 O_t+2 ... O_T | q_t = S_i , hmm)
            # Initialize beta
            beta = np.zeros((nStates,nSamples))
            beta[:,nSamples-1] = 1
            beta[:,nSamples-1] = c[nSamples-1] * beta[:,nSamples-1]
            # Update beta backwards from end of sequence
            for t in range(len(observations)-1,0,-1):
                beta[:,t-1] = np.dot(self.A, (self.B[:,observations[t]] * beta[:,t]))
                beta[:,t-1] = c[t-1] * beta[:,t-1]

            xi = np.zeros((nStates,nStates,nSamples-1));
            for t in range(nSamples-1):
                denom = np.dot(np.dot(alpha[:,t].T, self.A) * self.B[:,observations[t+1]].T,
                               beta[:,t+1])
                for i in range(nStates):
                    numer = alpha[i,t] * self.A[i,:] * self.B[:,observations[t+1]].T * \
                            beta[:,t+1].T
                    xi[i,:,t] = numer / denom
  
            # gamma_t(i) = P(q_t = S_i | O, hmm)
            gamma = np.squeeze(np.sum(xi,axis=1))
            # Need final gamma element for new B
            prod =  (alpha[:,nSamples-1] * beta[:,nSamples-1]).reshape((-1,1))
            gamma = np.hstack((gamma,  prod / np.sum(prod)))                     #append one more to gamma!!!

            newpi = gamma[:,0]
            newA = np.sum(xi,2) / np.sum(gamma[:,:-1],axis=1).reshape((-1,1))
            newB = copy(B)

            numLevels = self.B.shape[1]         
            sumgamma = np.sum(gamma,axis=1)               #按照样本求和

            for lev in range(numLevels):
                mask = observations == lev      #选取observations中等于lev的状态
                newB[:,lev] = np.sum(gamma[:,mask],axis=1) / sumgamma

            if np.max(abs(pi - newpi)) < criterion and \
                   np.max(abs(A - newA)) < criterion and \
                   np.max(abs(B - newB)) < criterion:
                done = 1;
  
            A[:],B[:],pi[:] = newA,newB,newpi

        self.A[:] = newA
        self.B[:] = newB
        self.pi[:] = newpi
        self.gamma = gamma
        
        
    def viterbi(self, obs):
        hid_states = self.hid_states           #隐状态        
        obs_states = self.obs_states           #观测状态
        pi = self.pi
        
        hid_number, obs_number = self.B.shape
        
        #转移矩阵，字典类型
        outdict = defaultdict(list)
        for i in range(hid_number):
            indict = defaultdict(list)
            for j in range(hid_number):
                indict[hid_states[j]] = self.A[i][j]           
            outdict[hid_states[i]] = indict
        trans_p = outdict


        #生成矩阵，字典类型
        outdict = defaultdict(list)
        for i in range(hid_number):
            indict = defaultdict(list)
            for j in range(obs_number):
                indict[obs_states[j]] = self.B[i][j]          #观测状态的各个元素
            outdict[hid_states[i]] = indict
        emit_p = outdict
        
        # 路径概率表 V[时间][隐状态] = 概率
        V = [{}]
        path = {}
     
        # 初始化初始状态 (t == 0)
        # V的形式 [{key:value, key:value},{},..]
        for y in hid_states:
            V[0][y] = pi[y] * emit_p[y][obs[0]]
            path[y] = [y]

        # 对 t > 0 跑一遍维特比算法
        for t in range(1, len(obs)):
            V.append({})
            newpath = {}
     
            for y in hid_states:
                # 概率 隐状态 =    前状态是y0的概率 * y0转移到y的概率 * y表现为当前状态的概率
                # max(a,b)以a为比较对象
                (prob, state) = max([(V[t - 1][y0] * trans_p[y0][y] * emit_p[y][obs[t]], y0) for y0 in hid_states])
                # 记录最大概率
                V[t][y] = prob
                # 记录路径 = 上次最佳路径 + 本次状态(而不是最大概率对应的状态state)
                # 最大概率 prob 对应的的state 只是本次的局部最优
                newpath[y] = path[state] + [y]

            # 保存最新路径
            path = newpath

        #找出最后一天的最大概率及其对应的状态
        prob, state = max([(V[len(obs) - 1][y], y) for y in hid_states])
        return prob, path[state]
        


