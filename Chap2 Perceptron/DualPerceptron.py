'''
Author : Superpig99
Date : 2021/12/05
'''
import numpy as np

class DaulPerceptron:
    def __init__(self,learning_rate,max_epoch):
        self.lr = learning_rate # 学习率
        self.me = max_epoch # 最大的训练次数
    # 给定X，预测y
    def predict(self,X):
        m = X.shape[0]
        y = self.Gram @ self.c + self.b
        y = np.where(y>=0,1,-1)
        return y

    def fit(self,X,y): # X是m*n的矩阵，y为m*1的向量，m为样本个数，n为特征个数
        m,n = X.shape[0],X.shape[1]
        self.a = np.zeros((m,1)) # 参数a是m*1的向量
        self.b = np.zeros(1)
        self.Gram = [[0]*m for _ in range(m)]
        for i in range(m):
            self.Gram[i][i] = X[i] @ X[i].T
            for j in range(i+1,m):
                self.Gram[i][j] = X[i] @ X[j].T
                self.Gram[j][i] = X[i] @ X[j].T
        print("Gram:",self.Gram)
        for i in range(self.me): # 开始训练
            self.c = self.a * y
            yhat = self.predict(X)
            wrong_index = np.where((y - yhat)!=0,1,0) # 指示矩阵，指示哪些地方预测错了
            self.a = self.a + self.lr*wrong_index # 修正a，a = a + lr
            self.b = self.b + self.lr*np.sum(wrong_index*y) # 修正b，b = b + lr * y
            # print('epoch:',i)
            # print(self.a.T,'\n',wrong_index.T)
            print('Epoch: %d, Wrong points: %d, Error Rate: %.2f'%(i,np.sum(wrong_index),np.sum(wrong_index)/m))
            if np.sum(wrong_index)==0: # 如果全部预测正确，则训练结束
                break
        return
    
    def evaluation(self,Yhat,Ytrue):
        if Yhat.shape == Ytrue.shape:
            acu = np.sum(np.where((Yhat - Ytrue)==0,1,0))/Ytrue.shape[0]
            return acu
        else:
            print('the shape of Yhat and Ytrue is different')
            

if __name__=='__main__':
    X = np.array([[3,3],[4,3],[1,1]])
    y = np.array([[1],[1],[-1]])
    per = DaulPerceptron(learning_rate=1,max_epoch=20)
    per.fit(X,y)
    yhat = per.predict(X)
    acu = per.evaluation(yhat,y)
    print('Accuarcy is %.2f'%acu)