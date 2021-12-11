'''
author : superpig99
date : 2021/12/11
'''
import pandas as pd
class naiveBayes:
    def __init__(self):
        pass
    def fit(self,data,label='y'): ## 学习联合分布
        # 还没实现异常处理
        # if not isinstance(data, pd.DataFrame):
        #     print('please input DataFrame')
        #     return
        # if not label in data.columns:
        #     print('please set name of label')
        #     return
        y_tmp = data.groupby(label).count() # 聚合，得到每个类别的频数
        self.prob_y = (y_tmp.iloc[:,0]).to_dict() ## 将dataframe转换为dict
        self.prob_x = {} # 初始化特征的频数
        # 对每一维特征xi
        for xi in data.columns:
            if xi!=label: # 排除y
                xi_tmp = data.groupby([label,xi]).count() # 按照y和xi进行聚类求和，得到对应频数
                val = xi_tmp.iloc[:,0].to_dict() ## 将dataframe转换为dict，该字典为该维的频数
                self.prob_x[xi] = val # 将该字典加入到prob_x中
        print(self.prob_x)
        return

    def predict(self,X): ## 预测
        n,res = len(X.columns),[] # n为特征维数，res为返回结果
        for x in X.values: # 对每个样本
            y_prob, cls = 0,None # 初始化该样本对应的后验概率和输出类别
            for c in self.prob_y.keys(): # 对每个类别，计算后验概率
                prob = 1 # 初始化后验概率为1
                for i in range(n): # 对每一维特征，开始连乘
                    prob *= self.prob_x[X.columns[i]][(c,x[i])]/self.prob_y[c] # 频数相除，得到频率，即条件概率P(X=xi|Y=c)
                tmp = prob*self.prob_y[c]
                if tmp>y_prob: # 取最值
                    y_prob = tmp
                    cls = c
            res.append(cls)
        return res

if __name__ == '__main__':
    ## 测试样例来自《统计学习方法》P63
    trainSet = pd.DataFrame({'x1':[1,1,1,1,1,2,2,2,2,2,3,3,3,3,3],\
                             'x2':['S','M','M','S','S','S','M','M','L','L','L','M','M','L','L'],\
                             'y':[-1,-1,1,1,-1,-1,-1,1,1,1,1,1,1,1,-1]})
    testSet = pd.DataFrame({'x1':[2],'x2':['S'],'y':[-1]})
    mod = naiveBayes()
    mod.fit(trainSet)
    print('res: ', mod.predict(testSet.iloc[:,:-1]))
