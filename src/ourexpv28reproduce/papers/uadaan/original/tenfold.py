import pandas as pd
import random
import csv
import numpy as np

def cross_10folds(data,folds,jiange,start_index,end_index):
    df_test=data[start_index*jiange:end_index*jiange]
    df_test_index=list(df_test.index)
    df_test_flag=(data.index).isin(df_test_index)
    diff_flag = [not f for f in df_test_flag] 
    df_train= data[diff_flag]
    return df_train,df_test

def shuf(data, path, n):
    r = []
    for j in range (n):
        tmp=[]
        # 每个被试的8条作为一组打乱
        for i in range(j*8,j*8+8):
            tmp.append(np.asarray(data.iloc[i]))
        r.append(pd.Series(tmp))
    random.shuffle(r)
    with open(path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for row in r:
            writer.writerows(row)
    
path_c='./c_sub.csv'
path_d='./d_sub.csv'

data_c=pd.read_csv(path_c,header=None,index_col=None)
data_d=pd.read_csv(path_d,header=None,index_col=None)

folds = 10 # 十折
sub_c = 29 # 对照组
sub_d = 24 # 抑郁
# 按被试独立 每个被试的8条作为一组
jiange_c=int(sub_c/folds)*8
jiange_d=int(sub_d/folds)*8

for j in range(1,2):
    # 分别按被试打乱
    shuf(data_c, './shuffle_c.csv', 29)
    shuf(data_d, './shuffle_d.csv', 24)
    data_cc=pd.read_csv('./shuffle_c.csv',header=None,index_col=None)
    data_dd=pd.read_csv('./shuffle_d.csv',header=None,index_col=None)
    root = './'+str(j)
    for i in range(1,folds+1):
        # 训练和测试集中的对照组
        df_train_c, df_test_c=cross_10folds(data_cc,folds,jiange_c,i-1,i)
        # 训练和测试集中的抑郁
        df_train_d, df_test_d=cross_10folds(data_dd,folds,jiange_d,i-1,i)
        # 合并得到训练集
        df_train = pd.concat([df_train_c,df_train_d])
        df_train.to_csv(root+'/train'+str(i)+'.csv',index=False,header=None)
        # 合并得到测试集
        df_test = pd.concat([df_test_c,df_test_d])
        df_test.to_csv(root+'/test'+str(i)+'.csv',index=False,header=None)
    
