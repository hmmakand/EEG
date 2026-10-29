import torch
import torch.nn as nn
import discriminator
import torch.nn.functional as F
import extractor
import classifier
import bayesian as bnn
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.feature_extractor = extractor.Extractor() #特征提取器
        self.label_clf = classifier.Classifier() #标签分类器
        self.ce_loss = nn.CrossEntropyLoss(reduction='mean') #交叉熵损失
        self.adapt_loss = discriminator.DAANloss() #域对抗损失
        self.kl_loss = bnn.BKLLoss() #KL散度

    def forward(self, source, target, source_label):
        c = []
        f = []
        # 多次采样取平均
        for _ in range(10):
            # 源域特征
            feature = self.feature_extractor(source)
            # 源域分类结果
            clf = self.label_clf(feature)
            f.append(feature)
            c.append(clf)
        source_f = sum(f)/len(f)
        source_clf = sum(c)/len(c) + 1e-10
        kl = self.kl_loss(self.label_clf) # 先验kl
        ce = self.ce_loss(source_clf, source_label) #交叉熵
        clf_loss = ce * 2 + kl * 0.01 # 标签分类损失

        c1 = []
        f1 = []
        # 多次采样取平均
        for _ in range(10):
            # 目标域特征
            feature = self.feature_extractor(target)
            # 目标域分类结果
            clf = self.label_clf(feature)
            f1.append(feature)
            c1.append(clf)
        target_f = sum(f1)/len(f1)
        target_clf = sum(c1)/len(c1) + 1e-10
       
        # 域对抗损失
        transfer_loss = self.adapt_loss(source_f, target_f, source_clf, target_clf)

        return clf_loss, transfer_loss
    
    def predict(self, data):
        # 多次预测取平均
        r = []
        for _ in range(10):
            features = self.feature_extractor(data)
            clf = self.label_clf(features)
            r.append(clf)
        result = sum(r)/len(r)
        return result
    
    def pre_single(self, data):
        # 预测单个数据并画分布图
        d = []
        # 多次预测取平均
        for _ in  range(10):
            features = self.feature_extractor(data)
            clf = self.label_clf(features)
            # 抑郁的概率分布
            d.append(clf.squeeze(0)[1].cpu().numpy())
        print('predict mean: ' + str(np.mean(d)))
        print('predict std: ' + str(np.std(d)))
        sns.kdeplot(np.array(d),fill=True)
        plt.show()
    
    def get_feature(self, data):
        # 获得特征提取器提出来的深层特征
        return self.feature_extractor(data)
        
    def epoch_based_processing(self,n):
        # 更新动态对抗因子
        self.adapt_loss.update_dynamic(n)