import dataloader
import torch
import model
import numpy as np
import torch.optim as optim
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import pandas as pd
import umap

def loaddata(j,i):
    # 加载训练和测试数据
    train = dataloader.DatasetFromCSV('./'+str(j)+'/train'+str(i)+'.csv')
    train_loader = torch.utils.data.DataLoader(train, batch_size=98, shuffle=True,  num_workers=0)
    test = dataloader.DatasetFromCSV('./'+str(j)+'/test'+str(i)+'.csv')
    test_loader = torch.utils.data.DataLoader(test, batch_size=8, shuffle=True,  num_workers=0)
    return train_loader, test_loader

def loadtestdata(j,i):
    # 加载测试数据
    test = pd.read_csv('./'+str(j)+'/test'+str(i)+'.csv',header=None)
    X_test  = test.iloc[:, 2:642].values
    y_test = test.iloc[:, 1].values
    return X_test, y_test

def enable_dropout(model):
    # 在测试阶段打开dropout（MC Dropout）
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()

def test2(model, j,i):
    # 测试并得到指标
    model.eval()
    enable_dropout(model)
    X_test, y_test = loadtestdata(j,i)
    with torch.no_grad():
        data = torch.tensor(X_test).to(torch.float32).to("cuda")
        y_test = y_test.astype(np.int32)
        pre = model.predict(data)
        _, predicted = torch.max(pre.data, 1)
        predicted = predicted.cpu().numpy()
        # print("---------------------predict all test data-----------------------")
        # print("true labels: ")
        # print(y_test)
        # print("predict labels: ")
        # print(predicted)
        acc = accuracy_score(y_test, predicted)
        recall = recall_score(y_test, predicted)
        specificity = recall_score(y_test, predicted, pos_label=0)
        precision = precision_score(y_test, predicted)
        f1 = f1_score(y_test, predicted)
    return acc*100, recall*100, specificity*100, precision*100, f1*100

def pred_single(model, j, i, k):
    # 测试单个数据
    model.eval()
    enable_dropout(model)
    X_test, y_test = loadtestdata(j,i)
    with torch.no_grad():
        data = torch.tensor(X_test).to(torch.float32).to("cuda")
        print("--------------------predict single test data---------------------")
        print('true label: ' + str(y_test[k]))
        single = data[k].unsqueeze(0)
        model.pre_single(single)

def draw_feature(model, source_loader, target_loader):
    # 画经过模型前后特征分布图
    model.eval()
    enable_dropout(model)
    f = 0
    f1 = 0
    color = []
    # 源域：红色
    for _ in range(392):
        color.append('red')
    # 目标域：蓝色
    for _ in range(32):
        color.append('blue')
    with torch.no_grad():
        for data, label, domain in source_loader:
            data, label, domain = data.to(torch.float32).to("cuda"), label, domain
            fea = model.get_feature(data)
            data = data.cpu().numpy()
            fea = fea.cpu().numpy()
            if f == 0:
                c = data
                c1 = fea
                color2 = domain
                color3 = label
                f = 1
            else:
                c = np.concatenate([c, data], axis=0)
                c1 = np.concatenate([c1, fea], axis=0)
                color2 = np.concatenate([color2, domain], axis=0)
                color3 = np.concatenate([color3, label], axis=0)

        for data, label, domain in target_loader:
            data, label, domain = data.to(torch.float32).to("cuda"), label , domain
            fea = model.get_feature(data)
            data = data.cpu().numpy()
            fea = fea.cpu().numpy()
            if f1 == 0:
                c2 = data
                c3 = fea
                color4 = domain
                color5 = label
                f1 = 1
            else:
                c2 = np.concatenate([c2, data], axis=0)
                c3 = np.concatenate([c3, fea], axis=0)
                color4 = np.concatenate([color4, domain], axis=0)
                color5 = np.concatenate([color5, label], axis=0)
            c = np.concatenate([c, data], axis=0)
            c1 = np.concatenate([c1, fea], axis=0)
            color2 = np.concatenate([color2, domain], axis=0)
            color3 = np.concatenate([color3, label], axis=0)
        embedding = umap.UMAP(n_neighbors=10,min_dist=0.1,random_state=42).fit_transform(c)
        embedding1 = umap.UMAP(n_neighbors=10,min_dist=0.1,random_state=42).fit_transform(c1)
        #tsne = TSNE(n_components=2, perplexity=30, learning_rate="auto", init="random")
       # print("c:",c)
        tsne_result1 = embedding#tsne.fit_transform(c)
        tsne_result2 = embedding1#tsne.fit_transform(c1)

        colors = np.random.rand(53, 3)
        map1 = plt.cm.colors.ListedColormap(colors)
        s = 15
        
       
        plt.figure(figsize=(8,8))
        plt.subplot(2, 3, 1)
        plt.scatter(tsne_result1[:, 0], tsne_result1[:, 1], c=color,cmap=plt.cm.Spectral, s = s)
       
        plt.subplot(2, 3, 2)
        plt.scatter(tsne_result1[:, 0], tsne_result1[:, 1], c=color2,cmap=plt.cm.Spectral,  s = s)

        plt.subplot(2, 3, 3)
        plt.scatter(tsne_result1[:, 0], tsne_result1[:, 1], c=color3,cmap=plt.cm.Spectral,  s = s)
    
        plt.subplot(2, 3, 4)
        plt.scatter(tsne_result2[:, 0], tsne_result2[:, 1], c=color,cmap=plt.cm.Spectral, s = s)
    
        plt.subplot(2, 3, 5)
        plt.scatter(tsne_result2[:, 0], tsne_result2[:, 1], c=color2,cmap=plt.cm.Spectral, s = s)
 
        plt.subplot(2, 3, 6)
        plt.scatter(tsne_result2[:, 0], tsne_result2[:, 1], c=color3,cmap=plt.cm.Spectral, s = s)
        plt.show()

def train(model, j, i, epoch=20):
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-3)

    for e in range(1, epoch+1):
        source_loader, target_loader= loaddata(j, i)
        batch_source = len(source_loader)
        batch_target = len(target_loader)
        n_batch = min(batch_source,batch_target)
        iter_source = iter(source_loader)
        iter_target = iter(target_loader)

        model.train()
        # 更新动态对抗因子
        model.epoch_based_processing(n_batch)

        for _ in range(n_batch):
            data_source, label_source, _ = next(iter_source)
            data_target, _ , _= next(iter_target)

            data_source, label_source = data_source.to(torch.float32).to("cuda"), label_source.to(torch.long).to("cuda")
            data_target = data_target.to(torch.float32).to("cuda")
        
            clf_loss, transfer_loss = model(data_source, data_target, label_source)
            loss = clf_loss - transfer_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    draw_feature(model, source_loader, target_loader)
    pred_single(trainmodel, j, i, 30)
    acc, recall, specificity, precision, f1 = test2(model,j,i)
    print('test acc %f' % (acc))
    return acc, recall, specificity, precision, f1 

if __name__ == "__main__":
    print(torch.cuda.is_available())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    for j in range(1,2):
        a = []
        r = []
        s = []
        p = []
        f = []
        for i in range(10,11):
            torch.manual_seed(18) 
            torch.cuda.manual_seed(18) 
            trainmodel = model.Model().to(device)
            acc, recall, specificity, precision, f1 = train(trainmodel,j,i,30)
            a.append(acc)
            r.append(recall)
            s.append(specificity)
            p.append(precision)
            f.append(f1)
            torch.cuda.empty_cache()
        print("-----------------------evaluation metrics------------------------")
        print("acc:",np.mean(a))
        print("recall:",np.mean(r))
        print("specificity:",np.mean(s))
        print("precision:",np.mean(p))
        print("f1:",np.mean(f))
        