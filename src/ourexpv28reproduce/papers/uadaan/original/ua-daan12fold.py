# -*- coding: utf-8 -*-
"""
12折交叉训练+测试，直接读取已有的 train{i}.csv / test{i}.csv
用10个模块内的类/函数
"""
import os, torch, numpy as np, torch.optim as optim
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import dataloader, model

def seed_torch(seed=1234):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def make_loader(csv_file, batch_size, shuffle=True):
    return torch.utils.data.DataLoader(
        dataloader.DatasetFromCSV(csv_file),
        batch_size=batch_size, shuffle=shuffle, num_workers=0)

def train_test_one_fold(data_dir, fold_id, epoch=30, device='cuda'):
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    seed_torch()

    train_csv = os.path.join(data_dir, f'train{fold_id}.csv')
    test_csv  = os.path.join(data_dir, f'test{fold_id}.csv')
    assert os.path.exists(train_csv) and os.path.exists(test_csv), \
        f'缺少 {train_csv} 或 {test_csv}'

    train_loader = make_loader(train_csv, 98, True)
    test_loader  = make_loader(test_csv,   8, False)

    net = model.Model().to(device)
    optimizer = optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-3)

    # 训练
    for epoch_idx in range(1, epoch+1):
        net.train()
        net.epoch_based_processing(len(train_loader))
        tgt_iter = iter(test_loader)
        for src_x, src_y, _ in train_loader:
            try:
                tgt_x, _, _ = next(tgt_iter)
            except StopIteration:
                tgt_iter = iter(test_loader)
                tgt_x, _, _ = next(tgt_iter)
            src_x, src_y = src_x.to(device).float(), src_y.to(device).long()
            tgt_x = tgt_x.to(device).float()
            clf_loss, trans_loss = net(src_x, tgt_x, src_y)
            loss = clf_loss - trans_loss
            optimizer.zero_grad(); loss.backward(); optimizer.step()

    # 测试
    net.eval()
    x_list, y_list = [], []
    with torch.no_grad():
        for x, y, _ in test_loader:
            x_list.append(x); y_list.append(y)
    x_test = torch.cat(x_list).to(device).float()
    y_true = torch.cat(y_list).cpu().numpy()
    y_prob = net.predict(x_test).cpu()
    y_pred = y_prob.argmax(1).numpy()

    acc = accuracy_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred, zero_division=0)
    pre = precision_score(y_true, y_pred, zero_division=0)
    f1  = f1_score(y_true, y_pred, zero_division=0)
    return acc, rec, pre, f1

def main(data_dir='.', epoch=30):
    results = []
    for fold in range(1, 13):          # 1~12折
        acc, rec, pre, f1 = train_test_one_fold(data_dir, fold, epoch)
        results.append([acc, rec, pre, f1])
        print(f'Fold {fold}: acc={acc*100:.2f} recall={rec*100:.2f} '
              f'precision={pre*100:.2f} f1={f1*100:.2f}')
    avg = np.mean(results, axis=0)
    print('=== 12-fold average ===')
    print(f'acc={avg[0]*100:.2f} recall={avg[1]*100:.2f} '
          f'precision={avg[2]*100:.2f} f1={avg[3]*100:.2f}')

if __name__ == '__main__':
    import random
    main(data_dir='.', epoch=30)