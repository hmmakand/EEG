import torch
import numpy as np
import pandas as pd
from torch.utils.data.dataset import Dataset

class DatasetFromCSV(Dataset):
    def __init__(self, csv_path):
        self.data = pd.read_csv(csv_path,header=None)
        # 被试编号
        self.domains = np.asarray(self.data.iloc[:, 0])
        # 类别标签
        self.labels = np.asarray(self.data.iloc[:, 1])
 
    def __getitem__(self, index):
        single_label = self.labels[index]
        single_domain = self.domains[index]
        # 数据
        to_np = np.asarray(self.data.iloc[index][2:642]).astype(float)
        to_tensor = torch.tensor(to_np)
        return (to_tensor, single_label, single_domain)
 
    def __len__(self):
        return len(self.data.index)