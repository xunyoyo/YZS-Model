import os
import math
import numpy as np
import torch.optim as optim
import torch
import torch.nn as nn
from torch_geometric.data import DataLoader
import torch.nn.functional as F

from smiles2topology import *

device = torch.device('cuda:0')

if __name__ == '__main__':
    path1 = r'Datasets'
    print(os.path.abspath(path1))
    train_set = MyOwnDataset(path1, train=True)
    print(len(train_set))
    train_loader = DataLoader(train_set, batch_size=512, shuffle=True, num_workers=8)

    # os.path.join()
    # train_loader = DataLoader()