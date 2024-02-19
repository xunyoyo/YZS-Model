"""
By xunyoyo & kesmeey
Part of code come from GitHub:
https://github.com/ziduzidu/CSDTI
"""

import argparse
import math
import os

import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import r2_score
from torch.utils.data import random_split
from torch_geometric.data import DataLoader

from model import MYMODEL
from smiles2topology import *


def save_model_dict(model, model_dir, msg):
    model_path = os.path.join(model_dir, msg + '.pt')
    torch.save(model.state_dict(), model_path)
    print("model has been saved to %s." % model_path)


def main():

    params = dict(
        data_root="Datasets",
        save_dir="save",
        dataset="Ceasvlu",
        model_name="Epoch 37-643, Train Loss_ 1.4617, Val Loss_ 1.3560, Train R2_ 0.5638, Val R2_ 0.6029.pt"
    )

    save_dir = params.get("save_dir")
    save_model = params.get("save_model")
    DATASET = params.get("dataset")
    data_root = params.get("data_root")
    fpath = os.path.join(data_root, DATASET)

    test_dataset = MyOwnDataset(fpath, train=True)

    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True, num_workers=4)

    device = torch.device('cuda:0')

    model = MYMODEL().to(device)

    model.load_state_dict(torch.load(os.path.join(save_dir, params.get("model_name"))))

    criterion = nn.MSELoss()







if __name__ == '__main__':
    main()
