"""
By xunyoyo & kesmeey
Part of code come from GitHub:
https://github.com/ziduzidu/CSDTI
"""

import os
import math
import numpy as np
import torch.optim as optim
import torch
import torch.nn as nn
from torch_geometric.data import DataLoader
import argparse

from smiles2topology import *
from model import MYMODEL


def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--dataset', required=True, default='Ceasvlu' ,help='XD')
    parser.add_argument('--save_model', action='store_true', help='whether save model or not')
    parser.add_argument('--lr', type=float, default=0.0005, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=256, help='batch_size')
    args = parser.parse_args()

    params = dict(
        data_root="Datasets",
        save_dir="save",
        # dataset=args.dataset,
        save_model=args.save_model,
        lr=args.lr,
        batch_size=args.batch_size
    )

    # DATASET = params.get("dataset")
    DATASET = "Ceasvlu"
    save_model = params.get("save_model")
    data_root = params.get("data_root")
    fpath = os.path.join(data_root, DATASET)

    train_set = MyOwnDataset(fpath, train=True)

    train_loader = DataLoader(train_set, batch_size=10, shuffle=True, num_workers=2)

    device = torch.device('cuda:0')

    model = MYMODEL().to(device)

    epochs = 3000
    steps_per_epoch = 10
    num_iter = math.ceil((epochs * steps_per_epoch) / len(train_loader))
    break_flag = False

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    global_step = 0
    global_epoch = 0
    early_stop_epoch = 400

    model.train()

    for i in range(num_iter):
        if break_flag:
            break

        for data in train_loader:
            global_step += 1

            data = data.to(device)
            pred = model(data)

            loss = criterion(pred.view(-1), data.y.view(-1))

            if i % 100 == 0:
                print("data:",data.y)
                print("pred:",pred)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


if __name__ == '__main__':
    main()
