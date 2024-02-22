"""
By xunyoyo & kesmeey
Part of code come from GitHub:
https://github.com/ziduzidu/CSDTI
"""

import math

import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import r2_score
from torch.utils.data import random_split
from torch_geometric.data import DataLoader
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

from model import MYMODEL
from smiles2topology import *


def objective(params):
    batch_size = 128
    data_root = "Datasets"
    DATASET = "Ceasvlu"
    lr = 0.0005
    fpath = os.path.join(data_root, DATASET)

    full_dataset = MyOwnDataset(fpath, train=True)

    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size

    train_set, val_set = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=10)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=10)

    device = torch.device('cuda:0')
    model = MYMODEL(24, int(params['dim']), params['dropout'], int(params['depth']), int(params['heads']), int(params['dim_head']),
                    int(params['mlp_dim'])).to(device)

    epochs = 1000
    steps_per_epoch = 15
    num_iter = math.ceil((epochs * steps_per_epoch) / len(train_loader))

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(num_iter):

        model.train()
        train_loss = 0.0
        train_preds = []
        train_targets = []

        for data in train_loader:
            data = data.to(device)
            pred = model(data)
            loss = criterion(pred.view(-1), data.y.view(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * data.num_graphs
            train_preds.extend(pred.view(-1).detach().cpu().numpy())
            train_targets.extend(data.y.view(-1).detach().cpu().numpy())

        train_r2 = r2_score(train_targets, train_preds)
        train_loss = math.sqrt(train_loss / len(train_loader.dataset))

        msg = (f"Epoch {epoch + 1}-{num_iter}, Train Loss_ {train_loss:.4f}, "
               f"Train R2_ {train_r2:.4f}")
        print(msg)

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for data in val_loader:
            data = data.to(device)
            pred = model(data)
            loss = criterion(pred.view(-1), data.y.view(-1))

            val_loss += loss.item() * data.num_graphs

    val_loss = math.sqrt(val_loss / len(val_loader.dataset))

    return {'loss': val_loss, 'status': STATUS_OK}


def main():
    space = {
        'dim': hp.quniform('dim', 24, 72, 2),
        'dropout': hp.uniform('dropout', 0.1, 0.5),
        'depth': hp.quniform('depth', 2, 8, 1),
        'heads': hp.quniform('heads', 2, 16, 2),
        'dim_head': hp.choice('dim_head', [16, 24, 32, 64]),
        'mlp_dim': hp.choice('mlp_dim', [32, 64, 128, 256, 512, 1024])
    }

    trials = Trials()
    best = fmin(
        fn=objective,
        space=space,
        algo=tpe.suggest,
        max_evals=100,
        trials=trials
    )

    print("最佳超参数：", best)


if __name__ == '__main__':
    main()
