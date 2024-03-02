"""
By xunyoyo & kesmeey
Part of code come from GitHub:
https://github.com/ziduzidu/CSDTI
"""

import math

import torch.nn as nn
import torch.optim as optim
import logging
from torch.utils.data import random_split
from torch_geometric.data import DataLoader
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from model import MYMODEL
from smiles2topology import *


def objective(params):
    best_val_loss = float('inf')
    batch_size = 64
    data_root = "Datasets"
    DATASET = "Ceasvlu"
    lr = 0.0005
    fpath = os.path.join(data_root, DATASET)

    full_dataset = MyOwnDataset(fpath, train=True)

    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size

    train_set, val_set = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=2)

    device = torch.device('cuda:0')
    # model = MYMODEL(24, int(params['dim']), params['dropout'], int(params['depth']), int(params['heads']), int(params['dim_head']),
    #                 int(params['mlp_dim'])).to(device)
    model = MYMODEL(24, int(params['dim']), params['dropout'], int(params['depth']), int(params['heads'])).to(device)

    epochs = 60000
    steps_per_epoch = 15
    num_iter = math.ceil((epochs * steps_per_epoch) / len(train_loader))
    cunt = 0

    optimizer = optim.Adam(model.parameters(), lr=0.00066)
    criterion = nn.MSELoss()

    for epoch in tqdm(range(num_iter)):

        model.train()
        for data in train_loader:
            data = data.to(device)
            pred = model(data)
            loss = criterion(pred.view(-1), data.y.view(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                pred = model(data)
                loss = criterion(pred.view(-1), data.y.view(-1))

                val_loss += loss.item() * data.num_graphs

        val_loss = math.sqrt(val_loss / len(val_loader.dataset))

        cunt += 1

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            cunt = 0
        elif cunt > 10:
            break

        print(params)

    logging.info(f"{params},best_val_loss:【{best_val_loss}】")
    return {'loss': best_val_loss, 'status': STATUS_OK}


def main():
    space = {
        'lr': hp.uniform('lr', 0.0005, 0.002),
        'dim': hp.quniform('dim', 30, 40, 1),
        'dropout': hp.uniform('dropout', 0.25, 0.35),
        'depth': hp.quniform('depth', [6, 8, 10, 12]),
        'heads': hp.choice('heads', [8, 16]),
        # 'batch_size': hp.quniform('batch_size', 30, 40, 1),
    }

    trials = Trials()
    best = fmin(
        fn=objective,
        space=space,
        algo=tpe.suggest,
        max_evals=100,
        trials=trials
    )

    logging.info(f"最佳超参数：{best}")
    print("最佳超参数：", best)


if __name__ == '__main__':
    log_file = os.path.join('log', "model.log")
    logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(message)s')

    main()
