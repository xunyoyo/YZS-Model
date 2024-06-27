"""
By xunyoyo & kesmeey
Part of code come from GitHub:
https://github.com/ziduzidu/CSDTI
"""

import logging
import math
import datetime

import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import r2_score
from torch.utils.data import random_split
from torch_geometric.data import DataLoader

from model import YZS
from smiles2topology import *


def save_model_dict(model, model_dir, msg):
    model_path = os.path.join(model_dir, msg + '.pt')
    torch.save(model.state_dict(), model_path)
    print("model has been saved to %s." % model_path)


def main():

    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = os.path.join('log', f"{current_time}-model.log")

    logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(message)s')
    logging.info('Training started')

    params = dict(
        data_root="Datasets",
        save_dir="save",
        dataset="Ceasvlu",
        save_model=True,
        lr=0.0005,
        batch_size=128,
        is_using_trained_data=False,
        model_name=""
    )

    save_dir = params.get("save_dir")
    save_model = params.get("save_model")
    DATASET = params.get("dataset")
    data_root = params.get("data_root")
    fpath = os.path.join(data_root, DATASET)

    full_dataset = MyOwnDataset(fpath, train=True)

    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size

    train_set, val_set = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=64, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=64, shuffle=True, num_workers=4)

    device = torch.device('cuda:0')

    model = YZS(92, 98, 0.30467697373969527, 4, 16).to(device)

    if params.get('is_using_trained_data'):
        model.load_state_dict(torch.load(os.path.join(save_dir, params.get("model_name"))))

    epochs = 6000
    steps_per_epoch = 15
    num_iter = math.ceil((epochs * steps_per_epoch) / len(train_loader))

    optimizer = optim.Adam(model.parameters(), lr=params.get("lr"))
    criterion = nn.MSELoss()

    best_val_r2 = -float('inf')
    epochs_no_improve = 0
    early_stop_epoch = 50

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

        model.eval()
        val_loss = 0.0
        val_preds = []
        val_targets = []

        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                pred = model(data)
                loss = criterion(pred.view(-1), data.y.view(-1))

                val_loss += loss.item() * data.num_graphs
                val_preds.extend(pred.view(-1).detach().cpu().numpy())
                val_targets.extend(data.y.view(-1).detach().cpu().numpy())

        val_r2 = r2_score(val_targets, val_preds)

        train_loss = math.sqrt(train_loss / len(train_loader.dataset))
        val_loss = math.sqrt(val_loss / len(val_loader.dataset))

        msg = (f"Epoch {epoch + 1}-{num_iter}, Train Loss_ {train_loss:.4f}, Val Loss_ {val_loss:.4f}, "
               f"Train R2_ {train_r2:.4f}, Val R2_ {val_r2:.4f}")

        logging.info(msg)
        print(msg)

        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
            epochs_no_improve = 0
            if save_model:
                save_model_dict(model, save_dir, msg)
                msg = os.path.join(save_dir, msg + '.pt')
                logging.info(f"model has been saved to save {msg}")

        else:
            epochs_no_improve += 1

        if epochs_no_improve >= early_stop_epoch:
            logging.info(f"Early stopping triggered after {early_stop_epoch} epochs without improvement.")
            print(f"Early stopping triggered after {early_stop_epoch} epochs without improvement.")
            break


if __name__ == '__main__':

    log_dir = 'log'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    main()