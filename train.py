import logging
import math
import datetime
import os

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader, random_split
from torch_geometric.data import DataLoader as GeoDataLoader

from model import YZS
from smiles2topology import MyOwnDataset


def setup_logging():
    """ Sets up logging with a timestamped log file. """
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = os.path.join('log', f"{current_time}-model.log")
    logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(message)s')
    logging.info('Training started')


def save_model(model, directory, epoch, r2_score):
    """ Saves the model state to a file. """
    model_path = os.path.join(directory, f'model_epoch_{epoch}_r2_{r2_score:.4f}.pt')
    torch.save(model.state_dict(), model_path)
    logging.info(f"Model saved to {model_path}")


def train_epoch(model, device, data_loader, optimizer, criterion):
    """ Trains the model for one epoch and returns average loss and R2 score. """
    model.train()
    total_loss = 0
    preds, targets = [], []
    for data in data_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output.view(-1), data.y.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
        preds.extend(output.view(-1).detach().cpu().numpy())
        targets.extend(data.y.view(-1).detach().cpu().numpy())
    avg_loss = math.sqrt(total_loss / len(data_loader.dataset))
    r2 = r2_score(targets, preds)
    return avg_loss, r2


def validate(model, device, data_loader, criterion):
    """ Validates the model and returns average loss and R2 score. """
    model.eval()
    total_loss = 0
    preds, targets = [], []
    with torch.no_grad():
        for data in data_loader:
            data = data.to(device)
            output = model(data)
            loss = criterion(output.view(-1), data.y.view(-1))
            total_loss += loss.item() * data.num_graphs
            preds.extend(output.view(-1).detach().cpu().numpy())
            targets.extend(data.y.view(-1).detach().cpu().numpy())
    avg_loss = math.sqrt(total_loss / len(data_loader.dataset))
    r2 = r2_score(targets, preds)
    return avg_loss, r2


def main():
    setup_logging()

    params = {
        "data_root": "Datasets",
        "save_dir": "save",
        "dataset": "Ceasvlu",
        "save_model": True,
        "lr": 0.0005,
        "batch_size": 128,
        "is_using_trained_data": False,
        "model_name": ""
    }

    os.makedirs(params["save_dir"], exist_ok=True)

    device = torch.device('cuda:0')
    model = YZS(92, 98, 0.30467697373969527, 4, 16).to(device)

    if params["is_using_trained_data"]:
        model_path = os.path.join(params["save_dir"], params["model_name"])
        model.load_state_dict(torch.load(model_path))
        logging.info(f"Loaded model from {model_path}")

    full_dataset = MyOwnDataset(os.path.join(params["data_root"], params["dataset"]), train=True)
    train_size = int(0.9 * len(full_dataset))
    train_set, val_set = random_split(full_dataset, [train_size, len(full_dataset) - train_size])
    train_loader = GeoDataLoader(train_set, batch_size=64, shuffle=True, num_workers=4)
    val_loader = GeoDataLoader(val_set, batch_size=64, shuffle=False, num_workers=4)

    optimizer = optim.Adam(model.parameters(), lr=params["lr"])
    criterion = nn.MSELoss()

    best_val_r2 = -float('inf')
    epochs_no_improve = 0
    early_stop_epoch = 50

    for epoch in range(6000):
        train_loss, train_r2 = train_epoch(model, device, train_loader, optimizer, criterion)
        val_loss, val_r2 = validate(model, device, val_loader, criterion)

        logging.info(
            f"Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Train R2: {train_r2:.4f}, Val R2: {val_r2:.4f}")

        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
            epochs_no_improve = 0
            if params["save_model"]:
                save_model(model, params["save_dir"], epoch + 1, val_r2)
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= early_stop_epoch:
            logging.info("Early stopping triggered.")
            break


if __name__ == '__main__':
    main()
