"""
By xunyoyo & kesmeey
Part of code come from GitHub:
https://github.com/ziduzidu/CSDTI
"""

import os

import numpy as np
from torch_geometric.data import DataLoader
from sklearn.metrics import r2_score, mean_squared_error

from model import MYMODEL
from smiles2topology import *


def val(model, dataloader, device):
    model.eval()

    pred_list = []
    label_list = []

    for data in dataloader:
        data = data.to(device)

        with torch.no_grad():
            pred = model(data)
            label = data.y
            pred_list.append(pred.view(-1).detach().cpu().numpy())
            label_list.append(label.detach().cpu().numpy())
            # update

    pred = np.concatenate(pred_list, axis=0)
    label = np.concatenate(label_list, axis=0)

    # print(pred, label)
    epoch_r2 = r2_score(label, pred)
    epoch_rmse = mean_squared_error(label, pred)

    return epoch_rmse, epoch_r2


def main():
    params = dict(
        data_root="Datasets",
        save_dir="save",
        dataset="Lovric",
        model_name="Epoch 249-643, Train Loss_ 0.8742, Val Loss_ 0.9914, Train R2_ 0.8440, Val R2_ 0.7865.pt"
    )

    save_dir = params.get("save_dir")
    DATASET = params.get("dataset")
    data_root = params.get("data_root")
    fpath = os.path.join(data_root, DATASET)

    test_dataset = MyOwnDataset(fpath, train=True)
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=True, num_workers=4)

    device = torch.device('cuda:0')
    model = MYMODEL().to(device)
    model.load_state_dict(torch.load(os.path.join(save_dir, params.get("model_name"))))

    rmse, r2 = val(model, test_loader, device)

    msg = f"Loss: {rmse:.4f}, R2: {r2:.4f}"
    print(msg)


if __name__ == '__main__':
    main()
