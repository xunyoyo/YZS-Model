import torch
from torch_geometric.data import DataLoader
from model import MYMODEL
from smiles2topology import MyOwnDataset
import os
import numpy as np
from sklearn.metrics import mean_squared_error

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def permutation_feature_importance(model, data_loader, num_features=24):
    # model.load_state_dict(torch.load(os.path.join("save",
    #                                               "Epoch 23-643, Train Loss_ 1.4097, Val Loss_ 1.3560, Train R2_ 0.5915, Val R2_ 0.6253.pt")))
    model.eval()

    feature_importance = np.zeros(num_features)  # 初始化特征重要性数组

    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)
            original_output = model(batch).cpu().numpy()
            original_mse = mean_squared_error(batch.y.cpu().numpy(), original_output)

            for i in range(num_features):  # 对每个特征进行排列
                shuffled_batch = batch.clone()
                shuffled_batch.x[:, i] = shuffled_batch.x[torch.randperm(shuffled_batch.num_nodes), i]
                shuffled_output = model(shuffled_batch).cpu().numpy()
                shuffled_mse = mean_squared_error(shuffled_batch.y.cpu().numpy(), shuffled_output)
                feature_importance[i] += shuffled_mse - original_mse

    feature_importance /= len(data_loader)
    feature_ranking = np.argsort(feature_importance)[::-1]

    print("Feature ranking (from most to least important):", feature_ranking)

if __name__ == "__main__":
    dataset = MyOwnDataset(os.path.join('Datasets', 'Lovric'))
    mymodel = MYMODEL().to(device)
    my_data_loader = DataLoader(dataset, batch_size=8, shuffle=True)
    permutation_feature_importance(mymodel, my_data_loader)
