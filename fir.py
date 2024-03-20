import shap
import torch
from torch_geometric.data import DataLoader, Data
from model import MYMODEL
from smiles2topology import MyOwnDataset
import os
import numpy as np
from sklearn.metrics import mean_squared_error
from pdpbox import pdp
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def permutation_feature_importance(model, data_loader, num_features=24):
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

def data_loader_to_numpy(data_loader):
    data_list = []
    max_nodes = 0  # 记录最大节点数

    for data in data_loader:
        # 获取节点数
        num_nodes = data.num_nodes
        if num_nodes > max_nodes:
            max_nodes = num_nodes

        # 将 DataBatch 中的张量提取出来并进行填充或截断
        x = data.x.cpu().numpy()
        edge_index = data.edge_index.cpu().numpy()
        edge_index2 = data.edge_index2.cpu().numpy()
        edge_attr = data.edge_attr.cpu().numpy()
        y = data.y.cpu().numpy()

        # 假设每个 DataBatch 的 x 和 edge_attr 需要填充到相同的大小
        pad_size = max_nodes - num_nodes
        x_padded = np.pad(x, ((0, pad_size), (0, 0)), mode='constant', constant_values=0)
        edge_attr_padded = np.pad(edge_attr, ((0, pad_size), (0, 0)), mode='constant', constant_values=0)

        # 将处理后的数据收集到列表中
        data_list.append((x_padded, edge_index, edge_index2, edge_attr_padded, y))

    # 分别处理列表中的每个元素
    x_concat = np.concatenate([item[0] for item in data_list], axis=0)
    edge_index_concat = np.concatenate([item[1] for item in data_list], axis=1)
    edge_index2_concat = np.concatenate([item[2] for item in data_list], axis=1)
    edge_attr_concat = np.concatenate([item[3] for item in data_list], axis=0)
    y_concat = np.concatenate([item[4] for item in data_list], axis=0)

    return x_concat, edge_index_concat, edge_index2_concat, edge_attr_concat, y_concat

def model_wrapper(model, inputs):
    x, edge_index, edge_index2, edge_attr = inputs
    data = Data(x=x, edge_index=edge_index, edge_index2=edge_index2, edge_attr=edge_attr)
    data = data.to(device)
    return model(data)
def permutation_feature_importance2(model, data_loader, num_features=24):
    model.eval()

    feature_importance = np.zeros(num_features)  # 初始化特征重要性数组

    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)
            original_output = model(batch.x, batch.edge_index, batch.edge_index2, batch.edge_attr).cpu().numpy()
            original_mse = mean_squared_error(batch.y.cpu().numpy(), original_output)

            for i in range(num_features):  # 对每个特征进行排列
                shuffled_batch = batch.clone()
                shuffled_batch.x[:, i] = shuffled_batch.x[torch.randperm(shuffled_batch.num_nodes), i]
                shuffled_output = model(shuffled_batch.x, shuffled_batch.edge_index, shuffled_batch.edge_index2, shuffled_batch.edge_attr).cpu().numpy()
                shuffled_mse = mean_squared_error(shuffled_batch.y.cpu().numpy(), shuffled_output)
                feature_importance[i] += shuffled_mse - original_mse

    feature_importance /= len(data_loader)
    feature_ranking = np.argsort(feature_importance)[::-1]

    print("Feature ranking (from most to least important):", feature_ranking)

    # 将 DataLoader 转换为 NumPy 数组
    x, edge_index, edge_index2, edge_attr, y = data_loader_to_numpy(data_loader)

    # 计算SHAP值
    explainer = shap.Explainer(lambda inputs: model(*inputs), (x, edge_index, edge_index2, edge_attr))
    shap_values = explainer.shap_values((x, edge_index, edge_index2, edge_attr))

    # 计算每个原子符号的SHAP值的绝对值之和
    feature_importance_shap = np.abs(shap_values[0]).mean(axis=0)  # 假设是第一个类别的SHAP值

    # 打印原子符号的重要性
    feature_ranking_shap = np.argsort(feature_importance_shap)[::-1]
    print("Feature ranking based on SHAP values (from most to least important):", feature_ranking_shap)

    # 使用pdp生成部分依赖图
    for i in range(num_features):
        feature_name = f"Element_{i}"  # 假设每个特征表示一个元素
        # 获取元素特征的范围（假设范围为0到1）
        feature_range = (0, 1)
        # 生成部分依赖图
        pdp_goals = pdp.pdp_isolate(model, dataset=(x, edge_index, edge_index2, edge_attr), model_features=data_loader.dataset.get_nodes,
                                     feature=feature_name, grid_type='equal')
        pdp.pdp_plot(pdp_goals, feature_name, plot_lines=True, frac_to_plot=100, plot_pts_dist=True)

if __name__ == "__main__":
    params = dict(
        data_root="Datasets",
        save_dir="save",
        dataset="APtest",
        model_name="Epoch 62-720, Train Loss_ 0.9255, Val Loss_ 0.8965, Test1 Loss_ 0.6001, Test2 Loss_ 0.9728, Train R2_ 0.8257, Val R2_ 0.8208, Test1 R2_ 0.5433, Test2 R2_ 0.4095.pt"
    )

    save_dir = params.get("save_dir")
    DATASET = params.get("dataset")
    data_root = params.get("data_root")
    fpath = os.path.join(data_root, DATASET)

    dataset = MyOwnDataset(os.path.join('Datasets', 'APtest'))
    mymodel = MYMODEL().to(device)
    my_data_loader = DataLoader(dataset, batch_size=72, shuffle=True)
    permutation_feature_importance2(mymodel, my_data_loader)
