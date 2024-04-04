import os
import numpy as np
import shap
import torch
from torch_geometric.loader import DataLoader  # 更新导入路径
from torch_geometric.data import Data
from model import MYMODEL  # 确保这是您的模型
import matplotlib.pyplot as plt
from smiles2topology import MyOwnDataset

class ModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super(ModelWrapper, self).__init__()
        self.model = model

    def forward(self, x):
        # 创建一个包含必要信息的Data对象，这里只包含x作为示例
        data = Data(x=x)
        return self.model(data)

def shap_feature_importance(model, loader, break_ratio=0.01):
    """利用SHAP计算特征的重要性分数，并在处理了一定比例的数据后中断。
    修改后的版本以适应torch_geometric的数据和模型结构。
    """
    model.eval()
    background_batch = next(iter(loader))
    background_data = background_batch if isinstance(background_batch, torch.Tensor) else background_batch[0]
    background_features = background_data.x.to(device)  # 仅提取特征数据

    model_wrapper = ModelWrapper(model)
    explainer = shap.DeepExplainer(model_wrapper, background_features)

    total_batches = len(loader)
    break_after = int(total_batches * break_ratio)

    accumulated_shap_values = None
    processed_batches = 0

    for batch_data in loader:
        data = batch_data if isinstance(batch_data, torch.Tensor) else batch_data[0]
        batch_features = data.x.to(device)
        shap_values = explainer.shap_values(batch_features)

        if accumulated_shap_values is None:
            accumulated_shap_values = np.array(shap_values)
        else:
            accumulated_shap_values += np.array(shap_values)

        processed_batches += 1
        if processed_batches >= break_after:
            print(f"Processed {break_ratio * 100}% of batches, stopping early.")
            break

    average_shap_values = accumulated_shap_values / processed_batches
    return average_shap_values

# 示例用法
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = MYMODEL().to(device)  # 加载您的模型
test_dataset = MyOwnDataset(os.path.join('Datasets', 'Llinas2020'))
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)  # 使用您的数据集
shap_values_list = shap_feature_importance(model, test_loader)
