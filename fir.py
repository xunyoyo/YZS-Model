import os

import numpy as np
import torch
from torch_geometric.data import DataLoader
from model import MYMODEL  # 确保这是您的模型
import matplotlib.pyplot as plt
from smiles2topology import MyOwnDataset


def feature_importance(model, loader, num_features, break_ratio=0.01):
    """计算特征的重要性分数，并在处理了一定比例的数据后中断。

    Args:
        model: 训练好的模型。
        loader: 数据加载器。
        num_features: 特征的数量。
        break_ratio: 处理数据的比例，用于中断计算。
    """
    model.eval()
    importance_scores = np.zeros(num_features)
    total_batches = len(loader)
    break_after = int(total_batches * break_ratio)  # 在处理了这么多批次后中断

    for i, data in enumerate(loader, start=1):
        if i > break_after:
            print(f"Breaking after processing {break_ratio * 100}% of batches.")
            break

        data = data.to(device)
        original_pred = model(data).detach().cpu().numpy()
        for feature_idx in range(num_features):
            modified_data = data.clone()
            modified_data.x[:, feature_idx] = 0  # 将特定特征置零
            modified_pred = model(modified_data).detach().cpu().numpy()
            loss_increase = np.abs(original_pred - modified_pred).mean()  # 计算预测差异
            importance_scores[feature_idx] += loss_increase

    importance_scores /= break_after  # 使用实际处理的批次数来取平均值
    return importance_scores


# 示例用法
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = MYMODEL().to(device)  # 加载您的模型
test_dataset=MyOwnDataset(os.path.join('Datasets', 'Llinas2020'))
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)  # 使用您的数据集

num_features = 92  # 假设您有92个特征
importance_scores = feature_importance(model, test_loader, num_features)
print(importance_scores)
# 特征名称 - 假设我们简单地用数字来代表每个特征
feature_names = [f"Feature {i}" for i in range(1, num_features + 1)]

# 创建一个条形图
plt.figure(figsize=(15, 10))  # 设置图形的大小
plt.barh(feature_names, importance_scores, color='skyblue')  # 水平条形图
plt.xlabel('Importance Score')  # x轴标签
plt.ylabel('Features')  # y轴标签
plt.title('Feature Importance')  # 图形标题
plt.gca().invert_yaxis()  # 逆转y轴，使得重要性最高的特征在顶部
plt.show()
# 假定importance_scores是已经计算好的特征重要性得分
# feature_names是对应的特征名称列表

# 根据importance_scores对特征名称进行排序，获取排序后的索引
sorted_indices = np.argsort(importance_scores)[::-1]  # 降序排序

# 选择得分最高的前10个特征的索引
top_10_indices = sorted_indices[:10]


element_symbols = ['K', 'Y', 'V', 'Sm', 'Dy', 'In', 'Lu', 'Hg', 'Co', 'Mg', 'Cu', 'Rh', 'Hf', 'O', 'As', 'Ge', 'Au', 'Mo', 'Br', 'Ce', 'Zr', 'Ag', 'Ba', 'N', 'Cr', 'Sr', 'Fe', 'Gd', 'I', 'Al', 'B', 'Se', 'Pr', 'Te', 'Cd', 'Pd', 'Si', 'Zn', 'Pb', 'Sn', 'Cl', 'Mn', 'Cs', 'Na', 'S', 'Ti', 'Ni', 'Ru', 'Ca', 'Nd', 'W', 'H', 'Li', 'Sb', 'Bi', 'La', 'Pt', 'Nb', 'P', 'F', 'C', 'Re', 'Ta', 'Ir', 'Be', 'Tl']
feature_names = element_symbols + ['Degree_1', 'Degree_2', 'Degree_3', 'Degree_4', 'Degree_5', 'Degree_6', 'Degree_7', 'Degree_8', 'FormalCharge', 'RadicalElectrons', 'Hybridization_s', 'Hybridization_sp', 'Hybridization_sp2', 'Hybridization_sp3', 'Hybridization_sp3d', 'Hybridization_sp3d2', 'Hybridization_other', 'Aromatic', 'Hydrogen_0', 'Hydrogen_1', 'Hydrogen_2', 'Hydrogen_3', 'Hydrogen_4', 'Chirality', 'Chirality_R', 'Chirality_S']
sorted_indices = np.argsort(importance_scores)[::-1]  # 得到降序排序的索引
top_10_indices = sorted_indices[:10]  # 取得分最高的前10个索引

top_10_feature_names = [feature_names[i] for i in top_10_indices]  # 根据索引获取特征名称
top_10_importance_scores = importance_scores[top_10_indices]  # 获取对应的重要性得分

# 绘制条形图
plt.figure(figsize=(10, 7))  # 设置图形的大小
plt.barh(top_10_feature_names, top_10_importance_scores, color='skyblue')  # 水平条形图展示这些特征
plt.xlabel('Importance Score')  # x轴标签
plt.ylabel('Features')  # y轴标签
plt.title('Top 10 Feature Importance')  # 图形标题
plt.gca().invert_yaxis()  # 逆转y轴，让得分最高的特征显示在顶部
plt.show()




# 新的特征分类和对应的原始特征
new_feature_categories = {
    'Elemental': element_symbols,  # 所有原子元素
    'Degree': ['Degree_1', 'Degree_2', 'Degree_3', 'Degree_4', 'Degree_5', 'Degree_6', 'Degree_7', 'Degree_8'],
    'FormalCharge': ['FormalCharge'],
    'RadicalElectrons': ['RadicalElectrons'],
    'Hybridization': ['Hybridization_s', 'Hybridization_sp', 'Hybridization_sp2', 'Hybridization_sp3', 'Hybridization_sp3d', 'Hybridization_sp3d2', 'Hybridization_other'],
    'Aromatic': ['Aromatic'],
    'Hydrogen': ['Hydrogen_0', 'Hydrogen_1', 'Hydrogen_2', 'Hydrogen_3', 'Hydrogen_4'],
    'Chirality': ['Chirality', 'Chirality_R', 'Chirality_S']
}

# 初始化一个新的得分矩阵
new_scores = np.zeros(len(new_feature_categories))

# 对于新的特征类别中的每一类，累加其对应的旧特征的重要性得分
for i, (category, old_features) in enumerate(new_feature_categories.items()):
    # 找到旧特征名称在 feature_names 中的索引
    indices = [feature_names.index(old_feature) for old_feature in old_features if old_feature in feature_names]
    # 累加这些索引对应的重要性得分
    new_scores[i] = importance_scores[indices].sum()

# 更新 feature_names 为新的特征分类名称
new_feature_names = list(new_feature_categories.keys())

# 下面是使用新的特征名称和得分来进行可视化等操作
# 例如，打印新的特征名称和对应的重要性得分
for name, score in zip(new_feature_names, new_scores):
    print(f"{name}: {score}")

colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'orange', 'purple', 'lime', 'pink']

explode = [0.1] + [0] * (len(new_feature_names) - 1)  # 只分离第一部分

# 创建饼图
plt.figure(figsize=(8, 6))
plt.pie(new_scores, labels=new_feature_names, autopct='%1.1f%%', startangle=140, explode=explode)

# 添加图例
plt.legend(new_feature_names, loc="best")

# 添加标题
plt.title('Feature Importance Distribution')


# 显示图形
plt.show()