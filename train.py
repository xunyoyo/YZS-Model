import os
import numpy as np
import torch
from torch_geometric.data import DataLoader
from sklearn.metrics import r2_score, mean_squared_error
from model import MYMODEL  # 确保模型文件正确导入
# 确保MyOwnDataset类已正确定义，用于处理您的数据
from dataset import MyOwnDataset


def save_model(model, model_dir, epoch, val_r2):
    model_path = os.path.join(model_dir, f"model_epoch_{epoch}_valR2_{val_r2:.4f}.pt")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}.")


def train(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output.view(-1), data.y.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
    return total_loss / len(train_loader.dataset)


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    preds, targets = [], []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            output = model(data)
            loss = criterion(output.view(-1), data.y.view(-1))
            total_loss += loss.item() * data.num_graphs
            preds.append(output.view(-1).cpu().numpy())
            targets.append(data.y.view(-1).cpu().numpy())
    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
    rmse = mean_squared_error(targets, preds, squared=False)
    r2 = r2_score(targets, preds)
    return total_loss / len(loader.dataset), rmse, r2


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MYMODEL().to(device)  # 请根据实际模型调整构造函数
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()

    # 假设已经定义了MyOwnDataset
    train_dataset = MyOwnDataset("./data/train.csv")
    val_dataset = MyOwnDataset("./data/val.csv")
    test_dataset = MyOwnDataset("./data/test.csv")

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    for epoch in range(1, 101):  # 假设训练100个epochs
        train_loss = train(model, train_loader, optimizer, criterion, device)
        val_loss, val_rmse, val_r2 = evaluate(model, val_loader, criterion, device)
        test_loss, test_rmse, test_r2 = evaluate(model, test_loader, criterion, device)

        print(
            f"Epoch: {epoch}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val RMSE: {val_rmse:.4f}, Val R2: {val_r2:.4f}, Test Loss: {test_loss:.4f}, Test RMSE: {test_rmse:.4f}, Test R2: {test_r2:.4f}")

        # 可以根据验证集的R2或其他指标保存最佳模型
        if val_r2 > 0.8:  # 假设使用0.8作为保存阈值
            save_model(model, "./models", epoch, val_r2)


if __name__ == "__main__":
    main()
