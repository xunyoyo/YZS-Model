"""
By xunyoyo & kesmeey
用处: 计算两个数据集相同分子的LogS0数据的一致化程度
使用方法: 两个分子的数据集仅保留两列[SMILE,LogS0]即可
"""

import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


def main():
    # 读取两个CSV文件的前两列
    df1 = pd.read_csv('1.csv', usecols=[0, 1], header=None)
    df2 = pd.read_csv('2.csv', usecols=[0, 1], header=None)

    # 重命名列以方便合并
    df1.columns = ['SMILES', 'Target_1']
    df2.columns = ['SMILES', 'Target_2']

    # 找到两个数据集中相同的分子及其对应的目标值
    common_molecules = pd.merge(df1, df2, on='SMILES')

    # 计算MAE和RMSE
    mae = mean_absolute_error(common_molecules['Target_1'], common_molecules['Target_2'])
    rmse = mean_squared_error(common_molecules['Target_1'], common_molecules['Target_2'], squared=False)
    correlation = common_molecules['Target_1'].corr(common_molecules['Target_2'])

    # 输出MAE和RMSE
    print(f'MAE: {mae}')
    print(f'RMSE: {rmse}')
    print(f'相关系数: {correlation}')


if __name__ == '__main__':
    main()
