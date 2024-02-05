import os.path as op
import numpy
import pandas as pd
import torch
from torch_geometric.data import InMemoryDataset
from rdkit import Chem
from rdkit.Chem import MolFromSmiles
from tqdm import tqdm


class MyOwnDataset(InMemoryDataset):

    def __init__(self, root, train=True, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        if train:
            self.data, self.slices = torch.load(self.processed_paths[0])
        else:
            self.data, self.slices = torch.load(self.processed_paths[1])

    @property
    def raw_file_names(self):
        return ['data_train.csv']

    @property
    def processed_file_names(self):
        return ['processed_data_train.pt']

    def download(self):
        pass

    def pro_process(self, data_path, data_dict):
        file = pd.read_csv(data_path)

        data_lists = []
        for index, row in file.iterrows():
            smiles = row['isomeric_smiles']
            logS = row['logS0']




            # 重构一下process方法
    def process(self):
        pass

    pass
