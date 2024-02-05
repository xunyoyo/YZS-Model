import os.path as op
import numpy
import pandas as pd
import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric import data as DATA
import networkx as nx
from rdkit import Chem
from rdkit.Chem import MolFromSmiles
from rdkit.Chem import ChemicalFeatures
from rdkit import RDConfig
from tqdm import tqdm
fdef_name = op.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
chem_feature_factory = ChemicalFeatures.BuildFeatureFactory(fdef_name)


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

    def pre_process(self, data_path, data_dict):
        file = pd.read_csv(data_path)

        data_lists = []
        for index, row in file.iterrows():
            smiles = row['isomeric_smiles']
            logS = row['logS0']

            # 重构一下process方法
    def process(self):
        file_train = pd.read_csv(self.raw_paths[0])
        file = file_train

        smiles = file['isomeric_smiles'].unique()
        graph_dict = dict()

        for smile in tqdm(smiles, total=len(smiles)):
            mol = Chem.MolFromSmiles(smile)
            g = self.mol2graph(mol)
            graph_dict[smile] = g

        # train_list =

        # if self.pre_filter is not None:
        #     train_list = [train for train in train_list if self.pre_filter(train)]
        #     test_list = [test for test in test_list if self.pre_filter(test)]
        #
        # if self.pre_transform is not None:
        #     train_list = [self.pre_transform(train) for train in train_list]
        #     test_list = [self.pre_transform(test) for test in test_list]

    def mol2graph(self, mol):
        if mol is None:
            return None

        features = chem_feature_factory.GetFeaturesForMol(mol)
        graph = nx.DiGraph()

        for i in range(mol.GetNumAtoms()):
            atom_i = mol.GetAtomWithIdx(i)
            graph.add_node(i,
                       # a_type=atom_i.GetSymbol(),
                       # a_num=atom_i.GetAtomicNum(),
                       # acceptor=0,
                       # donor=0,
                       # aromatic=atom_i.GetIsAromatic(),
                       # hybridization=atom_i.GetHybridization(),
                       # num_h=atom_i.GetTotalNumHs(),
                       #
                       # # 5 more node features
                       # ExplicitValence=atom_i.GetExplicitValence(),
                       # FormalCharge=atom_i.GetFormalCharge(),
                       # ImplicitValence=atom_i.GetImplicitValence(),
                       # NumExplicitHs=atom_i.GetNumExplicitHs(),
                       # NumRadicalElectrons=atom_i.GetNumRadicalElectrons(),
                       )


        for i in range(len(features)):
            pass

        for i in range(mol.GetNumAtoms()):
            pass
