import pandas as pd
import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric import data as DATA
from rdkit import Chem
from rdkit.Chem import rdmolops
from tqdm import tqdm

def atom_feature(mol):
    """
    Generate atom features and adjacency matrix from a molecule object.
    
    Args:
        mol (Mol): A molecule object from RDKit.
    
    Returns:
        features (Tensor): A tensor containing atom features.
        edge_index (Tensor): Edge indices in COO format for graph representation.
        edge_attr (Tensor): Attributes of the edges (bonds).
        adj (numpy.ndarray): Adjacency matrix of the molecule.
    """
    # List of elements in the dataset for encoding.
    symbols = ['K', 'Y', 'V', 'Sm', 'Dy', 'In', 'Lu', 'Hg', 'Co', 'Mg',  
               'Cu', 'Rh', 'Hf', 'O', 'As', 'Ge', 'Au', 'Mo', 'Br', 'Ce',
               'Zr', 'Ag', 'Ba', 'N', 'Cr', 'Sr', 'Fe', 'Gd', 'I', 'Al',
               'B', 'Se', 'Pr', 'Te', 'Cd', 'Pd', 'Si', 'Zn', 'Pb', 'Sn',
               'Cl', 'Mn', 'Cs', 'Na', 'S', 'Ti', 'Ni', 'Ru', 'Ca', 'Nd',
               'W', 'H', 'Li', 'Sb', 'Bi', 'La', 'Pt', 'Nb', 'P', 'F', 'C',
               'Re', 'Ta', 'Ir', 'Be', 'Tl']

    # Possible hybridization states of the atoms.
    hybridizations = [
        Chem.rdchem.HybridizationType.S,
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2,
        'other',
    ]

    # Stereochemistry configurations.
    stereos = [
        Chem.rdchem.BondStereo.STEREONONE,
        Chem.rdchem.BondStereo.STEREOANY,
        Chem.rdchem.BondStereo.STEREOZ,
        Chem.rdchem.BondStereo.STEREOE,
    ]

    # Initialize storage for features and adjacency information.
    features = []
    xs = []
    adj = rdmolops.GetAdjacencyMatrix(mol, useBO=True)
    for atom in mol.GetAtoms():
        # Create feature vector for each atom.
        symbol = [0.] * len(symbols)
        symbol[symbols.index(atom.GetSymbol())] = 1.
        degree = [0.] * 8
        degree[atom.GetDegree()] = 1.
        formal_charge = atom.GetFormalCharge()
        radical_electrons = atom.GetNumRadicalElectrons()
        hybridization = [0.] * len(hybridizations)
        hybridization[hybridizations.index(atom.GetHybridization())] = 1.
        aromaticity = 1. if atom.GetIsAromatic() else 0.
        hydrogens = [0.] * 5
        hydrogens[atom.GetTotalNumHs()] = 1.
        chirality = 1. if atom.HasProp('_ChiralityPossible') else 0.
        chirality_type = [0.] * 2
        if atom.HasProp('_CIPCode'):
            chirality_type[['R', 'S'].index(atom.GetProp('_CIPCode'))] = 1.

        x = torch.tensor(symbol + degree + [formal_charge] +
                         [radical_electrons] + hybridization +
                         [aromaticity] + hydrogens + [chirality] +
                         chirality_type)
        xs.append(x)

        features = torch.stack(xs, dim=0)

    # Process bond information to create graph edges and attributes.
    edge_indices = []
    edge_attrs = []
    for bond in mol.GetBonds():
        edge_indices += [[bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()]]
        edge_indices += [[bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()]]

        bond_type = bond.GetBondType()
        single = 1. if bond_type == Chem.rdchem.BondType.SINGLE else 0.
        double = 1. if bond_type == Chem.rdchem.BondType.DOUBLE else 0.
        triple = 1. if bond_type == Chem.rdchem.BondType.TRIPLE else 0.
        aromatic = 1. if bond_type == Chem.rdchem.BondType.AROMATIC else 0.
        conjugation = 1. if bond.GetIsConjugated() else 0.
        ring = 1. if bond.IsInRing() else 0.
        stereo = [0.] * 4
        stereo[stereos.index(bond.GetStereo())] = 1.

        edge_attr = torch.tensor([single, double, triple, aromatic, conjugation, ring] + stereo)
        edge_attrs += [edge_attr, edge_attr]

    if len(edge_attrs) == 0:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, 10), dtype=torch.float)
    else:
        edge_index = torch.tensor(edge_indices).t().contiguous()
        edge_attr = torch.stack(edge_attrs, dim=0)
    return features, edge_index, edge_attr, adj

class MyOwnDataset(InMemoryDataset):
    """
    Custom dataset class for handling graph data derived from molecular structures.
    """
    def __init__(self, root, train=True, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        # Load different data slices based on whether it's for training or other purposes.
        if train:
            self.data, self.slices = torch.load(self.processed_paths[0])
        else:
            self.data, self.slices = torch.load(self.processed_paths[1])

    @property
    def raw_file_names(self):
        # Specifies the expected raw file names to be found in the directory.
        return ['data_train.csv']

    @property
    def processed_file_names(self):
        # Specifies the names of the processed files to be saved.
        return ['processed_data_train.pt']

    def download(self):
        # Method to handle data downloading (not implemented here).
        pass

    def process(self):
        """
        Processes raw data files to prepare graph representations suitable for graph neural network models.
        """
        df = pd.read_csv(self.raw_paths[0])
        data_list = []

        for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing molecules"):
            smile = row['SMILES']
            label = row['logS']  # Example property to be predicted
            mol = Chem.MolFromSmiles(smile)

            if mol is None:
                print(f"Cannot parse SMILE: {smile}")
                continue

            features, edge_index, edge_attr, adj = atom_feature(mol)

            graph = DATA.Data(
                x=torch.Tensor(features),
                edge_index=edge_index,
                edge_attr=edge_attr,
                y=torch.FloatTensor([label]),
                A=adj,
                smiles=str(smile),
            )
            print(graph)
            data_list.append(graph)

        if len(data_list) > 0:
            self.data, self.slices = self.collate(data_list)
            torch.save((self.data, self.slices), self.processed_paths[0])
        else:
            print("No data to save.")

        print("Data processing and saving completed.")



if __name__ == "__main__":
    pass
    # MyOwnDataset(os.path.join('Datasets','Lovric'))
    # MyOwnDataset(os.path.join('Datasets', 'Llinas2020'))
    # MyOwnDataset(os.path.join('Datasets', 'Llinas2020-2'))
    # MyOwnDataset(os.path.join('Datasets', 'Ceasvlu'))
