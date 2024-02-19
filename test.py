"""
By xunyoyo & kesmeey
Part of code come from GitHub:
https://github.com/ziduzidu/CSDTI
"""

import os
import math
import numpy as np
import torch.optim as optim
import torch
import torch.nn as nn
from torch_geometric.data import DataLoader
import torch.nn.functional as F
import argparse

from smiles2topology import *
from model import MYMODEL


