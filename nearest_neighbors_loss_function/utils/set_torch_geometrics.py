from torch_geometric.data.data import DataEdgeAttr
from torch_geometric.data import Data, HeteroData
from torch_geometric.data.data import DataEdgeAttr
import torch
from torch_geometric.data.storage import (
    GlobalStorage,
    NodeStorage,
    EdgeStorage,
)
from torch_geometric.data.data import (
    DataTensorAttr,
    DataEdgeAttr,
)
import torch

torch.serialization.add_safe_globals([
    Data,
    HeteroData,
    GlobalStorage,
    NodeStorage,
    EdgeStorage,
    DataTensorAttr,
    DataEdgeAttr,
])