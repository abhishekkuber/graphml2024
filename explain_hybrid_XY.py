"""
To use the explainer from GNNExplainer, the input data must be in the following format:
x(torch.Tensor): Shape (N, F) where N is the number of nodes and F is the number of features per node.
edge_index(torch.Tensor): Shape (2, E) where E is the number of edges.
edge_weight(torch.Tensor): Shape (E,) where E is the number of edges (optional).
index (int): shape () or (I,) where I is the number of nodes to explain (optional).

data[mirna_emb]: torch.Size([1618, 32])
data[disease_emb]: torch.Size([3679, 32])
data[pcg_emb]: torch.Size([100, 32])
data[train_tensor]: torch.Size([9184, 2])
data[train_lbl_tensor]: torch.Size([9184])
data[disease_pcg_pairs]: torch.Size([351777, 2])
data[disease_pcg_weight]: torch.Size([351777])
data[mirna_pcg_pairs]: torch.Size([155345, 2])
data[mirna_pcg_weight]: torch.Size([155345])
data[mirna_edgelist]: torch.Size([3672, 2])
data[mirna_edgeweight]: torch.Size([3672])
data[disease_edgelist]: torch.Size([4776, 2])
data[disease_edgeweight]: torch.Size([4776])
data[ppi_edgelist]: torch.Size([898, 2])
data[ppi_edgeweight]: torch.Size([898])
"""

import torch
from torch_geometric.explain import GNNExplainer, ExplainerConfig, ModelConfig, Explainer
from model import MuCoMiD
import matplotlib.pyplot as plt
from torch_geometric.utils import add_remaining_self_loops
from utils import *
from torch import nn


class MuCoMiDWrapper(nn.Module):
    def __init__(self, model, data):
        super(MuCoMiDWrapper, self).__init__()
        self.model = model
        self.data = data

    def forward(self, x, edge_index, edge_weight=None):
        return None



def explain_model(model, data, device):
    model.eval()
    wrapper_model = MuCoMiDWrapper(model, data)

    # print all data and shapes
    for key in data.keys():
        print(f"data[{key}]: {data[key].shape}")

    model_config = ModelConfig(
        mode='multiclass_classification',
        task_level='node',
        return_type='log_probs'
    )

    explainer = Explainer(
        model=wrapper_model,
        algorithm=GNNExplainer(epochs=200),
        explanation_type='model',
        node_mask_type='attributes',
        edge_mask_type='object',
        model_config=model_config
    )

    # We need to predict the link between a specific miRNA \( X \) and a disease \( Y \) using GNNExplainer,
    # while focusing on understanding how the neighborhood of \( X \) in the miRNA family graph influences this prediction.

    # 1. **Select the Graph and Node of Interest**:
    #    - Focus on the subgraph involving miRNA \( X \) and its neighbors. If \( X \) is represented by a node index
    #    \( n \) in your graph, gather all edges that connect \( n \) with its neighbors.
    #    - Extract the subgraph that includes these nodes and edges.

    #2. **Prepare the Data**:
   # - **Node Features (x)**: Use the embeddings of the nodes in the subgraph.
   # - **Edge List**: Use only the edges that are part of the subgraph.
   # - **Edge Weights**: Include these if your model uses them.


    # 3. **Initialize GNNExplainer**:
    #    - Instantiate GNNExplainer with your trained model, passing the node features, edge index, and optionally edge weights that are specific to the subgraph of interest.

    # ### Step 3: Run the Explainer
    # Run the GNNExplainer for the node \( X \), attempting to explain the prediction regarding its link with disease \( Y \).
    # The explainer will try to identify which features of \( X \) and which connections (edges) in its neighborhood are most influential in predicting a link to \( Y \).
    return None, None



def main():
    device = get_device()
    data = load_data(device)

    model = MuCoMiD(32, 32).to(device)
    model.load_state_dict(load_hybrid_model())

    _, _ = explain_model(model, data, device)

if __name__ == "__main__":
    main()
