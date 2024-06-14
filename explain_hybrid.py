import torch
from torch_geometric.explain import GNNExplainer, ExplainerConfig, ModelConfig, Explainer
from model import MuCoMiD, MuCoMiDWrapper
import matplotlib.pyplot as plt
from torch_geometric.utils import add_remaining_self_loops
from utils import *



def visualize_feature_importance(explanation, node_idx, top_k=50):
    path = f"explanations/top50_features_node{node_idx}.png"
    explanation.visualize_feature_importance(path, top_k=top_k)
    print(f"Feature importance plot for node {node_idx} has been saved to '{path}'")

import torch
from torch_geometric.utils import add_remaining_self_loops

def explain_model(model, data, device):
    model.eval()
    wrapper_model = MuCoMiDWrapper(model, data)

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

    ind = [0]  # Example of using a single node index for explanation

    x = data["mirna_emb"]
    edge_index = data["mirna_edgelist"].t()
    edge_weight = data["mirna_edgeweight"]

    # Ensure correct tensor types
    edge_index = edge_index.long()
    edge_weight = edge_weight.float()

    print("Before any operations:")
    print("x.shape:", x.shape)
    print("edge_index.shape:", edge_index.shape)
    print("edge_weight.shape:", edge_weight.shape)

    # Validate that edge_index does not have invalid node indices
    num_nodes = x.size(0)
    if edge_index.max().item() >= num_nodes:
        raise ValueError("Edge index contains invalid node indices.")

    edge_index, edge_weight = add_remaining_self_loops(
        edge_index, edge_weight, fill_value=1, num_nodes=num_nodes)

    print("After adding self-loops:")
    print("edge_index.shape:", edge_index.shape)
    print("edge_weight.shape:", edge_weight.shape)

    explanations = []
    gnn_explainer_feature_masks = []
    for node_idx in ind:
        print(f"Attempting explanation for node {node_idx}:")
        print(f"shapes: x: {x.shape}, edge_index: {edge_index.shape}, edge_weight: {edge_weight.shape}")

        try:
            explanation = explainer(x, edge_index=edge_index, edge_weight=edge_weight, index=node_idx)
            feature_mask = explanation.node_mask
            feature_importance = feature_mask.sum(dim=0)
            gnn_explainer_feature_masks.append(feature_importance)
            explanations.append(explanation)
            visualize_feature_importance(explanation, node_idx)
        except Exception as e:
            print("Error during model explanation:", e)
            break  # Optional: Break on first error to focus debugging

    return explanations, gnn_explainer_feature_masks



def main():
    device = get_device()
    data = load_data(device)

    model = MuCoMiD(32, 32).to(device)
    model.load_state_dict(load_hybrid_model())

    _, _ = explain_model(model, data, device)

if __name__ == "__main__":
    main()
