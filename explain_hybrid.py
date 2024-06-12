import torch
from torch_geometric.explain import GNNExplainer, ExplainerConfig, ModelConfig, Explainer
from model import MuCoMiD, MuCoMiDWrapper  # Import your model definition
import matplotlib.pyplot as plt

from utils import *


def visualize_feature_importance(explanation, node_idx, top_k=50):
    path = f"explanations/top50_features_node{node_idx}.png"
    explanation.visualize_feature_importance(path, top_k=top_k)
    print(f"Feature importance plot for node {node_idx} has been saved to '{path}'")


def explain_model(model, data, device):
    model.eval()

    additional_data = {
        "mirna_pcg_pairs": data["mirna_pcg_pairs"],
        "disease_pcg_pairs": data["disease_pcg_pairs"],
        "mirna_edgeweight": data["mirna_edgeweight"],
        "disease_edgeweight": data["disease_edgeweight"],
        "ppi_edgeweight": data["ppi_edgeweight"],
        "train_tensor": data["train_tensor"]
    }

    wrapper_model = MuCoMiDWrapper(model, additional_data)

    wrapper_model.model.num_mirna_nodes = data["mirna_emb"].size(0)
    wrapper_model.model.num_disease_nodes = data["disease_emb"].size(0)
    wrapper_model.model.num_pcg_nodes = data["pcg_emb"].size(0)

    wrapper_model.model.num_mirna_edges = data["mirna_edgelist"].size(0)
    wrapper_model.model.num_disease_edges = data["disease_edgelist"].size(0)
    wrapper_model.model.num_pcg_edges = data["pcg_emb"].size(0)

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

    # Choose a node indexes to explain
    ind = [0]

    # Prepare the input data as required by GNNExplainer
    x = torch.cat((data["mirna_emb"], data["disease_emb"], data["pcg_emb"]), 0)
    edge_index = torch.cat((data["mirna_edgelist"], data["disease_edgelist"], data["ppi_edgelist"]), 0)
    edge_weight = torch.cat((data["mirna_edgeweight"], data["disease_edgeweight"], data["ppi_edgeweight"]), 0)

    explanations = []
    gnn_explainer_feature_masks = []
    for node_idx in ind:
        explanation = explainer(x, edge_index, index=node_idx)
        # Access the 'node_mask' attribute directly
        feature_mask = explanation.node_mask
        # Sum up the feature mask across all nodes to get the feature importance scores
        feature_importance = feature_mask.sum(dim=0)
        gnn_explainer_feature_masks.append(feature_importance)
        explanations.append(explanation)
        visualize_feature_importance(explanation, node_idx)

    return explanations, gnn_explainer_feature_masks


def main():
    device = get_device()
    data = load_data(device)

    # Initialize the model
    model = MuCoMiD(32, 32).to(device)
    model.load_state_dict(load_hybrid_model())

    # Explain the model
    _, _ = explain_model(model, data, device)


if __name__ == "__main__":
    main()