import torch
from torch_geometric.explain import GNNExplainer, ExplainerConfig, ModelConfig, Explainer
from model import MuCoMiD, MuCoMiDWrapper  # Import your model definition
import matplotlib.pyplot as plt


def plot_loss(loss_history):
    plt.figure(figsize=(10, 5))
    plt.plot(loss_history, label='Training Loss')
    plt.title('Training Loss per Epoch')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

from utils import *
import os  # Import the os module

def visualize_feature_importance(explanation, node_idx, top_k=50):
    path = f"explanations/top50_features_node{node_idx}.png"
    # Ensure the directory exists before saving the file
    os.makedirs(os.path.dirname(path), exist_ok=True)
    explanation.visualize_feature_importance(path, top_k=top_k)
    print(f"Feature importance plot for node {node_idx} has been saved to '{path}'")



def explain_model(model, data, device):
    model.eval()

    wrapper_model = MuCoMiDWrapper(model, data)

    model_config = ModelConfig(
        mode='regression',
        task_level='node',
        return_type='raw'
    )

    explainer = Explainer(
        model=wrapper_model,
        algorithm=GNNExplainer(epochs=2000, lr=0.01),
        explanation_type='model',
        node_mask_type='attributes',
        edge_mask_type='object',
        model_config=model_config
    )

    # Choose a node indexes to explain
    ind = [1]

    # Prepare the input data as required by GNNExplainer
    # x = torch.cat((data["mirna_emb"], data["disease_emb"], data["pcg_emb"]), 0)
    # edge_index = torch.cat((data["mirna_edgelist"], data["disease_edgelist"], data["ppi_edgelist"]), 0)
    # edge_weight = torch.cat((data["mirna_edgeweight"], data["disease_edgeweight"], data["ppi_edgeweight"]), 0)

    # Explain the mirna embeddings, make sure to use the right settings in the wrapper too for this.
    x = data["mirna_emb"]
    edge_index = data["mirna_edgelist"].t()

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