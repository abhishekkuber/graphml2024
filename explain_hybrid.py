import torch
from torch_geometric.explain import GNNExplainer, ExplainerConfig, ModelConfig, Explainer
from model import MuCoMiD, MuCoMiDWrapper  # Import your model definition
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from utils import *
import os  # Import the os module
import seaborn as sns


def plot_loss(loss_history):
    plt.figure(figsize=(10, 5))
    plt.plot(loss_history, label='Training Loss')
    plt.title('Training Loss per Epoch')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()


def visualize_feature_importance(explanation, idx, top_k=12, task="node"):
    path = f"explanations/top{top_k}_features_{task}{idx}.png"
    # Ensure the directory exists before saving the file
    os.makedirs(os.path.dirname(path), exist_ok=True)
    explanation.visualize_feature_importance(path, top_k=top_k)
    print(f"Feature importance plot for {task} {idx} has been saved to '{path}'")


def visualize_feature_importance_heatmap(explanations, indices, top_k=12, task="node"):
    path = f"explanations/top{top_k}_features_{task}_multiple.png"
    # Ensure the directory exists before saving the file
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Get the feature importance for all indices
    feature_importances = [explanation.node_mask.sum(dim=0).cpu().numpy() for explanation in explanations]

    # Convert the list of feature importances to a numpy array
    feature_importances = np.array(feature_importances)

    # Create a heatmap
    plt.figure(figsize=(10, 10))
    sns.heatmap(feature_importances, cmap='viridis')
    plt.title(f'Feature Importance for {task} {indices}')
    plt.xlabel('Features')
    plt.ylabel(f"{task}s")

    # Save the figure
    plt.savefig(path)
    print(f"Feature importance heatmap for {task} {indices} has been saved to '{path}'")


def explain_model(model, data, device, task_level):
    model.eval()

    wrapper_model = MuCoMiDWrapper(model, data)

    model_config = ModelConfig(
        mode='regression',
        task_level=task_level,
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

    # Choose a node or edge index to explain
    ind = [1, 2, 3]

    # Explain the mirna embeddings, make sure to use the right settings in the wrapper too for this.
    x = data["mirna_emb"]
    edge_index = data["mirna_edgelist"].t()

    explanations = []
    gnn_explainer_feature_masks = []
    for idx in ind:
        explanation = explainer(x, edge_index, index=idx)
        # Access the 'node_mask' attribute directly
        feature_mask = explanation.node_mask
        # Sum up the feature mask across all nodes to get the feature importance scores
        feature_importance = feature_mask.sum(dim=0)
        gnn_explainer_feature_masks.append(feature_importance)
        explanations.append(explanation)
        visualize_feature_importance(explanation, idx, task=task_level)

    visualize_feature_importance_heatmap(explanations, ind, task=task_level)

    return explanations, gnn_explainer_feature_masks


def main():
    device = get_device()
    data = load_data(device)

    # Initialize the model
    model = MuCoMiD(32, 32).to(device)
    model.load_state_dict(load_hybrid_model())

    # Explain the model
    _, _ = explain_model(model, data, device, task_level='node')
    _, _ = explain_model(model, data, device, task_level='edge')


if __name__ == "__main__":
    main()
