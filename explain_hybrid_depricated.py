import torch
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from torch_geometric.explain.algorithm.gnn_explainer import GNNExplainer_

# model = GCN model
from model import MuCoMiD, MuCoMiDWrapper
from utils import *


def explain_model(model, data, device, ind):
    model.eval()

    wrapper_model = MuCoMiDWrapper(model, data)
    x = data["mirna_emb"]
    edge_index = data["mirna_edgelist"].t()
    explainer = GNNExplainer_(wrapper_model, epochs=100)

    explanations_gnn_explainer = []
    for i in ind:
        explanation = explainer.explain_node(i, x, edge_index)
        explanations_gnn_explainer.append(explanation)

    # node 1 explanation is a tuple: [(num. features), (num. edges)]
    explanations_gnn_explainer[0][0].shape, explanations_gnn_explainer[0][1].shape   # (num. features),  (num. edges)

    ################################
    #add your code here to visualize the feature importance
    # aggregate the feature importance of the 10 nodes
    feature_importance = torch.zeros(x.shape[1])
    for explanation in explanations_gnn_explainer:
        feature_importance += explanation[0].squeeze()

    plt.bar(range(x.shape[1]), feature_importance)
    plt.xlabel('Feature')
    plt.ylabel('Importance')
    plt.title('Feature Importance')
    plt.show()

    top_k = 32
    top_k_indices = feature_importance.argsort(descending=True)[:top_k]
    plt.bar(range(top_k), feature_importance[top_k_indices])
    plt.xlabel('Feature')
    plt.ylabel('Importance')
    plt.title('Top 50 Feature Importance')
    plt.show()

    ########################
    edge_importance = torch.zeros(edge_index.shape[1])
    for explanation in explanations_gnn_explainer:
        edge_importance += explanation[1].squeeze()

    # Plotting total edge importance
    plt.bar(range(len(edge_importance)), edge_importance)
    plt.xlabel('Edge')
    plt.ylabel('Importance')
    plt.title('Edge Importance')
    plt.show()


    top_k_indices = edge_importance.argsort(descending=True)[:top_k]
    plt.bar(range(top_k), edge_importance[top_k_indices])
    plt.xlabel('Edge')
    plt.ylabel('Importance')
    plt.title('Top 50 Edge Importance')
    plt.show()


def main():
    device = get_device()
    data = load_data(device)

    # Initialize the model
    model = MuCoMiD(32, 32).to(device)
    model.load_state_dict(load_hybrid_model())

    # Explain the model
    explain_model(model, data, device, ind=[1,2,3])


if __name__ == "__main__":
    main()