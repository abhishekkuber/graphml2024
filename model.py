#this file defines the model

import torch
import torch as t
from torch import nn
import numpy as np
from torch_geometric.nn import conv

class MuCoMiD(nn.Module):
    def __init__(self, emb_size, hidden_dim):
        super(MuCoMiD, self).__init__()

        self.emb_size = emb_size
        self.hidden_dim = hidden_dim

        self.mgcn = conv.GCNConv(emb_size, hidden_dim)
        self.dgcn = conv.GCNConv(emb_size, hidden_dim)
        self.pgcn = conv.GCNConv(emb_size, hidden_dim)
        self.m1 = nn.Linear(hidden_dim, hidden_dim)
        self.d1 = nn.Linear(hidden_dim, hidden_dim)
        self.p1 = nn.Linear(hidden_dim, hidden_dim)
        self.mirna_pcg = nn.Linear(hidden_dim, 1)
        self.disease_pcg = nn.Linear(hidden_dim, 1)
        self.assoc_clf = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    # order in the global network: virus -> human -> go
    def forward(self, data, label_tensor):

        mhid = self.mgcn(data["mirna_emb"], data["mirna_edgelist"].t(), data["mirna_edgeweight"])
        dhid = self.dgcn(data["disease_emb"], data["disease_edgelist"].t(), data["disease_edgeweight"])
        phid = self.pgcn(data["pcg_emb"], data["ppi_edgelist"].t(), data["ppi_edgeweight"])

        mhid = self.relu(mhid)
        dhid = self.relu(dhid)
        phid = self.relu(phid)

        mirna_vec = mhid[label_tensor[:,0]]
        disease_vec = dhid[label_tensor[:, 1]]
        assoc_vec = mirna_vec * disease_vec

        vec1 = mhid[data["mirna_pcg_pairs"][:,0]]
        vec2 = phid[data["mirna_pcg_pairs"][:, 1]]
        mp_vec = vec1 * vec2

        vec1 = dhid[data["disease_pcg_pairs"][:,0]]
        vec2 = phid[data["disease_pcg_pairs"][:, 1]]
        dp_vec = vec1 * vec2

        assoc_out = self.sigmoid(self.assoc_clf(assoc_vec))
        mirna_pcg_out = self.sigmoid(self.mirna_pcg(mp_vec))
        disease_pcg_out = self.sigmoid(self.disease_pcg(dp_vec))

        return assoc_out.squeeze(), mirna_pcg_out.squeeze(), disease_pcg_out.squeeze()


class MuCoMiDWrapper(nn.Module):
    def __init__(self, model, additional_data):
        super(MuCoMiDWrapper, self).__init__()
        self.model = model
        self.additional_data = additional_data

    def forward(self, x, edge_index, edge_weight=None):
        print(f"x: {x.shape}, edge_index: {edge_index.shape}")
        # Reconstruct the data dictionary expected by MuCoMiD
        data = {
            "mirna_emb": x[:self.model.num_mirna_nodes],
            "disease_emb": x[self.model.num_mirna_nodes:self.model.num_mirna_nodes+self.model.num_disease_nodes],
            "pcg_emb": x[-self.model.num_pcg_nodes:],

            "mirna_edgelist": edge_index[:self.model.num_mirna_edges, :],
            "disease_edgelist": edge_index[self.model.num_mirna_edges:self.model.num_mirna_edges+self.model.num_disease_edges, :],
            "ppi_edgelist": edge_index[self.model.num_mirna_edges+self.model.num_disease_edges:, :],

            # Add the additional data stored in the wrapper
            "mirna_pcg_pairs": self.additional_data["mirna_pcg_pairs"],
            "disease_pcg_pairs": self.additional_data["disease_pcg_pairs"],
            "mirna_edgeweight": self.additional_data["mirna_edgeweight"],
            "disease_edgeweight": self.additional_data["disease_edgeweight"],
            "ppi_edgeweight": self.additional_data["ppi_edgeweight"]
        }

        # Call the model with the reconstructed data dictionary
        result, _, _ = self.model(data, self.additional_data["train_tensor"])
        return result
