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
    def forward(self, memb, demb, pemb, mirna_edgelist, mirna_edgeweight, disease_edge_list, disease_edgeweight, pcg_edge_list, pcg_edgeweight, mirna_pcg_pairs, disease_pcg_pairs, mirna_disease_pairs):
        mhid = self.mgcn(memb, mirna_edgelist.t(), mirna_edgeweight)
        dhid = self.dgcn(demb, disease_edge_list.t(), disease_edgeweight)
        phid = self.pgcn(pemb, pcg_edge_list.t(), pcg_edgeweight)

        mhid = self.relu(mhid)
        dhid = self.relu(dhid)
        phid = self.relu(phid)

        mirna_vec = mhid[mirna_disease_pairs[:,0]]
        disease_vec = dhid[mirna_disease_pairs[:, 1]]
        assoc_vec = mirna_vec * disease_vec

        vec1 = mhid[mirna_pcg_pairs[:,0]]
        vec2 = phid[mirna_pcg_pairs[:, 1]]
        mp_vec = vec1 * vec2

        vec1 = dhid[disease_pcg_pairs[:,0]]
        vec2 = phid[disease_pcg_pairs[:, 1]]
        dp_vec = vec1 * vec2

        assoc_out = self.sigmoid(self.assoc_clf(assoc_vec))
        mirna_pcg_out = self.sigmoid(self.mirna_pcg(mp_vec))
        disease_pcg_out = self.sigmoid(self.disease_pcg(dp_vec))

        return assoc_out.squeeze(), mirna_pcg_out.squeeze(), disease_pcg_out.squeeze()