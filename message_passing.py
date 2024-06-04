import networkx as nx
from utils import *
import math

# Message Passing framework code
# Message propagation on the network up to $hops number of hops
# given the input matrix
# the message passing layer model each row as a directed graph
# built up from the input row and the given human protein functional network stored in ppi_path
# save the inferred profile to outpath
def propagate(inpath, ppi_path, outpath, hops=1):
    df = pd.read_csv(inpath, header=None).values
    n_pcg = df.shape[1]
    pcgs = list(range(n_pcg))# pcg indexes
    hppi = pd.read_csv(ppi_path).values.tolist()
    hppi = [[int(item[0]), int(item[1])] for item in hppi]# hppi edgelist
    pcg_view = df.transpose()# each pcg is a row now
    print('pcg view shape: ', pcg_view.shape)# n_pcg x n_mirna or n_pcg x n_disease
    pcg_view = pcg_view.tolist()# each row item in the list is the pcg associated value for all mirna/disease

    G = nx.DiGraph()# construct a directed graph from the human protein functional network
    G.add_nodes_from(pcgs)
    G.add_edges_from(hppi)
    print('Finish loading graph...')

    prev_inf_vals = list(pcg_view)
    # Iterate for $hops times
    for i in range(hops):
        current_inf_vals = list()# store inferred values up to the current iteration
        # process each pcg one by one
        for p in pcgs:
            parents = G.predecessors(p)
            ori_vals = pcg_view[p]# original input value
            p_current_vals = list(prev_inf_vals[p])# inferred value from the previous iteration
            in_p = G.in_degree(p) # current node indegree
            for par in parents:
                out_par = G.out_degree(par)# parent node out degree
                parent_vals = prev_inf_vals[par]
                p_current_vals = [item + wj / math.sqrt(in_p * out_par) if ori == 0 else item for wj, item, ori in zip(parent_vals, p_current_vals, ori_vals)]# only update for nodes with zero weights
            current_inf_vals.append(p_current_vals)# add in the inferred profile matrix
        prev_inf_vals = current_inf_vals
    arr = np.array(prev_inf_vals).transpose()
    print('shape of the updated vals: ', arr.shape)
    df = pd.DataFrame(arr)
    df.to_csv(outpath, index=False, header=False)

if __name__ == "__main__":
    dir = 'data/'
    hppi_path = dir + 'hppi.csv'
    mirna_pcg = dir + 'mirna_pcg.csv'
    disease_pcg = dir + 'disease_pcg.csv'

    # one hop message propagation
    propagate(mirna_pcg, hppi_path, dir + 'rmirna_pcg1.csv')
    propagate(disease_pcg, hppi_path, dir + 'rdisease_pcg1.csv')
    print('Finish one hop message propagation')

    # Two hops propagation
    propagate(mirna_pcg, hppi_path, dir + 'rmirna_pcg2.csv', hops=2)
    propagate(disease_pcg, hppi_path, dir + 'rdisease_pcg2.csv', hops=2)
    print('Finish two hops message propagation')

    # Ten hops propagation
    propagate(mirna_pcg, hppi_path, dir + 'rmirna_pcg10.csv', hops=10)
    propagate(disease_pcg, hppi_path, dir + 'rdisease_pcg10.csv', hops=10)
    print('Finish ten hops message propagation')

