#this file evaluates the performance of the hybrid model (mpm+cmtt)
from matplotlib import pyplot as plt

from model import *
from utils import *
import torch as t
import pandas as pd

from utils import *
import numpy as np
import sdne as sdne

def read_int_data(pos_path, neg_path):
    pos_df = pd.read_csv(pos_path).values.tolist()
    neg_df = pd.read_csv(neg_path).values.tolist()
    int_edges = pos_df + neg_df
    int_lbl = [1] * len(pos_df) + [0] * len(neg_df)
    return pos_df, t.LongTensor(int_edges), t.FloatTensor(int_lbl)

# To understand about Pareto optimality, this reddit post helped 
# https://www.reddit.com/r/statistics/comments/ogq7p3/d_understanding_pareto_optimality/

def objectives(loss0, loss1, loss2):
        return np.array([loss0.item(), loss1.item(), loss2.item()])

# A set of scores s1 is said to Pareto dominate another set s2 if all
# elements of s1 are less than or equal to the corresponding elements of s2, 
# and at least one element of s1 is strictly less than the corresponding element of s2.
def check_pareto_dom(s1, s2):
    if np.all(s1 <= s2) and np.any(s1 < s2):
        return True
    return False

def get_pareto_front(losses, weights, num_samples=100):
    # Randomly generate a set of weights from a uniform distribution
    solutions = np.random.uniform(size=(num_samples, len(weights)))
    pareto_front = []
    for solution in solutions:
        # For each solution, calculate the weighted loss
        scores = np.dot(losses, solution)
        
        # Check if each solution is Pareto dominated by any other solution.
        # The solutions that are not Pareto dominated by any other form the Pareto front, representing the optimal trade-offs among the objectives.
        if not any(check_pareto_dom(scores, np.dot(losses, other)) for other in solutions):
            pareto_front.append((solution, scores))

    return pareto_front

def get_embedding(vectors: dict, x):
    matrix = np.zeros((
        x,
        len(list(vectors.values())[0])
    ))
    for key, value in vectors.items():
        matrix[int(key), :] = value
    return matrix

def Get_embedding_Matrix(mirna_disease, mirna_pcg, disease_pcg, pcg_pcg, mirna_mirna, disease_disease, n_mirna, n_disease, n_pcg):
    # mirna, disease and pcgs can have the same identifiers. So, we need to prefix them with their type to differentiate them
    md = [[f'm_{item[0]}', f'd_{item[1]}'] for item in mirna_disease]
    mp = [[f'm_{item[0]}', f'p_{item[1]}'] for item in mirna_pcg]
    dp = [[f'd_{item[0]}', f'p_{item[1]}'] for item in disease_pcg]
    pp = [[f'p_{item[0]}', f'p_{item[1]}'] for item in pcg_pcg]
    mm = [[f'm_{item[0]}', f'm_{item[1]}'] for item in mirna_mirna]
    dd = [[f'd_{item[0]}', f'd_{item[1]}'] for item in disease_disease]

    # split the graph into 2 subgraphs, one for training and one for loss calculation.
    # both subgraphs are half the size of the original graph
    # the graph is partitioned such that it remains connected
    from networkx import Graph
    from networkx.algorithms import community

    # start with splitting the mirna_pcg graph into 2 subgraphs. Then add the other edges to the subgraphs accordingly
    mirna_pcg_graph = Graph()
    mirna_pcg_graph.add_edges_from(mp)

    # split into 2 equally sized subgraphs
    mirna_pcg_subgraphs = community.kernighan_lin_bisection(mirna_pcg_graph)
    g1 = mirna_pcg_graph.subgraph(mirna_pcg_subgraphs[0])
    g2 = mirna_pcg_graph.subgraph(mirna_pcg_subgraphs[1])  # 800 nodes in each subgraph

    # unfreeze the subgraphs
    g1 = Graph(g1)
    g2 = Graph(g2)

    # add the edges to the subgraphs
    g1.add_edges_from(md)
    g1.add_edges_from(dp)
    g1.add_edges_from(pp)
    g1.add_edges_from(mm)
    g1.add_edges_from(dd)

    g2.add_edges_from(md)
    g2.add_edges_from(dp)
    g2.add_edges_from(pp)
    g2.add_edges_from(mm)
    g2.add_edges_from(dd)

    # get the training and loss edges
    n_mirna_train = len(g1.nodes)
    n_disease_train = len(g1.nodes)
    n_pcg_train = len(g1.nodes)
    mm_train = [[int(item[0].split('_')[1]), int(item[1].split('_')[1])] for item in g1.edges if 'm_' in item[0] and 'm_' in item[1]]
    dd_train = [[int(item[0].split('_')[1]), int(item[1].split('_')[1])] for item in g1.edges if 'd_' in item[0] and 'd_' in item[1]]
    mp_train = [[int(item[0].split('_')[1]), int(item[1].split('_')[1])] for item in g1.edges if 'm_' in item[0] and 'p_' in item[1]]
    dp_train = [[int(item[0].split('_')[1]), int(item[1].split('_')[1])] for item in g1.edges if 'd_' in item[0] and 'p_' in item[1]]
    pp_train = [[int(item[0].split('_')[1]), int(item[1].split('_')[1])] for item in g1.edges if 'p_' in item[0] and 'p_' in item[1]]
    md_train = [[int(item[0].split('_')[1]), int(item[1].split('_')[1])] for item in g1.edges if 'm_' in item[0] and 'd_' in item[1]]

    n_mirna_loss = len(g2.nodes)
    n_disease_loss = len(g2.nodes)
    n_pcg_loss = len(g2.nodes)
    mm_loss = [[int(item[0].split('_')[1]), int(item[1].split('_')[1])] for item in g2.edges if 'm_' in item[0] and 'm_' in item[1]]
    dd_loss = [[int(item[0].split('_')[1]), int(item[1].split('_')[1])] for item in g2.edges if 'd_' in item[0] and 'd_' in item[1]]
    mp_loss = [[int(item[0].split('_')[1]), int(item[1].split('_')[1])] for item in g2.edges if 'm_' in item[0] and 'p_' in item[1]]
    dp_loss = [[int(item[0].split('_')[1]), int(item[1].split('_')[1])] for item in g2.edges if 'd_' in item[0] and 'p_' in item[1]]
    pp_loss = [[int(item[0].split('_')[1]), int(item[1].split('_')[1])] for item in g2.edges if 'p_' in item[0] and 'p_' in item[1]]
    md_loss = [[int(item[0].split('_')[1]), int(item[1].split('_')[1])] for item in g2.edges if 'm_' in item[0] and 'd_' in item[1]]

    print("sizes of partitioned graphs: ", len(g1.nodes), len(g2.nodes))  # 4855 and 4872

    np.savetxt(f"./split_edges/mirna_mirna_split_edges_loss_connected.csv", mm_loss, delimiter=",")
    np.savetxt(f"./split_edges/disease_disease_split_edges_loss_connected.csv", dd_loss, delimiter=",")
    np.savetxt(f"./split_edges/pcg_pcg_split_edges_loss_connected.csv", pp_loss, delimiter=",")
    np.savetxt(f"./split_edges/mirna_pcg_split_edges_loss_connected.csv", mp_loss, delimiter=",")
    np.savetxt(f"./split_edges/disease_pcg_split_edges_loss_connected.csv", dp_loss, delimiter=",")
    # np.savetxt(f"./split_edges/mirna_disease_split_edges_loss_connected.csv", md_loss, delimiter=",")

    graph1 = sdne.Graph()
    graph1.add_edgelist(md_train, mp_train, dp_train, pp_train, mm_train, dd_train, n_mirna_train, n_disease_train, n_pcg_train)
    model = sdne.SDNE(graph1, [1000, 32]) #32 is the number of features per node
    return get_embedding(model.vectors, n_mirna_train + n_disease_train + n_pcg_train), n_mirna_train, n_disease_train, n_pcg_train

def embed():
    if not os.path.exists('./split_edges'): os.makedirs('./split_edges')

    # select 100 most important PCGs
    fs_df = pd.read_csv('../data/original_data/relieff_raw_pcg.csv', header=None).values.tolist()
    all_selected_feats = [item[0] for item in fs_df]
    selected_feats = all_selected_feats[:100]

    # get and process miRNA-miRNA relationship
    mirna_mirna = pd.read_csv('../data/original_data/mirna_fam_pos.csv').values.tolist()

    # get and process disease-disease relationship
    disease_disease = pd.read_csv('../data/original_data/disease_onto_pos.csv').values.tolist()

    # get and process PCG-PCG relationship
    ppi_mat = pd.read_csv('../data/original_data/hppi.csv')
    ppi_mat = ppi_mat[(ppi_mat['hprot1'].isin(selected_feats)) & (ppi_mat['hprot2'].isin(selected_feats))]
    pcg_pcg = [[selected_feats.index(int(item[0])), selected_feats.index(int(item[1]))] for item in ppi_mat.values.tolist()]

    # get and process miRNA-PCG relationship
    mirna_pcg_mat = pd.read_csv('../data/original_data/rmirna_pcg10.csv', header=None)
    mirna_pcg_mat = pd.DataFrame(mirna_pcg_mat.iloc[:, selected_feats]).values.tolist()
    mirna_pcg = list()
    for i, rec in enumerate(mirna_pcg_mat):
        for j, item in enumerate(rec):
            if item != 0:
                mirna_pcg.append([i, j])

    # get and process disease-PCG relationship
    disease_pcg_mat = pd.read_csv('../data/original_data/rdisease_pcg10.csv', header=None)
    disease_pcg_mat = pd.DataFrame(disease_pcg_mat.iloc[:, selected_feats]).values.tolist()
    disease_pcg = list()
    for i, rec in enumerate(disease_pcg_mat):
        for j, item in enumerate(rec):
            if item != 0:
                disease_pcg.append([i, j])

    # get and process miRNA-disease relationship
    mirna_diease = pd.read_csv('../data/training_data/hmdd2_pos.csv').values.tolist()

    # get the number of miRNA, disease and PCG
    n_mirna = len(mirna_pcg_mat)
    n_disease = len(disease_pcg_mat)
    n_pcg = 100

    # generate the embeddings
    # embeddings = Get_embedding_Matrix(mirna_diease, mirna_pcg, disease_pcg, pcg_pcg, None, None, n_mirna, n_disease, n_pcg)
    embeddings, n_mirna, n_disease, n_pcg = Get_embedding_Matrix(mirna_diease, mirna_pcg, disease_pcg, pcg_pcg, mirna_mirna, disease_disease, n_mirna, n_disease,
                                      n_pcg)  # also pass miRNA-miRNA and disease-disease relationships

    mirna_emb = np.array(embeddings[0:n_mirna, 0:])
    disease_emb = np.array(embeddings[n_mirna:n_mirna + n_disease, 0:])
    pcg_emb = np.array(embeddings[n_mirna + n_disease:n_mirna + n_disease + n_pcg, 0:])

    # save the embeddings into csv files
    np.savetxt(f"./split_edges/mirna_emb_split_edges_connected.csv", mirna_emb, delimiter=",")
    np.savetxt(f"./split_edges/disease_emb_split_edges_connected.csv", disease_emb, delimiter=",")
    np.savetxt(f"./split_edges/pcg_emb_split_edges_connected.csv", pcg_emb, delimiter=",")

def mpm_hybrid():
    ## read data from files

    # embeddings
    mirna = pd.read_csv(f'./split_edges/mirna_emb_split_edges_connected.csv',
                        header=None).values.tolist()
    mirna_emb = t.FloatTensor(mirna)
    disease = pd.read_csv(f'./split_edges/disease_emb_split_edges_connected.csv',
                          header=None).values.tolist()
    disease_emb = t.FloatTensor(disease)
    pcg = pd.read_csv(f'./split_edges/pcg_emb_split_edges_connected.csv',
                      header=None).values.tolist()
    pcg_emb = t.FloatTensor(pcg)

    # training data
    _, train_tensor, train_lbl_tensor = read_int_data(
        '../data/training_data/hmdd2_pos.csv',
        '../data/training_data/hmdd2_neg1_0.csv')

    # others
    disease_onto = pd.read_csv(f'./split_edges/disease_disease_split_edges_loss_connected.csv', header=None).values.tolist()
    disease_pcg = pd.read_csv(f'./split_edges/disease_pcg_split_edges_loss_connected.csv', header=None).values.tolist()
    hppi = pd.read_csv(f'./split_edges/pcg_pcg_split_edges_loss_connected.csv', header=None).values.tolist()
    mirna_family = pd.read_csv(f'./split_edges/mirna_mirna_split_edges_loss_connected.csv', header=None).values.tolist()
    mirna_pcg = pd.read_csv(f'./split_edges/mirna_pcg_split_edges_loss_connected.csv',header=None).values.tolist()

    disease_pcg_pairs = list()
    disease_pcg_weight = list()
    mirna_pcg_pairs = list()
    mirna_pcg_weight = list()
    mirna_edgelist = list()
    mirna_edgeweight = list()
    disease_edgelist = list()
    disease_edgeweight = list()
    ppi_edgelist = list()
    ppi_edgeweight = list()

    for p in mirna_family:  # 2 mirnas are connected if they belong to the same family
        idx1 = p[0]
        idx2 = p[1]
        mirna_edgelist.append([idx1, idx2])
        mirna_edgelist.append([idx2, idx1])
        mirna_edgeweight.append(1)
        mirna_edgeweight.append(1)

    for p in disease_onto:  # [children, parent] disease pairs
        idx1 = p[0]
        idx2 = p[1]
        disease_edgelist.append([idx1, idx2])
        disease_edgeweight.append(1)

    for p in hppi:  # [hprot1,hprot2,score] (human protein-protein interaction)
        idx1 = int(p[0])
        idx2 = int(p[1])
        ppi_edgelist.append([idx1, idx2])
        ppi_edgeweight.append(1)
        ppi_edgelist.append([idx2, idx1])
        ppi_edgeweight.append(1)

    for p in mirna_pcg:  # [mirna,pcg,score]
        idx1 = p[0]
        idx2 = p[1]
        mirna_pcg_pairs.append([idx1, idx2])
        mirna_pcg_weight.append(1)

    for p in disease_pcg:  # [disease,pcg,score]
        idx1 = p[0]
        idx2 = p[1]
        disease_pcg_pairs.append([idx1, idx2])
        disease_pcg_weight.append(1)

    disease_pcg_pairs = t.LongTensor(disease_pcg_pairs)
    disease_pcg_weight = t.FloatTensor(disease_pcg_weight)
    mirna_pcg_pairs = t.LongTensor(mirna_pcg_pairs)
    mirna_pcg_weight = t.FloatTensor(mirna_pcg_weight)
    mirna_edgelist = t.LongTensor(mirna_edgelist)
    mirna_edgeweight = t.FloatTensor(mirna_edgeweight)
    disease_edgelist = t.LongTensor(disease_edgelist)
    disease_edgeweight = t.FloatTensor(disease_edgeweight)
    ppi_edgelist = t.LongTensor(ppi_edgelist)
    ppi_edgeweight = t.FloatTensor(ppi_edgeweight)

    criterion = t.nn.BCELoss()
    l1loss = t.nn.MSELoss()

    # Check for CUDA availability
    device = t.device('cuda' if t.cuda.is_available() else 'cpu')

    # Move tensors to device (GPU or CPU)
    mirna_emb = mirna_emb.to(device)
    disease_emb = disease_emb.to(device)
    pcg_emb = pcg_emb.to(device)
    mirna_edgelist = mirna_edgelist.to(device)
    mirna_edgeweight = mirna_edgeweight.to(device)
    disease_edgelist = disease_edgelist.to(device)
    disease_edgeweight = disease_edgeweight.to(device)
    ppi_edgelist = ppi_edgelist.to(device)
    ppi_edgeweight = ppi_edgeweight.to(device)
    mirna_pcg_pairs = mirna_pcg_pairs.to(device)
    disease_pcg_pairs = disease_pcg_pairs.to(device)
    train_tensor = train_tensor.to(device)
    train_lbl_tensor = train_lbl_tensor.to(device)
    mirna_pcg_weight = mirna_pcg_weight.to(device)
    disease_pcg_weight = disease_pcg_weight.to(device)
    criterion = criterion.to(device)
    l1loss = l1loss.to(device)

    pos_test_path = '../data/test_data/new_mirna_pos.csv'  # change this for a different test dataset
    neg_test_pre = pos_test_path.replace('_pos.csv', '_neg.csv')
    datasrc = pos_test_path[pos_test_path.rfind('/') + 1:].replace('_pos.csv', '')

    def eval(model, neg_rates, datasrc=datasrc, print_results=True):
        ## evaluate the model
        all_scores = list()

        avg_acc = 0
        avg_loss = 0

        for testrate in neg_rates:
            avg_acc = 0
            avg_loss = 0

            for testset in range(10):
                cur_score = [datasrc, str(testrate), str(testset), str(1)]
                neg_test_path = neg_test_pre.replace('.csv', str(testrate) + '_' + str(testset) + '.csv')
                _, test_tensor, test_lbl = read_int_data(pos_test_path, neg_test_path)
                datasrc = pos_test_path[pos_test_path.rfind('/') + 1:].replace('.csv', '')
                test_tensor = test_tensor.to(device)
                assoc_out, mirna_pcg_out, disease_pcg_out = model(mirna_emb, disease_emb, pcg_emb, mirna_edgelist,
                                                                  mirna_edgeweight, disease_edgelist, disease_edgeweight,
                                                                  ppi_edgelist, ppi_edgeweight, mirna_pcg_pairs, disease_pcg_pairs, test_tensor)

                auc_score, ap_score, sn, sp, acc, prec, rec, f1, mcc = get_all_score(test_lbl, assoc_out.detach().cpu().numpy())
                tmp_score = [auc_score, ap_score, sn, sp, acc, prec, rec, f1]
                for ftmp in tmp_score:
                    cur_score.append(str(round(ftmp, 5)))
                if print_results: print(cur_score)
                all_scores.append(cur_score)

                # #############
                # Loss and acc
                avg_acc += acc

                tloss0, tloss1, tloss2 = criterion(assoc_out, test_lbl), l1loss(mirna_pcg_out, mirna_pcg_weight), l1loss(disease_pcg_out,
                                                                                                                         disease_pcg_weight)
                tloss = w1 * tloss0 + w2 * tloss1 + w3 * tloss2
                test_loss = tloss.item()
                avg_loss += test_loss

            avg_acc /= 10
            avg_loss /= 10

        return all_scores, avg_acc, avg_loss

    # ############################
    ## train the model
    w1 = 1.0
    w2 = 1.0
    w3 = 1.0
    model = MuCoMiD(32, 32).to(device)
    optimizer = t.optim.Adam(model.parameters(), lr=0.001)

    train_losses, test_losses, test_accuracies = list(), list(), list()

    for epoch in range(0, 200):
        model.train()
        model.zero_grad()
        assoc_out, mirna_pcg_out, disease_pcg_out = model(mirna_emb,
                                                          disease_emb, pcg_emb,
                                                          mirna_edgelist,
                                                          mirna_edgeweight,
                                                          disease_edgelist,
                                                          disease_edgeweight,
                                                          ppi_edgelist,
                                                          ppi_edgeweight,
                                                          mirna_pcg_pairs,
                                                          disease_pcg_pairs,
                                                          train_tensor)
        loss0 = criterion(assoc_out, train_lbl_tensor)
        loss1 = l1loss(mirna_pcg_out, mirna_pcg_weight)
        loss2 = l1loss(disease_pcg_out, disease_pcg_weight)
        loss = w1 * loss0 + w2 * loss1 + w3 * loss2
        
        # l1 = loss0.item()
        # l2 = loss1.item()
        # l3 = loss2.item()
        # w1 = 1
        # w2 = l3 / (l1 + l2 + l3 + 1e-10)
        # w3 = l2 / (l1 + l2 + l3 + 1e-10)

        current_objectives = objectives(loss0, loss1, loss2)
        pareto_front = get_pareto_front(current_objectives, np.array([w1, w2, w3]))
        w1 = pareto_front[0][0][0]
        w2 = pareto_front[0][0][1]
        w3 = pareto_front[0][0][2]

        loss.backward()
        optimizer.step()
        loss_val = loss.item()

        print('Epoch: ', epoch, ' loss: ', loss_val / train_lbl_tensor.size(0))

        if epoch % 20 == 0:
            train_losses.append(loss_val)

            all_scores, avg_acc, avg_loss = eval(model, neg_rates=[10], print_results=False)
            test_losses.append(avg_loss)
            test_accuracies.append(avg_acc)

            print('Test loss: ', avg_loss, ' Test acc: ', avg_acc)


    # Plot the losses and test accuracy
    if not os.path.exists('plots/pareto'): os.makedirs('plots/pareto')
    plt.figure()
    plt.plot(train_losses, label='Training loss')
    plt.plot(test_losses, label='Test loss')
    plt.legend()
    plt.title('Losses')
    plt.savefig('plots/pareto/pareto_loss_curves.png')

    plt.figure()
    plt.plot(test_accuracies, label='Test accuracy')
    plt.legend()
    plt.title('Test Accuracy')
    plt.savefig('plots/pareto/pareto_test_acc.png')

    np.save('plots/pareto/pareto_train_losses.npy', train_losses)
    np.save('plots/pareto/pareto_test_losses.npy', test_losses)
    np.save('plots/pareto/pareto_test_accuracies.npy', test_accuracies)

    ## evaluate the model
    all_scores, _, _ = eval(model, neg_rates=[1,5,10])

    ## write the results into file
    writer = open(f'./results/hybrid_output_split_edges_connected.txt', 'w+')
    writer.write('data,run,auc_score, ap_score, sn, sp, acc, prec, rec, f1\n')
    for line in all_scores:
        writer.write(','.join(line))
        writer.write('\n')
    writer.close()


if __name__ == "__main__":
    embed()
    mpm_hybrid()
