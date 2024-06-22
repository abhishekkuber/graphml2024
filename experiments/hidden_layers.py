#this file evaluates the performance of the hybrid model (mpm+cmtt)
from matplotlib import pyplot as plt

from model import *
from utils import *
import torch as t
import pandas as pd

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

class MuCoMiD_experiment(nn.Module):
    def __init__(self, emb_size, hidden_layers, edge_operation=lambda x, y: x * y, edge_operation_name='other'):
        super(MuCoMiD_experiment, self).__init__()

        self.emb_size = emb_size
        # self.hidden_dim = hidden_dim
        # self.mgcn = conv.GCNConv(emb_size, hidden_dim)
        # self.dgcn = conv.GCNConv(emb_size, hidden_dim)
        # self.pgcn = conv.GCNConv(emb_size, hidden_dim)

        self.mgcn = nn.ModuleList(
            [conv.GCNConv(emb_size if i == 0 else hidden_layers[i - 1], hidden_layers[i]) for i in range(len(hidden_layers))])
        self.dgcn = nn.ModuleList(
            [conv.GCNConv(emb_size if i == 0 else hidden_layers[i - 1], hidden_layers[i]) for i in range(len(hidden_layers))])
        self.pgcn = nn.ModuleList(
            [conv.GCNConv(emb_size if i == 0 else hidden_layers[i - 1], hidden_layers[i]) for i in range(len(hidden_layers))])

        hidden_dim = hidden_layers[-1]
        # if concatenation operation, hidden_dim = hidden_dim * 2:
        if edge_operation_name == 'concat':
            hidden_dim = hidden_dim * 2

        self.mirna_pcg = nn.Linear(hidden_dim, 1)
        self.disease_pcg = nn.Linear(hidden_dim, 1)
        self.assoc_clf = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

        self.edge_operation = edge_operation


    # order in the global network: virus -> human -> go
    def forward(self, memb, demb, pemb, mirna_edgelist, mirna_edgeweight, disease_edge_list, disease_edgeweight, pcg_edge_list, pcg_edgeweight, mirna_pcg_pairs, disease_pcg_pairs, mirna_disease_pairs):
        # mhid = self.mgcn(memb, mirna_edgelist.t(), mirna_edgeweight)
        # dhid = self.dgcn(demb, disease_edge_list.t(), disease_edgeweight)
        # phid = self.pgcn(pemb, pcg_edge_list.t(), pcg_edgeweight)
        mhid, dhid, phid = memb.clone(), demb.clone(), pemb.clone()
        for layer in self.mgcn: mhid = layer(mhid, mirna_edgelist.t(), mirna_edgeweight)
        for layer in self.dgcn: dhid = layer(dhid, disease_edge_list.t(), disease_edgeweight)
        for layer in self.pgcn: phid = layer(phid, pcg_edge_list.t(), pcg_edgeweight)

        mhid = self.relu(mhid)
        dhid = self.relu(dhid)
        phid = self.relu(phid)

        mirna_vec = mhid[mirna_disease_pairs[:,0]]
        disease_vec = dhid[mirna_disease_pairs[:, 1]]
        assoc_vec = self.edge_operation(mirna_vec, disease_vec)

        vec1 = mhid[mirna_pcg_pairs[:,0]]
        vec2 = phid[mirna_pcg_pairs[:, 1]]
        mp_vec = self.edge_operation(vec1, vec2)

        vec1 = dhid[disease_pcg_pairs[:,0]]
        vec2 = phid[disease_pcg_pairs[:, 1]]
        dp_vec = self.edge_operation(vec1, vec2)

        assoc_out = self.sigmoid(self.assoc_clf(assoc_vec))
        mirna_pcg_out = self.sigmoid(self.mirna_pcg(mp_vec))
        disease_pcg_out = self.sigmoid(self.disease_pcg(dp_vec))

        return assoc_out.squeeze(), mirna_pcg_out.squeeze(), disease_pcg_out.squeeze()

def main(hidden_layers, edge_operation, edge_operation_name):
    ## read data from files

    # embeddings
    mirna = pd.read_csv('../data/generated_data/mirna_emb_dim64.csv',
                        header=None).values.tolist()
    mirna_emb = t.FloatTensor(mirna)
    disease = pd.read_csv('../data/generated_data/disease_emb_dim64.csv',
                          header=None).values.tolist()
    disease_emb = t.FloatTensor(disease)
    pcg = pd.read_csv('../data/generated_data/pcg_emb_dim64.csv',
                      header=None).values.tolist()
    pcg_emb = t.FloatTensor(pcg)

    # training data
    _, train_tensor, train_lbl_tensor = read_int_data(
        '../data/training_data/hmdd2_pos.csv',
        '../data/training_data/hmdd2_neg1_0.csv')

    # others
    disease_onto = pd.read_csv(
        '../data/original_data/disease_onto_pos.csv').values.tolist()
    disease_pcg = pd.read_csv('../data/generated_data/disease_pcg.csv',
                              header=None).values.tolist()
    hppi = pd.read_csv('../data/generated_data/pcg_pcg.csv',
                       header=None).values.tolist()
    mirna_family = pd.read_csv(
        '../data/original_data/mirna_fam_pos.csv').values.tolist()
    mirna_pcg = pd.read_csv('../data/generated_data/mirna_pcg.csv',
                            header=None).values.tolist()

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

    for p in mirna_family:       # 2 mirnas are connected if they belong to the same family
        idx1 = p[0]
        idx2 = p[1]
        mirna_edgelist.append([idx1, idx2])
        mirna_edgelist.append([idx2, idx1])
        mirna_edgeweight.append(1)
        mirna_edgeweight.append(1)

    for p in disease_onto:       # [children, parent] disease pairs
        idx1 = p[0]
        idx2 = p[1]
        disease_edgelist.append([idx1, idx2])
        disease_edgeweight.append(1)

    for p in hppi:               # [hprot1,hprot2,score] (human protein-protein interaction)
        idx1 = int(p[0])
        idx2 = int(p[1])
        ppi_edgelist.append([idx1, idx2])
        ppi_edgeweight.append(1)
        ppi_edgelist.append([idx2, idx1])
        ppi_edgeweight.append(1)

    for p in mirna_pcg:           # [mirna,pcg,score]
        idx1 = p[0]
        idx2 = p[1]
        mirna_pcg_pairs.append([idx1, idx2])
        mirna_pcg_weight.append(1)

    for p in disease_pcg:        # [disease,pcg,score]
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

        avg_metrics = list()

        for testrate in neg_rates:
            avg_acc, avg_auc, avg_ap, avg_rec, avg_f1, avg_loss = 0, 0, 0, 0, 0, 0

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
                # Loss and metrics
                avg_acc += acc
                avg_auc += auc_score
                avg_ap += ap_score
                avg_rec += rec
                avg_f1 += f1

                tloss0, tloss1, tloss2 = criterion(assoc_out, test_lbl), l1loss(mirna_pcg_out, mirna_pcg_weight), l1loss(disease_pcg_out,
                                                                                                                         disease_pcg_weight)
                tloss = w1 * tloss0 + w2 * tloss1 + w3 * tloss2
                test_loss = tloss.item()
                avg_loss += test_loss

            avg_loss /= 10
            avg_acc, avg_auc, avg_ap, avg_rec, avg_f1 = avg_acc / 10, avg_auc / 10, avg_ap / 10, avg_rec / 10, avg_f1 / 10
            avg_metrics.append([avg_auc, avg_acc, avg_ap, avg_rec, avg_f1])

        return all_scores, avg_loss, avg_metrics

    # ############################
    ## train the model
    w1 = 1.0
    w2 = 1.0
    w3 = 1.0
    model = MuCoMiD_experiment(64, hidden_layers, edge_operation, edge_operation_name).to(device)
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

            all_scores, avg_loss, avg_metrics = eval(model, neg_rates=[1,5,10], print_results=False)
            test_losses.append(avg_loss)
            test_accuracies.append(avg_metrics[0][4])

            print('Test loss: ', avg_loss, ' Test acc: ', avg_metrics[0][4])


    # Plot the losses and test accuracy
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.legend()
    plt.title(f'Losses for hidden layers {hidden_layers}, edge operation {edge_operation_name}')
    plt.show()

    plt.plot(test_accuracies, label='Test Accuracy')
    plt.legend()
    plt.title(f'Accuracy for hidden layers {hidden_layers}, edge operation {edge_operation_name}')
    plt.show()

    ## evaluate the model
    neg_rates = [1, 5, 10]
    all_scores, _, avg_metrics = eval(model, neg_rates, print_results=False)

    ## write the results into file
    if not os.path.exists('./results'): os.makedirs('./results')
    if edge_operation_name == 'mul': edge_operation_name = ''
    else: edge_operation_name = f'_{edge_operation_name}'

    writer = open(f'./results/layers_{hidden_layers}{edge_operation_name}.csv', 'w')
    writer.write('data,neg_rate,avg_auc,avg_acc,avg_ap,avg_rec,avg_f1\n')
    for idx, line in enumerate(avg_metrics):
        line = [str(item) for item in line]
        line = [datasrc, str(neg_rates[idx])] + line
        writer.write(','.join(line))
        writer.write('\n')
    writer.close()


if __name__ == "__main__":
    # one hidden layer
    hidden_layers = [64]
    edge_operation = lambda x, y: x * y
    edge_operation_name = 'mul'
    main(hidden_layers, edge_operation, edge_operation_name)

    hidden_layers = [64]
    edge_operation = lambda x, y: x + y
    edge_operation_name = 'add'
    main(hidden_layers, edge_operation, edge_operation_name)

    hidden_layers = [64]
    edge_operation = lambda x, y: t.cat((x, y), 1)
    edge_operation_name = 'concat'
    main(hidden_layers, edge_operation, edge_operation_name)

    # try different configurations
    edge_operation = lambda x, y: x * y

    hidden_layers = [32, 32]
    main(hidden_layers, edge_operation, edge_operation_name='mul')

    hidden_layers = [32, 32, 32]
    main(hidden_layers, edge_operation, edge_operation_name='mul')

    hidden_layers = [64, 64]
    main(hidden_layers, edge_operation, edge_operation_name= 'mul')

    hidden_layers = [64, 64, 64]
    main(hidden_layers, edge_operation, edge_operation_name= 'mul')

    hidden_layers = [64, 32, 16]
    main(hidden_layers, edge_operation, edge_operation_name= 'mul')

    hidden_layers = [32, 64, 32]
    main(hidden_layers, edge_operation, edge_operation_name= 'mul')



