# this file evaluates the performance of the hybrid model (mpm+cmtt)
from matplotlib import pyplot as plt
from simple_mlp import SimpleMLP  # Import the new MLP model
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

def main():
    ## read data from files

    # embeddings
    mirna = pd.read_csv('data/generated_data/mirna_emb.csv',
                        header=None).values.tolist()
    mirna_emb = t.FloatTensor(mirna)
    disease = pd.read_csv('data/generated_data/disease_emb.csv',
                          header=None).values.tolist()
    disease_emb = t.FloatTensor(disease)
    pcg = pd.read_csv('data/generated_data/pcg_emb.csv',
                      header=None).values.tolist()
    pcg_emb = t.FloatTensor(pcg)

    # torch.Size([1618, 32])
    # print(mirna_emb.shape)
    # torch.Size([3679, 32])
    # print(disease_emb.shape)
    # torch.Size([100, 32])
    # print(pcg_emb.shape)

    # training data
    _, train_tensor, train_lbl_tensor = read_int_data(
        'data/training_data/hmdd2_pos.csv',
        'data/training_data/hmdd2_neg1_0.csv')

    # others
    disease_onto = pd.read_csv(
        'data/original_data/disease_onto_pos.csv').values.tolist()
    disease_pcg = pd.read_csv('data/generated_data/disease_pcg.csv',
                              header=None).values.tolist()
    hppi = pd.read_csv('data/generated_data/pcg_pcg.csv',
                       header=None).values.tolist()
    mirna_family = pd.read_csv(
        'data/original_data/mirna_fam_pos.csv').values.tolist()
    mirna_pcg = pd.read_csv('data/generated_data/mirna_pcg.csv',
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

    pos_test_path = 'data/test_data/new_mirna_pos.csv'  # change this for a different test dataset
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
                
                mirna_vec_test = mirna_emb[test_tensor[:,0]]
                disease_vec_test = disease_emb[test_tensor[:,1]]
                x_test = t.cat((mirna_vec_test, disease_vec_test), dim=1)
                
                assoc_out = model(x_test.float())

                auc_score, ap_score, sn, sp, acc, prec, rec, f1, mcc = get_all_score(test_lbl, assoc_out.detach().cpu().numpy())
                tmp_score = [auc_score, ap_score, sn, sp, acc, prec, rec, f1, mcc]
                for ftmp in tmp_score:
                    cur_score.append(str(round(ftmp, 5)))
                if print_results: print(cur_score)
                all_scores.append(cur_score)

                # #############
                # Loss and acc
                avg_acc += acc

                tloss0 = criterion(assoc_out, test_lbl)
                test_loss = tloss0.item()
                avg_loss += test_loss

            avg_acc /= 10
            avg_loss /= 10

        return all_scores, avg_acc, avg_loss

    # ############################
    ## train the model
    # input_dim = train_tensor.size(1)
    hidden_dim = 128 
    model = SimpleMLP(input_dim=64, hidden_dim=hidden_dim).to(device)
    optimizer = t.optim.Adam(model.parameters(), lr=0.001)

    train_losses, test_losses, test_accuracies = list(), list(), list()

    for epoch in range(0, 200):
        model.train()
        model.zero_grad()
        
        mirna_vec = mirna_emb[train_tensor[:,0]]
        disease_vec = disease_emb[train_tensor[:,1]]
        x = t.cat((mirna_vec, disease_vec), dim=1)

        # Both of them are of size : torch.Size([9184, 32])

        assoc_out = model(x.float())
        # torch.Size([9184, 1])
        # print(assoc_out.shape)
        # torch.Size([9184])
        # print(train_lbl_tensor.shape)
        loss = criterion(assoc_out, train_lbl_tensor)
        
        loss.backward()
        optimizer.step()
        loss_val = loss.item()

        print('Epoch: ', epoch, ' loss: ', loss_val / train_lbl_tensor.size(0))

        if epoch % 1 == 0:
            train_losses.append(loss_val)

            all_scores, avg_acc, avg_loss = eval(model, neg_rates=[5], print_results=False)
            test_losses.append(avg_loss)
            test_accuracies.append(avg_acc)

            print('Test loss: ', avg_loss, ' Test acc: ', avg_acc)


    # Plot the losses and test accuracy
    plt.figure()
    plt.plot(train_losses, label='Training loss')
    plt.plot(test_losses, label='Test loss')
    plt.legend()
    plt.title('Losses')
    plt.savefig('plots/baseline/baseline_loss_curves.png')

    plt.figure()
    plt.plot(test_accuracies, label='Test accuracy')
    plt.legend()
    plt.title('Test Accuracy')
    plt.savefig('plots/baseline/baseline_test_acc.png')

    np.save('plots/baseline/baseline_train_losses.npy', train_losses)
    np.save('plots/baseline/baseline_test_losses.npy', test_losses)
    np.save('plots/baseline/baseline_test_accuracies.npy', test_accuracies)


    ## evaluate the model
    all_scores, _, _ = eval(model, neg_rates=[10])

    ## write the results into file
    writer = open('data/eval_results/baseline_output.txt', 'w+')
    writer.write('data,run,auc_score, ap_score, sn, sp, acc, prec, rec, f1, mcc\n')
    for line in all_scores:
        writer.write(','.join(line))
        writer.write('\n')
    writer.close()


if __name__ == "__main__":
    main()
