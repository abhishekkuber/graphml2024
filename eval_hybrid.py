#this file evaluates the performance of the hybrid model (mpm+cmtt)

from model import *
from utils import *

def read_int_data(pos_path, neg_path):
    pos_df = pd.read_csv(pos_path).values.tolist()
    neg_df = pd.read_csv(neg_path).values.tolist()
    int_edges = pos_df + neg_df
    int_lbl = [1] * len(pos_df) + [0] * len(neg_df)
    return pos_df, t.LongTensor(int_edges), t.FloatTensor(int_lbl)


import torch as t
import pandas as pd


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

    for p in mirna_family:
        idx1 = p[0]
        idx2 = p[1]
        mirna_edgelist.append([idx1, idx2])
        mirna_edgelist.append([idx2, idx1])
        mirna_edgeweight.append(1)
        mirna_edgeweight.append(1)

    for p in disease_onto:
        idx1 = p[0]
        idx2 = p[1]
        disease_edgelist.append([idx1, idx2])
        disease_edgeweight.append(1)

    for p in hppi:
        idx1 = int(p[0])
        idx2 = int(p[1])
        ppi_edgelist.append([idx1, idx2])
        ppi_edgeweight.append(1)
        ppi_edgelist.append([idx2, idx1])
        ppi_edgeweight.append(1)

    for p in mirna_pcg:
        idx1 = p[0]
        idx2 = p[1]
        mirna_pcg_pairs.append([idx1, idx2])
        mirna_pcg_weight.append(1)

    for p in disease_pcg:
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

    ## train the model
    w1 = 1.0
    w2 = 1.0
    w3 = 1.0
    model = MuCoMiD(32, 32).to(device)
    optimizer = t.optim.Adam(model.parameters(), lr=0.001)

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
        l1 = loss0.item()
        l2 = loss1.item()
        l3 = loss2.item()
        w1 = 1
        w2 = l3 / (l1 + l2 + l3 + 1e-10)
        w3 = l2 / (l1 + l2 + l3 + 1e-10)

        loss.backward()
        optimizer.step()
        loss_val = loss.item()
        print('Epoch: ', epoch, ' loss: ', loss_val / train_lbl_tensor.size(0))

    ## evaluate the model
    pos_test_path = 'data/test_data/new_mirna_pos.csv'
    all_scores = list()
    neg_rates = [1, 5, 10]
    neg_test_pre = pos_test_path.replace('_pos.csv', '_neg.csv')
    datasrc = pos_test_path[pos_test_path.rfind('/') + 1:].replace('_pos.csv',
                                                                   '')
    for testrate in neg_rates:
        for testset in range(10):
            cur_score = [datasrc, str(testrate), str(testset), str(1)]
            neg_test_path = neg_test_pre.replace('.csv',
                                                 str(testrate) + '_' + str(
                                                     testset) + '.csv')
            _, test_tensor, test_lbl = read_int_data(pos_test_path,
                                                     neg_test_path)
            datasrc = pos_test_path[pos_test_path.rfind('/') + 1:].replace(
                '.csv', '')
            test_tensor = test_tensor.to(device)
            assoc_out, mirna_pcg_out, disease_pcg_out = model(mirna_emb,
                                                              disease_emb,
                                                              pcg_emb,
                                                              mirna_edgelist,
                                                              mirna_edgeweight,
                                                              disease_edgelist,
                                                              disease_edgeweight,
                                                              ppi_edgelist,
                                                              ppi_edgeweight,
                                                              mirna_pcg_pairs,
                                                              disease_pcg_pairs,
                                                              test_tensor)

            auc_score, ap_score, sn, sp, acc, prec, rec, f1, mcc = get_all_score(
                test_lbl, assoc_out.detach().cpu().numpy())
            tmp_score = [auc_score, ap_score, sn, sp, acc, prec, rec, f1]
            for ftmp in tmp_score:
                cur_score.append(str(round(ftmp, 5)))
            print(cur_score)
            all_scores.append(cur_score)

    ## write the results into file
    writer = open('data/eval_results/hybrid_output.txt', 'w+')
    writer.write('data,run,auc_score, ap_score, sn, sp, acc, prec, rec, f1\n')
    for line in all_scores:
        writer.write(','.join(line))
        writer.write('\n')
    writer.close()


if __name__ == "__main__":
    main()
