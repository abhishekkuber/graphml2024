#this file evaluates the performance of the mpm model

from utils import *
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import sdne as sdne

def get_embedding(vectors: dict, x):
    matrix = np.zeros((
        x,
        len(list(vectors.values())[0])
    ))
    for key, value in vectors.items():
        matrix[int(key), :] = value
    return matrix

def Get_embedding_Matrix(train_pairs, mirna_pcg, disease_pcg, pcg_pcg, n_mirna, n_disease, n_pcg):
    graph1 = sdne.Graph()
    graph1.add_edgelist(train_pairs, mirna_pcg, disease_pcg, pcg_pcg, None, None, n_mirna, n_disease, n_pcg)
    model = sdne.SDNE(graph1, [1000, 128])
    return get_embedding(model.vectors, n_mirna + n_disease + n_pcg)

def read_int(pos_path, neg_path):
    pos_pairs = pd.read_csv(pos_path).values.tolist()
    neg_pairs = pd.read_csv(neg_path).values.tolist()
    pairs = pos_pairs + neg_pairs
    lbls = [1] * len(pos_pairs) + [0] * len(neg_pairs)
    return pairs, lbls


def constructNet(miRNA_dis_matrix):
    miRNA_matrix = np.matrix(np.zeros((miRNA_dis_matrix.shape[0], miRNA_dis_matrix.shape[0]), dtype=np.int8))
    dis_matrix = np.matrix(np.zeros((miRNA_dis_matrix.shape[1], miRNA_dis_matrix.shape[1]),dtype=np.int8))

    mat1 = np.hstack((miRNA_matrix,miRNA_dis_matrix))
    mat2 = np.hstack((miRNA_dis_matrix.T,dis_matrix))

    return np.vstack((mat1,mat2))

def get_data(pairs, mfeat, dfeat, msim, dsim):
    data = list()
    for p in pairs:
        cur_data = mfeat[p[0]] + dfeat[p[1]] + msim[p[0]] + dsim[p[1]]
        data.append(cur_data)
    return np.array(data)

def gen_data(pos_path, neg_path, didx, n_mirna):
    pos_df = pd.read_csv(pos_path)
    pos_df.columns = ['mirna', 'disease']
    pos_test = pos_df[pos_df['disease'] == didx]
    pos_test_mirna = pos_test['mirna'].unique().tolist()
    all_mirna = list(range(n_mirna))
    neg_test_mirna = np.setdiff1d(all_mirna, pos_test_mirna)
    neg_test = [[item,didx] for item in neg_test_mirna]
    test_data = pos_test.values.tolist() + neg_test
    test_lbl = [1] * len(pos_test) + [0] * len(neg_test)

    pos_train = pos_df[pos_df['disease'] != didx]

    neg_df = pd.read_csv(neg_path)
    neg_train = neg_df[neg_df['disease'] != didx]
    train_data = pos_train.values.tolist() + neg_train.values.tolist()
    train_lbl = [1] * len(pos_train) + [0] * len(neg_train)

    return train_data, train_lbl, test_data, test_lbl

def eval(pos_train_path, neg_train_path, all_test_path, mirna_sim, disease_sim, mirna_emb, disease_emb):
    train_pairs, train_lbl = read_int(pos_train_path, neg_train_path)

    all_scores = list()
    
    train_data = get_data(train_pairs, mirna_emb, disease_emb, mirna_sim, disease_sim)
    train_lbl = np.array(train_lbl)

    clf = RandomForestClassifier(random_state=1, n_estimators=350, oob_score=False, n_jobs=-1)
    clf.fit(train_data, train_lbl)

    neg_rates = [1,5,10]
    for pos_test_path in all_test_path:
        neg_test_pre = pos_test_path.replace('_pos.csv', '_neg.csv')
        datasrc = pos_test_path[pos_test_path.rfind('/')+1:].replace('_pos.csv', '')
        for testrate in neg_rates:
            for testset in range(10):
                cur_score = [datasrc, str(testrate), str(testset), str(1)]
                neg_test_path  = neg_test_pre.replace('.csv', str(testrate) + '_' + str(testset) + '.csv')
                # cur_score = [datasrc, str(trainrate), str(trainset), str(1), str(1), str(irun)]
                test_pairs, test_lbl = read_int(pos_test_path, neg_test_path)
                datasrc = pos_test_path[pos_test_path.rfind('/')+1:].replace('.csv', '')
                test_data = get_data(test_pairs, mirna_emb, disease_emb, mirna_sim, disease_sim)
                test_pred_lbl = clf.predict_proba(test_data)[:, 1]

                auc_score, ap_score, sn, sp, acc, prec, rec, f1, mcc = get_all_score(test_lbl, test_pred_lbl)
                tmp_score = [auc_score, ap_score, sn, sp, acc, prec, rec, f1]
                for ftmp in tmp_score:
                    cur_score.append(str(round(ftmp, 5)))
                print(cur_score)
                all_scores.append(cur_score)

    return all_scores

def main():
    
    mirna_sim = pd.read_csv('data/original_data/mirna_family_mat.csv', header=None).values.tolist()
    disease_sim = pd.read_csv('data/original_data/disease_sim.csv', header=None).values.tolist()

    fs_df = pd.read_csv('data/original_data/relieff_raw_pcg.csv', header=None).values.tolist()
    all_selected_feats = [item[0] for item in fs_df]
    selected_feats = all_selected_feats[:100]

    disease_pcg_mat = pd.read_csv('data/original_data/rdisease_pcg10.csv', header=None)
    mirna_pcg_mat = pd.read_csv('data/original_data/rmirna_pcg10.csv', header=None)

    disease_pcg_mat = pd.DataFrame(disease_pcg_mat.iloc[:, selected_feats]).values.tolist()
    mirna_pcg_mat = pd.DataFrame(mirna_pcg_mat.iloc[:, selected_feats]).values.tolist()

    mirna_pcg = list()
    disease_pcg = list()
    for i, rec in enumerate(mirna_pcg_mat):
        for j, item in enumerate(rec):
            if item != 0:
                mirna_pcg.append([i,j])

    for i, rec in enumerate(disease_pcg_mat):
        for j, item in enumerate(rec):
            if item != 0:
                disease_pcg.append([i,j])

    ppi_mat = pd.read_csv('data/original_data/hppi.csv')
    ppi_mat = ppi_mat[(ppi_mat['hprot1'].isin(selected_feats)) & (ppi_mat['hprot2'].isin(selected_feats))]
    pcg_pcg = [[selected_feats.index(int(item[0])), selected_feats.index(int(item[1]))] for item in ppi_mat.values.tolist()]


    writer = open('data/eval_results/mpm_output.txt', 'w+')
    writer.write('data,run,auc_score, ap_score, sn, sp, acc, prec, rec, f1\n')

    pos_train_path = 'data/training_data/hmdd2_pos.csv'
    neg_train_path = 'data/training_data/hmdd2_neg1_0.csv'
    all_test_path = ['data/test_data/new_mirna_pos.csv']

    n_mirna = len(mirna_sim)
    n_disease = len(disease_sim)

    pos_df = pd.read_csv(pos_train_path)
    miRNA_disease_emb = Get_embedding_Matrix(pos_df.values.tolist(), mirna_pcg, disease_pcg, pcg_pcg, n_mirna, n_disease, 100)

    mirna_emb = np.array(miRNA_disease_emb[0:n_mirna, 0:]).tolist()
    disease_emb = np.array(miRNA_disease_emb[n_mirna:n_mirna+n_disease, 0:]).tolist()
  
    all_scores = eval(pos_train_path, neg_train_path, all_test_path, mirna_sim, disease_sim, mirna_emb, disease_emb)
    for line in all_scores:
        writer.write(','.join(line))
        writer.write('\n')
    writer.close()


if __name__ == "__main__":
    main()
