import argparse

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
    graph1.add_edgelist(train_pairs, mirna_pcg, disease_pcg, pcg_pcg, n_mirna, n_disease, n_pcg)
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

def eval(pos_train_path, neg_train_path, didx, mirna_sim, disease_sim, mirna_emb, disease_emb, dname, n_feat):
    n_miRNA = len(mirna_sim)
    train_pairs, train_lbl, test_pairs, test_lbl = gen_data(pos_train_path, neg_train_path, didx, n_miRNA)
    if dname == 'D010300':
        train_pairs = train_pairs + test_pairs
        train_lbl = train_lbl + test_lbl

    all_scores = list()
    case_studies = ['D004314', 'D010300']
    for run in range(1):
        train_data = get_data(train_pairs, mirna_emb, disease_emb, mirna_sim, disease_sim)
        train_lbl = np.array(train_lbl)

        clf = RandomForestClassifier(random_state=1, n_estimators=350, oob_score=False, n_jobs=-1)
        clf.fit(train_data, train_lbl)

        test_data = get_data(test_pairs, mirna_emb, disease_emb, mirna_sim, disease_sim)
        test_pred_lbl = clf.predict_proba(test_data)[:, 1]

        auc_score, ap_score, sn, sp, acc, prec, rec, f1 = get_all_score(test_lbl, test_pred_lbl)
        print('NEMII disease', didx, ' testing auc_score:%.4f, ap_score:%.4f, precision:%.4f, recall:%.4f, f1:%.4f' % (
            auc_score, ap_score, prec, rec, f1))

        if dname in case_studies:
            join_list = [[p[0],target,pred] for p, target,pred in zip(test_pairs, test_lbl, test_pred_lbl)]
            join_df = pd.DataFrame(np.array(join_list), columns=['mirna', 'target', 'pred'])
            join_df.to_csv(neg_train_path.replace('.csv', '_' + dname + '_' + str(n_feat) + '_mpm_nem.csv'), index=False)


        all_scores.append([auc_score, ap_score, sn, sp, acc, prec, rec, f1])

    return all_scores

def main():
    parser = argparse.ArgumentParser(description='A Message Passing framework with multiple data integration for miRNA-disease association prediction')
    parser.add_argument('--epochs', type=int, default=200, metavar='N',
                        help='number of epochs to train')
    parser.add_argument('--n_runs', type=int, default=2, metavar='N',
                        help='number of experiment runs')
    parser.add_argument('--data_dir', default='./data/', help='dataset directory')
    parser.add_argument('--disease_pcg_path', default='rdisease_pcg1_12500.csv', help='disease feature path')
    parser.add_argument('--pos_train_path', default='all_assoc_pos.csv', help='pos train path')
    parser.add_argument('--mirna_pcg_path', default='rmirna_pcg1_12500.csv', help='mirna feature path')
    parser.add_argument('--neg_train_path', default='all_assoc_neg1.csv', help='negative train prefix')
    parser.add_argument('--pcg_setup', default='rpcg1', help='pass raw_pcg for the raw PCG features, or inf_pcg1 for the inferred PCG output from the Message passing layer, and other to use the paths passed in')

    parser.add_argument('--fs_setup', default='relieff', help='hsic or relieff')

    args = parser.parse_args()
    args.data_dir = standardize_dir(args.data_dir)
    args.disease_pcg_path = args.data_dir + args.disease_pcg_path
    args.pos_train_path = args.data_dir + args.pos_train_path
    args.mirna_pcg_path = args.data_dir + args.mirna_pcg_path
    args.neg_train_path = args.data_dir + args.neg_train_path

    if args.fs_setup == 'hsic':
        args.feat_path = './data/hsic_raw_pcg.csv'
    else:
        args.feat_path = './data/relieff_raw_pcg.csv'
    if args.pcg_setup == 'raw_pcg':
        args.disease_pcg_path = args.data_dir + 'disease_pcg.csv'
        args.mirna_pcg_path = args.data_dir + 'mirna_pcg.csv'
    elif args.pcg_setup == 'rpcg1':
        args.disease_pcg_path = args.data_dir + 'rdisease_pcg1.csv'
        args.mirna_pcg_path = args.data_dir + 'rmirna_pcg1.csv'
    elif args.pcg_setup == 'rpcg2':
        args.disease_pcg_path = args.data_dir + 'rdisease_pcg2.csv'
        args.mirna_pcg_path = args.data_dir + 'rmirna_pcg2.csv'
    elif args.pcg_setup == 'rpcg10':
        args.disease_pcg_path = args.data_dir + 'rdisease_pcg10.csv'
        args.mirna_pcg_path = args.data_dir + 'rmirna_pcg10.csv'

    disease_freq = pd.read_csv('./data/sorted_disease_freq.csv')
    disease_freq = disease_freq[disease_freq['freq'] > 100]
    dnames_list = disease_freq['disease'].unique().tolist()
    # 'D004314 : Down syndrome, 'D010300': Parkinson disease
    dnames_list = dnames_list + ['D004314', 'D010300']
    disease2idx = pd.read_csv('./data/disease.csv').values.tolist()
    disease2idx = {item[0]:i for i, item in enumerate(disease2idx)}
    didxes = [disease2idx[item] for item in dnames_list if item in disease2idx]

    mirna_sim = pd.read_csv('./data/mirna_family_mat.csv', header=None).values.tolist()
    disease_sim = pd.read_csv('./data/sim_data/disease_sim.csv', header=None).values.tolist()
    n_disease = len(disease_sim)

    fs_df = pd.read_csv(args.feat_path, header=None).values.tolist()
    all_selected_feats = [item[0] for item in fs_df]
    all_nfeats = [int(50*i) for i in range(1, 11)]
    all_nfeats = all_nfeats + [len(all_selected_feats)]
    all_nfeats = [100,len(all_selected_feats)]
    for nfeat in all_nfeats:
        if len(all_selected_feats) < nfeat and nfeat > 500:
            break

        selected_feats = all_selected_feats if len(all_selected_feats) < nfeat else all_selected_feats[:nfeat]


        disease_pcg_mat = pd.read_csv(args.disease_pcg_path, header=None)
        mirna_pcg_mat = pd.read_csv(args.mirna_pcg_path, header=None)
        n_mirna = mirna_pcg_mat.shape[0]

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

        ppi_mat = pd.read_csv('./data/hppi.csv')
        ppi_mat = ppi_mat[(ppi_mat['hprot1'].isin(selected_feats)) & (ppi_mat['hprot2'].isin(selected_feats))]
        pcg_pcg = [[selected_feats.index(int(item[0])), selected_feats.index(int(item[1]))] for item in ppi_mat.values.tolist()]


        for name,idx in zip(dnames_list, didxes):
            writer = open('./data/mpm_nem_2/'+name+'_' + args.pcg_setup + '_mpm_nem_' + args.fs_setup + '_' + str(nfeat) + '_independent_results.txt', 'w')
            writer.write('trainrate,auc_score, ap_score, sn, sp, acc, prec, rec, f1\n')

            pos_train_path = './data/all_assoc_pos.csv'
            neg_train_path_ori = './data/all_assoc_neg1.csv'
            pos_df = pd.read_csv(pos_train_path)
            pos_df.columns = ['mirna', 'disease']
            pos_train = pos_df[pos_df['disease'] != idx]
            miRNA_disease_emb = Get_embedding_Matrix(pos_train.values.tolist(), mirna_pcg, disease_pcg, pcg_pcg, n_mirna, n_disease, len(selected_feats))

            mirna_emb = np.array(miRNA_disease_emb[0:n_mirna, 0:]).tolist()
            disease_emb = np.array(miRNA_disease_emb[n_mirna:n_mirna+n_disease, 0:]).tolist()
            for itrain in range(10):
                neg_train_path = neg_train_path_ori.replace('.csv', '_' + str(itrain) + '.csv')
                all_scores = eval(pos_train_path, neg_train_path, idx, mirna_sim, disease_sim, mirna_emb, disease_emb, name, nfeat)
                for score_list in all_scores:
                    for i, score in enumerate(score_list):
                        if i > 0:
                            writer.write(',')
                        writer.write(',' + str(round(score, 4)))
                    writer.write('\n')
            writer.close()


if __name__ == "__main__":
    main()


