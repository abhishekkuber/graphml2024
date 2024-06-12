import os
import pandas as pd
import numpy as np
from sklearn import metrics
import torch as t
import random
from sklearn.model_selection import train_test_split

# Generate data for the inductive testing setup
# If forHete = True then we need to update the disease index in the assoc data
# since in the Hete graph, its index starting from n_mirna
def create_data4case_study(pos_path, neg_path, didx, n_mirna, forHete = False):
    pos_df = pd.read_csv(pos_path)
    pos_df.columns = ['mirna', 'disease']
    pos_test = pos_df[pos_df['disease'] == didx]
    pos_test_mirna = pos_test['mirna'].unique().tolist()
    all_mirna = list(range(n_mirna))
    neg_test_mirna = np.setdiff1d(all_mirna, pos_test_mirna)
    neg_test = [[item,didx] for item in neg_test_mirna]
    test_data = pos_test.values.tolist() + neg_test
    if forHete:
        test_data = [[item[0], item[1] + n_mirna] for item in test_data]# account for the index diff
    test_lbl = [1] * len(pos_test) + [0] * len(neg_test)

    pos_train = pos_df[pos_df['disease'] != didx]
    pos_train, pos_val = train_test_split(pos_train.values.tolist(), test_size=0.05)

    neg_df = pd.read_csv(neg_path)
    neg_train = neg_df[neg_df['disease'] != didx]
    neg_df = neg_train.values.tolist()
    indexes = list(range(len(neg_df)))
    random.shuffle(indexes)
    selected_indexes = indexes[:int(10*len(pos_df))]
    neg_df = [neg_df[item] for item in selected_indexes]

    neg_train, neg_val = train_test_split(neg_df, test_size=0.05)
    train_data = pos_train + neg_train
    if forHete:
        train_data = [[item[0], item[1] + n_mirna] for item in train_data]# account for the index diff
    train_lbl = [1] * len(pos_train) + [0] * len(neg_train)

    val_data = pos_val + neg_val
    if forHete:
        val_data = [[item[0], item[1] + n_mirna] for item in val_data] # account for the index diff
    val_lbl = [1] * len(pos_val) + [0] * len(neg_val)
    val_tensor = t.LongTensor(val_data)

    return t.LongTensor(train_data), t.FloatTensor(train_lbl).squeeze(), t.LongTensor(test_data), test_lbl, val_tensor, val_lbl

# Read transductive testing data
# If forHete = True then we need to update the disease index in the assoc data
# since in the Hete graph, its index starting from n_mirna
def read_test_data(pos_path, neg_path, n_mirna, forHete=False):
    pos_df = pd.read_csv(pos_path).values.tolist()
    neg_df = pd.read_csv(neg_path).values.tolist()
    int_edges = pos_df + neg_df
    if forHete:
        int_edges = [[item[0], item[1] + n_mirna] for item in int_edges]
    int_lbl = [1] * len(pos_df) + [0] * len(neg_df)
    return int_edges, t.LongTensor(int_edges), t.FloatTensor(int_lbl).squeeze()

# Read transductive training data and split into train and validation sets
# If forHete = True then we need to update the disease index in the assoc data
# since in the Hete graph, its index starting from n_mirna
def read_train_data(pos_path, neg_path, n_mirna, forHete=False):
    pos_df = pd.read_csv(pos_path).values.tolist()
    pos_train, pos_val = train_test_split(pos_df, test_size=0.05)

    neg_df = pd.read_csv(neg_path).values.tolist()
    indexes = list(range(len(neg_df)))
    random.shuffle(indexes)
    selected_indexes = indexes[:int(10*len(pos_df))]
    neg_df = [neg_df[item] for item in selected_indexes]

    neg_train, neg_val = train_test_split(neg_df, test_size=0.05)
    train_data = pos_train + neg_train
    if forHete:
        train_data = [[item[0], item[1] + n_mirna] for item in train_data]
    train_lbl = [1] * len(pos_train) + [0] * len(neg_train)

    val_data = pos_val + neg_val
    if forHete:
        val_data = [[item[0], item[1] + n_mirna] for item in val_data]
    val_lbl = [1] * len(pos_val) + [0] * len(neg_val)
    val_tensor = t.LongTensor(val_data)
    val_lbl_tensor = t.FloatTensor(val_lbl)

    return train_data, val_data, t.LongTensor(train_data), t.FloatTensor(train_lbl).squeeze(), val_tensor, val_lbl_tensor.squeeze()

# for logistic regression with random walk embedding
def get_feat(pair_list, mfeat, dfeat):
    data = list()
    for p in pair_list:
        data.append(mfeat[p[0], :] * dfeat[p[1], :])

    return data

def save_model(model, save_path):
    t.save(model.state_dict(), save_path)

def load_model(model, model_path):
    model.load_state_dict(t.load(model_path))

def load_dict(path, has_header=True):
    if has_header:
        df = pd.read_csv(path).values.tolist()
    else:
        df = pd.read_csv(path, header=None).values.tolist()
    if len(df[0]) == 1:
        res_dict = {item[0]:i for i,item in enumerate(df)}
    else:
        res_dict = {item[0]:item[1] for item in df}
    return res_dict

def standardize_dir(dir):
    res_dir = dir
    if not res_dir.endswith('/') and not res_dir.endswith('\\'):
        res_dir += '/'

    if not os.path.exists(res_dir):
        os.mkdir(res_dir)
    return res_dir

# return auc and ap score
def get_score(targets, preds):
    auc_score = metrics.roc_auc_score(targets, preds, average='micro')

    ap_score = metrics.average_precision_score(targets, preds, average='micro')

    return auc_score, ap_score

# return all scores
def get_all_score(targets, preds, K=10):
    auc_score = metrics.roc_auc_score(targets, preds, average='micro')

    aupr_score = metrics.average_precision_score(targets, preds, average='micro')

    thres = 0.5
    y_preds = [0 if pred < thres else 1 for pred in preds]
    cm = metrics.confusion_matrix(targets, y_preds)
    tn, fp, fn, tp = cm.ravel()
    sn = round(float(tp) / (tp + fn),4)
    sp = round(float(tn) / (tn + fp),4)

    acc = round(metrics.accuracy_score(targets, y_preds),4)
    f1 = metrics.f1_score(targets, y_preds)
    precision = metrics.precision_score(targets, y_preds)
    recall = metrics.recall_score(targets, y_preds)
    mcc = metrics.matthews_corrcoef(targets, y_preds)

    return auc_score, aupr_score, sn, sp, acc, precision, recall, f1, mcc

def get_top(targets, preds, K=100):
    columns = ['y_pred', 'y_true','idx']
    indexes = [i for i in range(len(preds))]
    join_list = [[pred, target, idx] for pred, target, idx in zip(preds, targets, indexes)]
    df = pd.DataFrame(np.array(join_list), columns=columns)

    unique_y_pred = df['y_pred'].unique().tolist()
    tmp_df = df[df['y_true'] == 1.0]
    all_pos = len(tmp_df)

    selected_list = list()
    for item in sorted(unique_y_pred, reverse=True):
        pairs = df[df['y_pred'] == item].values.tolist()
        random.shuffle(pairs)
        for p in pairs:
            selected_list.append(p)
            if len(selected_list) >= K:
                arr = np.array(selected_list)
                lbls = arr[:,1].tolist()
                freq = lbls.count(1.0)
                return all_pos, freq

import numpy.linalg as LA
def calculate_gip(matrix):
    """
    calculate gip in regard for disease and miRNA
    The row of the matrix is miRNA, the columns represent diseases

    :param matrix: numpy array
    :return: tuple for first disease gip and then miRNA gip
    """
    A = matrix
    n_miRNA = A.shape[0]
    n_disease = A.shape[1]

    # calculate GIP for miRNA
    A = np.asmatrix(A)
    gamd = n_miRNA / (LA.norm(A, 'fro') ** 2)
    km = np.mat(np.zeros((n_miRNA, n_miRNA)))
    D = A * A.T
    for i in range(n_miRNA):
        for j in range(i, n_miRNA):
            km[j, i] = np.exp(-gamd * (D[i, i] + D[j, j] - 2 * D[i, j]))
    km = km + km.T - np.diag(np.diag(km))
    KM = np.asarray(km)

    # calculate GIP for disease
    gamm = n_disease / (LA.norm(A, 'fro') ** 2)
    kd = np.mat(np.zeros((n_disease, n_disease)))
    E = A.T * A
    for i in range(n_disease):
        for j in range(i, n_disease):
            kd[j, i] = np.exp(-gamm * (E[i, i] + E[j, j] - 2 * E[i, j]))
    kd = kd + kd.T - np.diag(np.diag(kd))
    KD = np.asarray(kd)

    return KM, KD


## GENERAL
def get_device():
    return t.device('cuda' if t.cuda.is_available() else 'cpu')


## SAVING AND LOADING MODELS
HYBRID_MODEL_PATH = "trained_models"


def get_hybrid_model_path():
    return f"{HYBRID_MODEL_PATH}/hybrid.model"


def save_hybrid_model(model):
    print("hybrid model saved")
    t.save(model.state_dict(), get_hybrid_model_path())


def load_hybrid_model():
    return t.load(get_hybrid_model_path())


##LOADING DATA

def read_int_data(pos_path, neg_path):
    pos_df = pd.read_csv(pos_path).values.tolist()
    neg_df = pd.read_csv(neg_path).values.tolist()
    int_edges = pos_df + neg_df
    int_lbl = [1] * len(pos_df) + [0] * len(neg_df)
    return pos_df, t.LongTensor(int_edges), t.FloatTensor(int_lbl)


def load_data(device='cpu'):
    # embeddings
    mirna_emb = t.FloatTensor(pd.read_csv('data/generated_data/mirna_emb.csv', header=None).values)
    disease_emb = t.FloatTensor(pd.read_csv('data/generated_data/disease_emb.csv', header=None).values)
    pcg_emb = t.FloatTensor(pd.read_csv('data/generated_data/pcg_emb.csv', header=None).values)

    # training data
    _, train_tensor, train_lbl_tensor = read_int_data('data/training_data/hmdd2_pos.csv',
                                                      'data/training_data/hmdd2_neg1_0.csv')

    # others
    disease_onto = pd.read_csv('data/original_data/disease_onto_pos.csv').values.tolist()
    disease_pcg = pd.read_csv('data/generated_data/disease_pcg.csv', header=None).values.tolist()
    hppi = pd.read_csv('data/generated_data/pcg_pcg.csv', header=None).values.tolist()
    mirna_family = pd.read_csv('data/original_data/mirna_fam_pos.csv').values.tolist()
    mirna_pcg = pd.read_csv('data/generated_data/mirna_pcg.csv', header=None).values.tolist()

    # Convert lists to tensors
    disease_pcg_pairs = t.LongTensor([[p[0], p[1]] for p in disease_pcg])
    disease_pcg_weight = t.FloatTensor([1 for _ in disease_pcg])
    mirna_pcg_pairs = t.LongTensor([[p[0], p[1]] for p in mirna_pcg])
    mirna_pcg_weight = t.FloatTensor([1 for _ in mirna_pcg])
    mirna_edgelist = t.LongTensor([[p[0], p[1]] for p in mirna_family] + [[p[1], p[0]] for p in mirna_family])
    mirna_edgeweight = t.FloatTensor([1 for _ in mirna_family] * 2)
    disease_edgelist = t.LongTensor([[p[0], p[1]] for p in disease_onto])
    disease_edgeweight = t.FloatTensor([1 for _ in disease_onto])
    ppi_edgelist = t.LongTensor([[int(p[0]), int(p[1])] for p in hppi] + [[int(p[1]), int(p[0])] for p in hppi])
    ppi_edgeweight = t.FloatTensor([1 for _ in hppi] * 2)

    data = {
        'mirna_emb': mirna_emb,
        'disease_emb': disease_emb,
        'pcg_emb': pcg_emb,
        'train_tensor': train_tensor,
        'train_lbl_tensor': train_lbl_tensor,
        'disease_pcg_pairs': disease_pcg_pairs,
        'disease_pcg_weight': disease_pcg_weight,
        'mirna_pcg_pairs': mirna_pcg_pairs,
        'mirna_pcg_weight': mirna_pcg_weight,
        'mirna_edgelist': mirna_edgelist,
        'mirna_edgeweight': mirna_edgeweight,
        'disease_edgelist': disease_edgelist,
        'disease_edgeweight': disease_edgeweight,
        'ppi_edgelist': ppi_edgelist,
        'ppi_edgeweight': ppi_edgeweight
    }

    for key, tensor in data.items():
        data[key] = tensor.to(device)

    return data
