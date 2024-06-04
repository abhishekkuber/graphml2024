from model import *
import argparse
from utils import *

def read_int_data(pos_path, neg_path):
    pos_df = pd.read_csv(pos_path).values.tolist()
    neg_df = pd.read_csv(neg_path).values.tolist()
    int_edges = pos_df + neg_df
    int_lbl = [1] * len(pos_df) + [0] * len(neg_df)
    return pos_df, t.LongTensor(int_edges), t.FloatTensor(int_lbl)

def case_studies():

    # Build lai data tu dau di cai' :(
    parser = argparse.ArgumentParser(description='Neural based matrix completion for virus-host PPI')
    parser.add_argument('--epochs', type=int, default=200, metavar='N',
                        help='number of epochs to train')
    parser.add_argument('--n_runs', type=int, default=5, metavar='N',
                        help='number of runs')
    parser.add_argument('--data_dir', default='D:\cmtt\data\\', help='dataset directory')
    parser.add_argument('--disease_path', default='disease.csv', help='human id path')
    parser.add_argument('--disease_onto_path', default='disease_onto.csv', help='human id path')
    parser.add_argument('--disease_pcg_path', default='disease_pcg.csv', help='virus go annotation path')
    parser.add_argument('--hppi_path', default='hppi.csv', help='human go annotation path')
    parser.add_argument('--mirna_path', default='mirna.csv', help='the go ontology path')
    parser.add_argument('--mirna_disease_path', default='hmdd2.csv', help='the go index dictionary')
    parser.add_argument('--mirna_fam_path', default='mirna_family.csv', help='pos train path')
    parser.add_argument('--mirna_pcg_path', default='mirna_pcg.csv', help='pos_test_filter.csv')
    parser.add_argument('--pcg_path', default='pcgs.csv', help='the go ontology path')
    parser.add_argument('--neg_train_path', default='hmdd2_neg.csv', help='random_neg_train_0.csv')
    parser.add_argument('--pos_test_path', default='held_out1.csv', help='random_neg_test_0.csv')
    parser.add_argument('--neg_test_path', default='held_out1_neg.csv', help='random_neg_test_0.csv')


    args = parser.parse_args()

    args.data_dir = standardize_dir(args.data_dir)
    args.disease_path = args.data_dir + args.disease_path
    args.disease_onto_path = args.data_dir + args.disease_onto_path
    args.disease_pcg_path = args.data_dir + args.disease_pcg_path
    args.hppi_path = args.data_dir + args.hppi_path
    args.mirna_path = args.data_dir + args.mirna_path
    args.mirna_disease_path = args.data_dir + args.mirna_disease_path
    args.mirna_fam_path = args.data_dir + args.mirna_fam_path
    args.mirna_pcg_path = args.data_dir + args.mirna_pcg_path
    args.pcg_path = args.data_dir + args.pcg_path
    args.neg_train_path = args.data_dir + args.neg_train_path
    args.pos_test_path = args.data_dir + args.pos_test_path
    args.neg_test_path = args.data_dir + args.neg_test_path
    args.n_runs = int(args.n_runs)

    disease_df = pd.read_csv(args.disease_path)
    n_disease = len(disease_df)
    disease_index_tensor = t.LongTensor(list(range(n_disease)))

    mirna_df = pd.read_csv(args.mirna_path)
    n_mirna = len(mirna_df)
    mirna_index_tensor = t.LongTensor(list(range(n_mirna)))

    mirna_df = pd.read_csv(args.pcg_path)
    n_pcg = len(mirna_df)
    pcg_index_tensor = t.LongTensor(list(range(n_pcg)))



    disease_onto = pd.read_csv(args.disease_onto_path).values.tolist()
    disease_pcg = pd.read_csv(args.disease_pcg_path)
    disease_pcg = disease_pcg.values.tolist()
    hppi = pd.read_csv(args.hppi_path).values.tolist()
    mirna_family = pd.read_csv(args.mirna_fam_path).values.tolist()
    mirna_pcg = pd.read_csv(args.mirna_pcg_path).values.tolist()

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
        ppi_edgeweight.append(p[2])
        ppi_edgelist.append([idx2, idx1])
        ppi_edgeweight.append(p[2])


    for p in mirna_pcg:
        idx1 = p[0]
        idx2 = p[1]
        mirna_pcg_pairs.append([idx1,idx2])
        mirna_pcg_weight.append(p[2])

    for p in disease_pcg:
        idx1 = p[0]
        idx2 = p[1]
        disease_pcg_pairs.append([idx1,idx2])
        disease_pcg_weight.append(p[2])#/max_dpcg)


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
    # l1loss = t.nn.KLDivLoss()#t.nn.BCELoss()# t.nn.L1Loss()


    args.epochs = int(args.epochs)
    lr = 0.001

    args.data_dir = 'D:\cmtt\data\\'
    case_studies_names = ['breast_cancer', 'pancreatic_cancer', 'diabetes']
    writer = open(args.data_dir.replace('data', 'results') + 'mtt_case_studies.csv', 'w')
    writer.write('disease, auc_score,  ap_score,top10,top20,top30,top40,top50,top60,top70,top80,top90,top100\n')
    all_means = list()
    all_std = list()
    args.n_runs = 5
    # args.epochs = 100
    for disease in case_studies_names:
        args.mirna_disease_path = args.data_dir + disease + '_pos_train.csv'
        args.pos_test_path = args.data_dir + disease + '_pos_test.csv'
        args.neg_train_path = args.data_dir + disease + '_neg_train.csv'
        args.neg_test_path = args.data_dir + disease + '_neg_test.csv'
        pos_train_pairs, train_tensor, train_lbl_tensor = read_int_data(args.mirna_disease_path, args.neg_train_path)
        neg_test_pairs, test_tensor, test_lbl_tensor = read_int_data(args.pos_test_path, args.neg_test_path)
        test_lbl = test_lbl_tensor.detach().numpy()

        if torch.cuda.is_available():
            mirna_index_tensor = mirna_index_tensor.cuda()
            disease_index_tensor = disease_index_tensor.cuda()
            pcg_index_tensor = pcg_index_tensor.cuda()
            mirna_edgelist = mirna_edgelist.cuda()
            mirna_edgeweight = mirna_edgeweight.cuda()
            disease_edgelist = disease_edgelist.cuda()
            disease_edgeweight = disease_edgeweight.cuda()
            ppi_edgelist = ppi_edgelist.cuda()
            ppi_edgeweight = ppi_edgeweight.cuda()
            mirna_pcg_pairs = mirna_pcg_pairs.cuda()
            disease_pcg_pairs = disease_pcg_pairs.cuda()
            train_tensor = train_tensor.cuda()
            train_lbl_tensor = train_lbl_tensor.cuda()
            mirna_pcg_weight = mirna_pcg_weight.cuda()
            disease_pcg_weight = disease_pcg_weight.cuda()
            criterion = criterion.cuda()
            l1loss = l1loss.cuda()
            test_tensor = test_tensor.cuda()
        all_scores = list()
        for irun in range(args.n_runs):
            model = MuCoMiD(32, n_mirna, n_disease, n_pcg, 32)
            if torch.cuda.is_available():
                model = model.cuda()
            optimizer = t.optim.Adam(model.parameters(), lr=lr)
            w1 = 1
            w2 = 1
            w3 = 1
            for epoch in range(0, args.epochs):
                model.train()
                model.zero_grad()
                assoc_out, mirna_pcg_out, disease_pcg_out = model(mirna_index_tensor, disease_index_tensor,
                                                                  pcg_index_tensor, mirna_edgelist, mirna_edgeweight,
                                                                  disease_edgelist, disease_edgeweight, ppi_edgelist,
                                                                  ppi_edgeweight, mirna_pcg_pairs, disease_pcg_pairs,
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

                if epoch % 10 == 0:
                    model.eval()
                    pred_score, _, _ = model(mirna_index_tensor, disease_index_tensor, pcg_index_tensor, mirna_edgelist,
                                             mirna_edgeweight, disease_edgelist, disease_edgeweight, ppi_edgelist,
                                             ppi_edgeweight, mirna_pcg_pairs, disease_pcg_pairs, test_tensor)
                    pred_score = pred_score.detach().numpy() if not torch.cuda.is_available() else pred_score.cpu().detach().numpy()

                    test_pred_lbl = pred_score.tolist()
                    test_pred_lbl = [item[0] if type(item) == list else item for item in test_pred_lbl]
                    auc_score, ap_score = get_score(test_lbl, test_pred_lbl)
                    print('auc_score:%.4f, ap_score:%.4f' % (auc_score, ap_score))
            model.eval()
            pred_score, _, _ = model(mirna_index_tensor, disease_index_tensor, pcg_index_tensor, mirna_edgelist, mirna_edgeweight, disease_edgelist, disease_edgeweight, ppi_edgelist,
                                     ppi_edgeweight, mirna_pcg_pairs, disease_pcg_pairs, test_tensor)
            pred_score = pred_score.detach().numpy() if not torch.cuda.is_available() else pred_score.cpu().detach().numpy()

            test_pred_lbl = pred_score.tolist()
            test_pred_lbl = [item[0] if type(item) == list else item for item in test_pred_lbl]
            cur_score = [ auc_score, ap_score]

            ks = [10,20,30,40,50,60,70,80,90,100]
            top_results = list()

            for k in ks:
                if len(test_pred_lbl) <= k:
                    top_results.append(int(len(test_lbl) / 2))
                    continue
                topk = get_top(test_lbl, test_pred_lbl, k)
                top_results.append(topk)
            cur_score.extend(top_results)

            all_scores.append(cur_score)
            writer.write(disease)
            for score in cur_score:
                writer.write(',' + str(round(score,5)))
            writer.write('\n')

        # for score in all_scores
        arr = np.array(all_scores)
        mean = list(np.mean(arr, axis=0))
        std = list(np.std(arr, axis=0))
        mean = [str(round(item,4)) for item in mean]
        std = [str(round(item,4)) for item in std]
        writer.write('Average_'+disease + ',' + ','.join(mean) + '\n')
        writer.write('std_'+disease + ',' + ','.join(std) + '\n')

def main():

    # Build lai data tu dau di cai' :(
    parser = argparse.ArgumentParser(description='Neural based matrix completion for virus-host PPI')
    parser.add_argument('--epochs', type=int, default=200, metavar='N',
                        help='number of epochs to train')
    parser.add_argument('--n_runs', type=int, default=5, metavar='N',
                        help='number of runs')
    parser.add_argument('--data_dir', default='D:\cmtt\data\\', help='dataset directory')
    parser.add_argument('--disease_path', default='disease.csv', help='human id path')
    parser.add_argument('--disease_onto_path', default='disease_onto.csv', help='human id path')
    parser.add_argument('--disease_pcg_path', default='disease_pcg.csv', help='virus go annotation path')
    parser.add_argument('--hppi_path', default='hppi.csv', help='human go annotation path')
    parser.add_argument('--mirna_path', default='mirna.csv', help='the go ontology path')
    parser.add_argument('--mirna_disease_path', default='hmdd2.csv', help='the go index dictionary')
    parser.add_argument('--mirna_fam_path', default='mirna_family.csv', help='pos train path')
    parser.add_argument('--mirna_pcg_path', default='mirna_pcg.csv', help='pos_test_filter.csv')
    parser.add_argument('--pcg_path', default='pcgs.csv', help='the go ontology path')
    parser.add_argument('--neg_train_path', default='hmdd2_neg.csv', help='random_neg_train_0.csv')
    parser.add_argument('--pos_test_path', default='held_out1.csv', help='random_neg_test_0.csv')
    parser.add_argument('--neg_test_path', default='held_out1_neg.csv', help='random_neg_test_0.csv')


    args = parser.parse_args()

    args.data_dir = standardize_dir(args.data_dir)
    args.disease_path = args.data_dir + args.disease_path
    args.disease_onto_path = args.data_dir + args.disease_onto_path
    args.disease_pcg_path = args.data_dir + args.disease_pcg_path
    args.hppi_path = args.data_dir + args.hppi_path
    args.mirna_path = args.data_dir + args.mirna_path
    args.mirna_disease_path = args.data_dir + args.mirna_disease_path
    args.mirna_fam_path = args.data_dir + args.mirna_fam_path
    args.mirna_pcg_path = args.data_dir + args.mirna_pcg_path
    args.pcg_path = args.data_dir + args.pcg_path
    args.neg_train_path = args.data_dir + args.neg_train_path
    args.pos_test_path = args.data_dir + args.pos_test_path
    args.neg_test_path = args.data_dir + args.neg_test_path
    args.n_runs = int(args.n_runs)

    disease_df = pd.read_csv(args.disease_path)
    n_disease = len(disease_df)
    disease_index_tensor = t.LongTensor(list(range(n_disease)))

    mirna_df = pd.read_csv(args.mirna_path)
    n_mirna = len(mirna_df)
    mirna_index_tensor = t.LongTensor(list(range(n_mirna)))

    mirna_df = pd.read_csv(args.pcg_path)
    n_pcg = len(mirna_df)
    pcg_index_tensor = t.LongTensor(list(range(n_pcg)))

    disease_onto = pd.read_csv(args.disease_onto_path).values.tolist()
    disease_pcg = pd.read_csv(args.disease_pcg_path)
    disease_pcg = disease_pcg.values.tolist()
    hppi = pd.read_csv(args.hppi_path).values.tolist()
    mirna_family = pd.read_csv(args.mirna_fam_path).values.tolist()
    mirna_pcg = pd.read_csv(args.mirna_pcg_path).values.tolist()

    pos_train_pairs, train_tensor, train_lbl_tensor = read_int_data(args.mirna_disease_path, args.neg_train_path)
    neg_test_pairs, test_tensor, test_lbl_tensor = read_int_data(args.pos_test_path, args.neg_test_path)
    test_lbl = test_lbl_tensor.detach().numpy()
    n_md = len(pos_train_pairs) + len(neg_test_pairs)

    neg_test_pairs2, test_tensor2, test_lbl_tensor2 = read_int_data(args.pos_test_path.replace('held_out1', 'new_mirna'),
                                                                    args.neg_test_path.replace('held_out1', 'new_mirna'))
    test_lbl2 = test_lbl_tensor2.detach().numpy()

    neg_test_pairs3, test_tensor3, test_lbl_tensor3 = read_int_data(args.pos_test_path.replace('held_out1', 'new_disease'),
                                                                    args.neg_test_path.replace('held_out1', 'new_disease'))
    test_lbl3 = test_lbl_tensor3.detach().numpy()

    neg_test_pairs4, test_tensor4, test_lbl_tensor4 = read_int_data(args.pos_test_path.replace('held_out1', 'held_out2'),
                                                                    args.neg_test_path.replace('held_out1', 'held_out2'))
    test_lbl4 = test_lbl_tensor4.detach().numpy()


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
        ppi_edgeweight.append(p[2])
        ppi_edgelist.append([idx2, idx1])
        ppi_edgeweight.append(p[2])


    for p in mirna_pcg:
        idx1 = p[0]
        idx2 = p[1]
        mirna_pcg_pairs.append([idx1,idx2])
        mirna_pcg_weight.append(p[2])

    for p in disease_pcg:
        idx1 = p[0]
        idx2 = p[1]
        disease_pcg_pairs.append([idx1,idx2])
        disease_pcg_weight.append(p[2])#/max_dpcg)

    n_dp = len(disease_pcg_pairs)
    n_mp = len(mirna_pcg_pairs)

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
    # l1loss = t.nn.KLDivLoss()#t.nn.BCELoss()# t.nn.L1Loss()


    args.epochs = int(args.epochs)
    lrs = [0.001]
    if torch.cuda.is_available():
        mirna_index_tensor = mirna_index_tensor.cuda()
        disease_index_tensor = disease_index_tensor.cuda()
        pcg_index_tensor = pcg_index_tensor.cuda()
        mirna_edgelist = mirna_edgelist.cuda()
        mirna_edgeweight = mirna_edgeweight.cuda()
        disease_edgelist = disease_edgelist.cuda()
        disease_edgeweight = disease_edgeweight.cuda()
        ppi_edgelist = ppi_edgelist.cuda()
        ppi_edgeweight = ppi_edgeweight.cuda()
        mirna_pcg_pairs = mirna_pcg_pairs.cuda()
        disease_pcg_pairs = disease_pcg_pairs.cuda()
        train_tensor = train_tensor.cuda()
        train_lbl_tensor = train_lbl_tensor.cuda()
        mirna_pcg_weight = mirna_pcg_weight.cuda()
        disease_pcg_weight = disease_pcg_weight.cuda()
        test_tensor = test_tensor.cuda()
        test_tensor2 = test_tensor2.cuda()
        test_tensor3 = test_tensor3.cuda()
        test_tensor4 = test_tensor4.cuda()
        criterion = criterion.cuda()
        l1loss = l1loss.cuda()

    for lr in lrs:

        all_scores = list()
        for irun in range(args.n_runs):
            for ibeta in range(1):
                w1 = 1.0
                w2 = 1.0
                w3 = 1.0
                model = MuCoMiD(32, n_mirna, n_disease, n_pcg, 32)
                if torch.cuda.is_available():
                    model = model.cuda()
                optimizer = t.optim.Adam(model.parameters(), lr=lr)
                if torch.cuda.is_available():
                    model = model.cuda()

                for epoch in range(0,args.epochs):
                    model.train()
                    model.zero_grad()
                    assoc_out, mirna_pcg_out, disease_pcg_out = model(mirna_index_tensor, disease_index_tensor, pcg_index_tensor, mirna_edgelist, mirna_edgeweight, disease_edgelist, disease_edgeweight, ppi_edgelist,
                                                                      ppi_edgeweight, mirna_pcg_pairs, disease_pcg_pairs, train_tensor)
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
                    print('Epoch: ', epoch, ' loss: ', loss_val/train_lbl_tensor.size(0))

                    if epoch%10 == 0:
                        model.eval()
                        pred_score, _, _ = model(mirna_index_tensor, disease_index_tensor, pcg_index_tensor, mirna_edgelist, mirna_edgeweight, disease_edgelist, disease_edgeweight, ppi_edgelist,
                                                 ppi_edgeweight, mirna_pcg_pairs, disease_pcg_pairs, test_tensor)
                        pred_score = pred_score.detach().numpy() if not torch.cuda.is_available() else pred_score.cpu().detach().numpy()

                        test_pred_lbl = pred_score.tolist()
                        test_pred_lbl = [item[0] if type(item) == list else item for item in test_pred_lbl]
                        auc_score, ap_score = get_score(test_lbl, test_pred_lbl)
                        print('auc_score:%.4f, ap_score:%.4f' %(auc_score, ap_score))
            model.eval()
            pred_score, _, _ = model(mirna_index_tensor, disease_index_tensor, pcg_index_tensor, mirna_edgelist, mirna_edgeweight, disease_edgelist, disease_edgeweight, ppi_edgelist,
                                     ppi_edgeweight, mirna_pcg_pairs, disease_pcg_pairs, test_tensor)
            pred_score = pred_score.detach().numpy() if not torch.cuda.is_available() else pred_score.cpu().detach().numpy()

            test_pred_lbl = pred_score.tolist()
            test_pred_lbl = [item[0] if type(item) == list else item for item in test_pred_lbl]

            auc_score, ap_score = get_score(test_lbl, test_pred_lbl)
            print('held_out1 auc_score:%.4f, ap_score:%.4f' %(auc_score, ap_score))
            pred_score, _, _ = model(mirna_index_tensor, disease_index_tensor, pcg_index_tensor, mirna_edgelist, mirna_edgeweight, disease_edgelist, disease_edgeweight, ppi_edgelist,
                                     ppi_edgeweight, mirna_pcg_pairs, disease_pcg_pairs, test_tensor2)
            pred_score = pred_score.detach().numpy() if not torch.cuda.is_available() else pred_score.cpu().detach().numpy()

            test_pred_lbl = pred_score.tolist()
            test_pred_lbl = [item[0] if type(item) == list else item for item in test_pred_lbl]

            auc_score2, ap_score2 = get_score(test_lbl2, test_pred_lbl)
            print('novel_mirna auc_score:%.4f, ap_score:%.4f' %(auc_score2, ap_score2))

            pred_score, _, _ = model(mirna_index_tensor, disease_index_tensor, pcg_index_tensor, mirna_edgelist, mirna_edgeweight, disease_edgelist, disease_edgeweight, ppi_edgelist,
                                     ppi_edgeweight, mirna_pcg_pairs, disease_pcg_pairs, test_tensor3)
            pred_score = pred_score.detach().numpy() if not torch.cuda.is_available() else pred_score.cpu().detach().numpy()

            test_pred_lbl = pred_score.tolist()
            test_pred_lbl = [item[0] if type(item) == list else item for item in test_pred_lbl]

            auc_score3, ap_score3 = get_score(test_lbl3, test_pred_lbl)
            print('novel_disease auc_score:%.4f, ap_score:%.4f' %(auc_score3, ap_score3))

            pred_score, _, _ = model(mirna_index_tensor, disease_index_tensor, pcg_index_tensor, mirna_edgelist, mirna_edgeweight, disease_edgelist, disease_edgeweight, ppi_edgelist,
                                     ppi_edgeweight, mirna_pcg_pairs, disease_pcg_pairs, test_tensor4)
            pred_score = pred_score.detach().numpy() if not torch.cuda.is_available() else pred_score.cpu().detach().numpy()

            test_pred_lbl = pred_score.tolist()
            test_pred_lbl = [item[0] if type(item) == list else item for item in test_pred_lbl]
            auc_score4, ap_score4 = get_score(test_lbl4, test_pred_lbl)
            print('held_out2 auc_score:%.4f, ap_score:%.4f' %(auc_score4, ap_score4))

            all_scores.append([auc_score, ap_score, auc_score2, ap_score2, auc_score3, ap_score3, auc_score4, ap_score4])
        print('all_scores: ', all_scores)
        arr = np.array(all_scores)
        mean = list(np.mean(arr, axis=0))
        mean = [round(item, 4) for item in mean]
        print('average auc_score, ap_score: ', np.mean(arr, axis=0))
        print('Standard deviation: ')
        print('auc_score, ap_score')
        print(np.std(arr, axis=0))
        std = list(np.std(arr, axis=0))
        print('mean: ', mean)
        print('std: ', std)
        return all_scores, mean, std

if __name__ == "__main__":
    writer = open('results/ind_mtt.csv', 'w')
    writer.write('run,held_out1 auc, aupr, new_mirna auc, aupr, new_disease auc, aupr, held_out2 auc, aupr\n')

    all_scores, mean, std = main()
    for i, scorelist in enumerate(all_scores):
        writer.write(str(i))
        for score in scorelist:
            writer.write(',' + str(round(score,5)))
        writer.write('\n')
    mean = [str(round(item,4)) for item in mean]
    std = [str(round(item,4)) for item in std]
    # for score in all_scores
    writer.write('Average,' + ','.join(mean) + '\n')
    writer.write('std,' + ','.join(std) + '\n')
    writer.close()

    case_studies()





