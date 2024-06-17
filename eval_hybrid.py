# This file evaluates the performance of the hybrid model (mpm+cmtt)
from matplotlib import pyplot as plt
from model import *
from utils import *
import torch as t
import pandas as pd
import os

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def read_int_data(pos_path, neg_path):
    pos_df = pd.read_csv(pos_path).values.tolist()
    neg_df = pd.read_csv(neg_path).values.tolist()
    int_edges = pos_df + neg_df
    int_lbl = [1] * len(pos_df) + [0] * len(neg_df)
    return pos_df, t.LongTensor(int_edges), t.FloatTensor(int_lbl)

def objectives(loss0, loss1, loss2):
    return np.array([loss0.item(), loss1.item(), loss2.item()])

def check_pareto_dom(s1, s2):
    if np.all(s1 <= s2) and np.any(s1 < s2):
        return True
    return False

def get_pareto_front(losses, weights, num_samples=100):
    solutions = np.random.uniform(size=(num_samples, len(weights)))
    pareto_front = []
    for solution in solutions:
        scores = np.dot(losses, solution)
        if not any(check_pareto_dom(scores, np.dot(losses, other)) for other in solutions):
            pareto_front.append((solution, scores))
    return pareto_front

def main():
    criterion = t.nn.BCELoss()
    l1loss = t.nn.MSELoss()
    device = get_device()
    data = load_data(device)

    criterion = criterion.to(device)
    l1loss = l1loss.to(device)

    pos_test_path = 'data/test_data/new_mirna_pos.csv'
    neg_test_pre = pos_test_path.replace('_pos.csv', '_neg.csv')
    datasrc = pos_test_path[pos_test_path.rfind('/') + 1:].replace('_pos.csv', '')

    def eval(model, neg_rates, datasrc=datasrc, print_results=True):
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
                assoc_out, mirna_pcg_out, disease_pcg_out = model(data, test_tensor)

                auc_score, ap_score, sn, sp, acc, prec, rec, f1, mcc = get_all_score(test_lbl, assoc_out.detach().cpu().numpy())
                tmp_score = [auc_score, ap_score, sn, sp, acc, prec, rec, f1]
                for ftmp in tmp_score:
                    cur_score.append(str(round(ftmp, 5)))
                if print_results: print(cur_score)
                all_scores.append(cur_score)

                avg_acc += acc
                tloss0, tloss1, tloss2 = criterion(assoc_out, test_lbl), l1loss(mirna_pcg_out, data["mirna_pcg_weight"]), l1loss(disease_pcg_out, data["disease_pcg_weight"])
                tloss = w1 * tloss0 + w2 * tloss1 + w3 * tloss2
                test_loss = tloss.item()
                avg_loss += test_loss

            avg_acc /= 10
            avg_loss /= 10

        return all_scores, avg_acc, avg_loss

    w1 = 1.0
    w2 = 1.0
    w3 = 1.0
    model = MuCoMiD(32, 32).to(device)
    optimizer = t.optim.Adam(model.parameters(), lr=0.001)

    train_losses, test_losses, test_accuracies = list(), list(), list()

    for epoch in range(0, 200):
        model.train()
        model.zero_grad()
        assoc_out, mirna_pcg_out, disease_pcg_out = model(data, data["train_tensor"])
        loss0 = criterion(assoc_out, data["train_lbl_tensor"])
        loss1 = l1loss(mirna_pcg_out, data["mirna_pcg_weight"])
        loss2 = l1loss(disease_pcg_out, data["disease_pcg_weight"])
        loss = w1 * loss0 + w2 * loss1 + w3 * loss2

        current_objectives = objectives(loss0, loss1, loss2)
        pareto_front = get_pareto_front(current_objectives, np.array([w1, w2, w3]))
        w1 = pareto_front[0][0][0]
        w2 = pareto_front[0][0][1]
        w3 = pareto_front[0][0][2]

        loss.backward()
        optimizer.step()
        loss_val = loss.item()

        print('Epoch: ', epoch, ' loss: ', loss_val / data["train_lbl_tensor"].size(0))

        if epoch % 1 == 0:
            train_losses.append(loss_val)
            all_scores, avg_acc, avg_loss = eval(model, neg_rates=[5], print_results=False)
            test_losses.append(avg_loss)
            test_accuracies.append(avg_acc)
            print('Test loss: ', avg_loss, ' Test acc: ', avg_acc)

    save_hybrid_model(model)
    ensure_dir('plots/pareto')

    # Improved plotting section
    plt.figure()
    plt.plot(train_losses, label='Training Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Test Loss over Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig('plots/pareto/pareto_loss_curves.png')

    plt.figure()
    plt.plot(test_accuracies, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Test Accuracy over Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig('plots/pareto/pareto_test_acc.png')

    np.save('plots/pareto/pareto_train_losses.npy', train_losses)
    np.save('plots/pareto/pareto_test_losses.npy', test_losses)
    np.save('plots/pareto/pareto_test_accuracies.npy', test_accuracies)

    all_scores, _, _ = eval(model, neg_rates=[1, 5, 10])
    ensure_dir('data/eval_results')

    writer = open('data/eval_results/hybrid_output.txt', 'w+')
    writer.write('data,run,auc_score, ap_score, sn, sp, acc, prec, rec, f1\n')
    for line in all_scores:
        writer.write(','.join(line))
        writer.write('\n')
    writer.close()

if __name__ == "__main__":
    main()
