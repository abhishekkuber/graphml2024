import random
import pandas as pd
import numpy as np

def save(vals, save_path, columns = None, drop_duplicates=True):
    if columns == None:
        df = pd.DataFrame(np.array(vals))
        if drop_duplicates:
            df.drop_duplicates(inplace=True)
        df.to_csv(save_path, index=False, header=False)
    else:
        df = pd.DataFrame(np.array(vals), columns = columns)
        if drop_duplicates:
            df.drop_duplicates(inplace=True)
        df.to_csv(save_path, index=False)
    return df

def gen_assoc_dict(inpath,key_col,val_col):
    df = pd.read_csv(inpath).values.tolist()
    res = dict()
    for p in df:
        if p[key_col] not in res:
            res[p[key_col]] = list()
        res[p[key_col]].append(p[val_col])
    return res

def neg_sampling(inlist, n_negative, columns, save_path):
    indexes = list(range(len(inlist)))
    random.shuffle(indexes)
    save_list = [inlist[idx] for idx in indexes[:n_negative]]
    save(save_list, save_path, columns=columns)

# nagative sampling
def assoc_negative_sampling(pos_path, save_path, neg_rate=1):
    all_known_dict = gen_assoc_dict('data/all_assoc_pos.csv', 1, 0)
    print('Association negative sampling')
    df = pd.read_csv(pos_path)
    diseases = df['disease'].unique().tolist()
    mirnas = df['mirna'].unique().tolist()
    neg_pool = list()

    for d in diseases:
        known_mi = all_known_dict[d]
        diff = list(np.setdiff1d(mirnas, known_mi))
        pool = [[m, d] for m in diff]
        neg_pool.extend(pool)

    indexes = list(range(len(neg_pool)))
    for iset in range(10):
        iseed = random.randint(0, 10000)
        random.seed(iseed)
        random.shuffle(indexes)
        neg_samples = [neg_pool[idx] for idx in indexes[:len(df) * neg_rate]]
        save(neg_samples, save_path.replace('.csv', str(neg_rate) + '_' + str(iset) + '.csv'), ['mirna', 'disease'])
