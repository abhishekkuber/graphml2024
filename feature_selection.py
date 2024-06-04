from ReliefF import ReliefF
import numpy as np
import pandas as pd

def relieff(num_features, X,y, save_path='data/relieff_all.csv'):
    fs = ReliefF(n_neighbors=20, n_features_to_keep=num_features)
    fs.fit(X, y)
    save_df = pd.DataFrame(np.array(fs.top_features))
    save_df.to_csv(save_path, header=False, index=False)

# disease_pcg_path = 'data/rdisease_pcg1.csv'
disease_pcg_path = 'data/rdisease_pcg10.csv'
disease_cat_path = 'data/disease_cat.csv'
feat_df = pd.read_csv(disease_pcg_path, header=None).values
n_pcg = feat_df.shape[1]

lbl_df = pd.read_csv(disease_cat_path)
lbl_df.columns = ['label']
raw_labels = lbl_df['label'].values.tolist()
raw_labels = np.array(raw_labels)
print(raw_labels)

relieff(n_pcg, feat_df, raw_labels, save_path='relieff_rpcg10.csv')

disease_pcg_path = 'data/rdisease_pcg2.csv'
feat_df = pd.read_csv(disease_pcg_path, header=None).values

relieff(n_pcg, feat_df, raw_labels, save_path='relieff_rpcg2.csv')



