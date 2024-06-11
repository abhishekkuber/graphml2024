# this file is used to generate the embeddings for miRNA, disease and PCG

from utils import *
import numpy as np
import sdne as sdne

def get_embedding(vectors: dict, x):
    matrix = np.zeros((
        x,
        len(list(vectors.values())[0])
    ))
    for key, value in vectors.items():
        matrix[int(key), :] = value
    return matrix

def Get_embedding_Matrix(mirna_disease, mirna_pcg, disease_pcg, pcg_pcg, mirna_mirna, disease_disease, n_mirna, n_disease, n_pcg):
    graph1 = sdne.Graph()
    graph1.add_edgelist(mirna_disease, mirna_pcg, disease_pcg, pcg_pcg, mirna_mirna, disease_disease, n_mirna, n_disease, n_pcg)
    model = sdne.SDNE(graph1, [1000, 32]) #32 is the number of features per node
    return get_embedding(model.vectors, n_mirna + n_disease + n_pcg)

def main():
    #select 100 most important PCGs
    fs_df = pd.read_csv('data/original_data/relieff_raw_pcg.csv', header=None).values.tolist()
    all_selected_feats = [item[0] for item in fs_df]
    selected_feats = all_selected_feats[:100]

    #get and process miRNA-miRNA relationship
    mirna_mirna = pd.read_csv('data/original_data/mirna_fam_pos.csv').values.tolist()

    #get and process disease-disease relationship
    disease_disease = pd.read_csv('data/original_data/disease_onto_pos.csv').values.tolist()

    #get and process PCG-PCG relationship
    ppi_mat = pd.read_csv('data/original_data/hppi.csv')
    ppi_mat = ppi_mat[(ppi_mat['hprot1'].isin(selected_feats)) & (ppi_mat['hprot2'].isin(selected_feats))]
    pcg_pcg = [[selected_feats.index(int(item[0])), selected_feats.index(int(item[1]))] for item in ppi_mat.values.tolist()]
    np.savetxt("data/generated_data/pcg_pcg.csv", pcg_pcg, delimiter=",")

    #get and process miRNA-PCG relationship
    mirna_pcg_mat = pd.read_csv('data/original_data/rmirna_pcg10.csv', header=None)
    mirna_pcg_mat = pd.DataFrame(mirna_pcg_mat.iloc[:, selected_feats]).values.tolist()
    mirna_pcg = list()
    for i, rec in enumerate(mirna_pcg_mat):
            for j, item in enumerate(rec):
                if item != 0:
                    mirna_pcg.append([i,j])
    np.savetxt("data/generated_data/mirna_pcg.csv", mirna_pcg, delimiter=",")

    #get and process disease-PCG relationship
    disease_pcg_mat = pd.read_csv('data/original_data/rdisease_pcg10.csv', header=None)
    disease_pcg_mat = pd.DataFrame(disease_pcg_mat.iloc[:, selected_feats]).values.tolist()
    disease_pcg = list()
    for i, rec in enumerate(disease_pcg_mat):
            for j, item in enumerate(rec):
                if item != 0:
                    disease_pcg.append([i,j])
    np.savetxt("data/generated_data/disease_pcg.csv", disease_pcg, delimiter=",")

    #get and process miRNA-disease relationship
    mirna_diease = pd.read_csv('data/training_data/hmdd2_pos.csv').values.tolist()

    #get the number of miRNA, disease and PCG
    n_mirna = len(mirna_pcg_mat)
    n_disease = len(disease_pcg_mat)
    n_pcg = 100

    #generate the embeddings
    embeddings = Get_embedding_Matrix(mirna_diease, mirna_pcg, disease_pcg, pcg_pcg, mirna_mirna, disease_disease, n_mirna, n_disease, n_pcg)

    mirna_emb = np.array(embeddings[0:n_mirna, 0:])
    disease_emb = np.array(embeddings[n_mirna:n_mirna+n_disease, 0:])
    pcg_emb = np.array(embeddings[n_mirna+n_disease:n_mirna+n_disease+n_pcg, 0:])

    #save the embeddings into csv files
    np.savetxt("data/generated_data/mirna_emb.csv", mirna_emb, delimiter=",")
    np.savetxt("data/generated_data/disease_emb.csv", disease_emb, delimiter=",")
    np.savetxt("data/generated_data/pcg_emb.csv", pcg_emb, delimiter=",")

if __name__ == "__main__":
    main()