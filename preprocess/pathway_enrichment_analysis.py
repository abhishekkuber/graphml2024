import pandas as pd
from reactome2py import analysis

# perform pathway enrichment analysis for the set of given genes
# save the pathay way id2name in the pathway_id2name variable
# also keep track of whether a particular pathway appearred in the associated pathway for any known disease (one of the feature return from Reactome)
def pathway_enrichment(genes, pathway_id2name, pathway_id2inMol):
    result = analysis.identifiers(ids=','.join(genes), species='9606', p_value=0.05, projection=True, page=-1, sort_by='ENTITIES_PVALUE')
    res = list()
    print(len(result['pathways']))
    for item in result['pathways']:
        pathway = item['stId']
        name = item['name']
        inDisease = item['inDisease']
        pvalue = item['entities']['pValue']
        res.append([pathway, pvalue])
        pathway_id2name[pathway] = name
        pathway_id2inMol[pathway] = inDisease
    print('returned pathway: ', len(res))
    return res

# test code
mirna_pcg = pd.read_csv('data/mirna_pcg.csv', header=None).values.tolist()
pcgs = pd.read_csv('data/pcg.csv').values.tolist()
idx2pcg = {i:item[0] for i, item in enumerate(pcgs)}
genes = [idx2pcg[idx] for idx, item in enumerate(mirna_pcg[0]) if item != 0]
pathwayid2name = {}
pathwayid2Disease= {}
res = pathway_enrichment(genes, pathwayid2name, pathwayid2Disease)
print(res)
print(pathwayid2name)
print(pathwayid2Disease)