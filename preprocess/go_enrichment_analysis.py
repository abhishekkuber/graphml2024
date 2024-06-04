import pandas as pd
from goscripts import enrichment_stats
from goscripts import gaf_parser
from goscripts import obo_tools
import os

# GO = Gene Ontology
def go_ontology_loading():
    """
    Load the gene association file and the GO file, and build the GO tree.
    """
    gafDict = gaf_parser.importGAF('data/goa_human.gaf', False)
    GOterms = obo_tools.importOBO('data/go-basic.obo', ignore_part_of=True)
    root_nodes = obo_tools.set_namespace_root('all')

    obo_tools.buildGOtree(GOterms, root_nodes)

    # Filter GO terms from gaf dictionaries if they are missing from the GO
    print(
        'Looking for inconsistencies between the gene association and GO files...\n'
    )
    to_remove = set()
    for gene in gafDict:
        for term in gafDict[gene]:
            if term not in GOterms and term not in to_remove:
                to_remove.add(term)
                print(
                    f'{term} was missing from GO file.\nRemoving term from gene assocation file...\n'
                )
    if to_remove:
        gafDict = {
            gene: terms
            for gene, terms in gafDict.items() if gene not in to_remove
        }

    # Perform enrichment test
    print(
        'Finished completing ontology...proceeding with enrichment tests...\n')
    return gafDict, GOterms, to_remove

def get_go(uniprots, gafDict, GOterms, to_remove):
    """
    Perform GO enrichment analysis for a given set of uniprot IDs.

    :param uniprots: The set of uniprot IDs to perform the analysis on.
    :param gafDict:  The gene association file dictionary.
    :param GOterms:  The GO terms dictionary.
    :param to_remove:  The set of GO terms to remove from the analysis.
    :return:  A dataframe with the tested GO terms and results.
    """
    # Generate a gene association file for the (pruned) subset of interest too.
    gafSubset = gaf_parser.createSubsetGafDict(uniprots, gafDict)
    if to_remove:
        gafSubset = {
            gene: terms
            for gene, terms in gafSubset.items() if gene not in to_remove
        }

    enrichmentResults = enrichment_stats.enrichmentAnalysis(
        GOterms,
        gafDict,
        gafSubset)
    if len(enrichmentResults['pValues']) == 0:
        return None
    # Update results with multiple testing correction
    enrichment_stats.multipleTestingCorrection(enrichmentResults)

    # Create dataframe with tested GO terms and results
    output = enrichment_stats.annotateOutput(enrichmentResults, GOterms,
                                             gafDict, gafSubset)
    return output # dataframe columns: ['GO id', 'GO name', 'GO namespace', 'p-value', 'corrected p-value', 'cluster freq', 'background freq']

# perform GO enrichment analysis for all miRNA/disease in the
# input pcg_profile_path and save the profile for each miRNA/disease to one file in tmp_dir
def go_annotation(pcg_profile_path, pcg_dict_path, mol_dict_path, tmp_dir):
    """
    Perform GO enrichment analysis for all miRNA/disease in the input pcg_profile_path and save the profile for each miRNA/disease to one file in tmp_dir
    :param pcg_profile_path:  the path to the pcg profile file
    :param pcg_dict_path:  the path to the pcg dictionary file
    :param mol_dict_path:  the path to the miRNA/disease dictionary file
    :param tmp_dir:  the directory to save the GO enrichment analysis results
    :return:
    """
    gafDict, GOterms, to_remove = go_ontology_loading()

    pcgs = pd.read_csv(pcg_dict_path).values.tolist()
    idx2pcg = {i:item[0] for i, item in enumerate(pcgs)}

    molecules = pd.read_csv(mol_dict_path).values.tolist()
    idx2mol = {i:item[0] for i, item in enumerate(molecules)}

    profiles = pd.read_csv(pcg_profile_path, header=None).values.tolist()
    gene2uniprot = pd.read_csv('data/gene2uniprot.tab', sep='\t').values.tolist()
    gene2uniprot = {item[0]:item[1] for item in gene2uniprot}

    cols = ['GO id', 'GO name', 'p-value', 'corrected p-value']
    for idx, profile in enumerate(profiles):
        mol = idx2mol[idx] # the miNRA name or disease MESH id
        if os.path.exists(tmp_dir + mol + '.csv'):
            continue
        print('processing item', idx, ': ', mol)
        genes = [idx2pcg[idx] for idx, item in enumerate(profile) if item != 0]
        print(len(genes))
        uniprots = [gene2uniprot[item] for item in genes if item in gene2uniprot]
        go_df = get_go(set(uniprots), gafDict, GOterms, to_remove)
        if str(go_df) != 'None':
            go_df = go_df[cols]
            go_df = go_df[go_df['corrected p-value'] <= 0.05]
            go_df.to_csv(tmp_dir + mol + '.csv', index=False)

mirna_pcg = 'data/mirna_pcg.csv'
tmpdir = 'data/tmp/'
go_annotation(mirna_pcg, 'data/pcg.csv', 'data/mirna.csv', tmpdir)

