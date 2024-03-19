import pickle
import numpy as np
import pandas as pd


def dataload(**cfg):
    response = cfg['path']['response']
    mutation = cfg['path']['mutation']
    methylation = cfg['path']['methylation']
    expression = cfg['path']['expression']
    pathway_file = cfg['path']['pathway']
    drug_fpFile_morgan = cfg['path']['morgan']
    drug_fpFile_espf = cfg['path']['espf']
    drug_fpFile_psfp = cfg['path']['psfp']

    # response load cell_line-drug pairs
    response = pd.read_csv(response, index_col=0)
    drug_key = response.columns.values
    # pair [depmap_id pubchem_id Ln_ic50]
    pair = []
    for index, row in response.iterrows():
        for i in drug_key:
            if np.isnan(row[i]) == False:
                pair.append([index, i, row[i]])

    # cell_lines load
    mut_feature = pd.read_csv(mutation, index_col=0)
    exp_feature = pd.read_csv(expression, index_col=0)
    methy_feature = pd.read_csv(methylation, index_col=0)
    # pathway
    pathway = pd.read_csv(pathway_file, index_col=0)

    # drug
    with open(drug_fpFile_morgan, 'rb') as f:
        morgan_fp = pickle.load(f)
    with open(drug_fpFile_espf, 'rb') as f:
        espf_fp = pickle.load(f)
    with open(drug_fpFile_psfp, 'rb') as f:
        pubchem_fp = pickle.load(f)
    drug_feature = {}
    for i in drug_key:
        drug_feature[i] = [morgan_fp[int(i)], espf_fp[int(i)], pubchem_fp[int(i)]]
    return drug_feature, mut_feature, exp_feature, methy_feature, pathway, pair, response.index.values, response.columns.values

