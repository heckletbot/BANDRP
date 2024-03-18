import os
import re

import hickle as hkl
import numpy as np
import pandas as pd
import pickle


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
    with open(drug_fpFile_espf,'rb') as f:
        espf_fp = pickle.load(f)
    with open(drug_fpFile_psfp,'rb') as f:
        pubchem_fp = pickle.load(f) 
    drug_feature = {}
    for i in drug_key:
        drug_feature[i] = [morgan_fp[int(i)], espf_fp[int(i)], pubchem_fp[int(i)]]
    return drug_feature, mut_feature, exp_feature, methy_feature, pathway, pair, response.index.values, response.columns.values



def dataload_easy():
    gexpr_feature = pd.read_csv('/homec/caocheng/comparative_exp/deepcdr/data/exp.csv',index_col=0)
    methylation_feature = pd.read_csv('/homec/caocheng/comparative_exp/deepcdr/data/meth.csv',index_col=0)
    mutation_feature = pd.read_csv('/homec/caocheng/comparative_exp/deepcdr/data/mut.csv',index_col=0)
    with open('/homec/caocheng/comparative_exp/deepcdr/data/drug_feature.pkl','rb') as f:
        drug_feature = pickle.load(f)
    with open( '/homec/caocheng/comparative_exp/deepcdr/data/response.pkl','rb') as f:
        data_idx = pickle.load(f)
    with open( '/homec/caocheng/comparative_exp/deepcdr/data/fp.pkl','rb') as f:
        fp = pickle.load(f)
    pathway = pd.read_csv('/homec/caocheng/comparative_exp/deepcdr/GSVA_deepcdr_result.csv',index_col=0)
    drug_id=[]
    cell_id=[]
    for i in data_idx:
        drug_id.append(i[1])
        cell_id.append(i[0])
    drug_id = list(set(drug_id))
    cell_id = list(set(cell_id))
    return fp, mutation_feature, gexpr_feature, methylation_feature,pathway, data_idx, drug_id, cell_id

if __name__ == "__main__":
    from config import get_cfg_defaults
    cfg = get_cfg_defaults()
    drug_feature, mut_feature, exp_feature, methy_feature, pathway, pair, depmapid, pubchemid = dataload(**cfg)