import codecs
import pickle

import numpy as np
import pandas as pd
from PyBioMed.PyMolecule.PubChemFingerprints import calcPubChemFingerAll
from rdkit import Chem
from rdkit.Chem import AllChem
from subword_nmt.apply_bpe import BPE

# drug_smiles is an array where elements are [puchem_id, smiles]
with open('../all_drug_smiles.pkl', 'rb') as f:
    drug_smiles = pickle.load(f)

# espf
vocab_path = '../drug_codes_chembl_freq_1500.txt'
sub_csv = pd.read_csv('../subword_units_map_chembl_freq_1500.csv')
bpe_codes_drug = codecs.open(vocab_path)
dbpe = BPE(bpe_codes_drug, merges=-1, separator='')

idx2word_d = sub_csv['index'].values
words2idx_d = dict(zip(idx2word_d, range(0, len(idx2word_d))))


def smiles2espf(x):
    t1 = dbpe.process_line(x[1]).split()  # split
    try:
        i1 = np.asarray([words2idx_d[i] for i in t1])  # index
        index = 0
    except:
        i1 = np.array([0])
        index = 1
        print('false')
    v1 = np.zeros(len(idx2word_d))
    v1[i1] = 1
    return v1, index


def smiles2morgan(s):
    try:
        # 将 SMILES 转化为分子对象
        mol = Chem.MolFromSmiles(s[1])
        # 计算摩根指纹
        fpgen = AllChem.GetMorganGenerator(radius=2)
        features = fpgen.GetFingerprintAsNumPy(mol)
        index = 0
    except:
        print('rdkit not found this smiles for morgan: ' + s[1] + ' convert to all 0 features')
        features = np.zeros((2048,))
        index = 1
    return features, index


def smiles2pubchem(s):
    try:
        mol = Chem.MolFromSmiles(s[1])
        features = calcPubChemFingerAll(mol)
        index = 0
    except:
        print('pubchem fingerprint not working for smiles: ' + s[1] + ' convert to 0 vectors')
        print(s)
        features = np.zeros((881,))
        index = 1
    return np.array(features), index


drug_morgan = {}
drug_pubchem = {}
drug_espf = {}

false1, false2, false3 = [], [], []

for i in range(len(drug_smiles)):
    drug_id = drug_smiles[i][0]
    drug_morgan[drug_id], index1 = smiles2morgan(drug_smiles[i])
    drug_pubchem[drug_id], index2 = smiles2pubchem(drug_smiles[i])
    drug_espf[drug_id], index3 = smiles2espf(drug_smiles[i])

    if index1 == 1:
        false1.append(drug_smiles[i][0])
        del drug_morgan[drug_id]
    if index2 == 1:
        false2.append(drug_smiles[i][0])
        del drug_pubchem[drug_id]
    if index3 == 1:
        false3.append(drug_smiles[i][0])
        del drug_espf[drug_id]

false_total = []
for i in range(3):
    false_total.extend(false1)
    false_total.extend(false2)
    false_total.extend(false3)

total_false = list(set(false_total))

lc = 'false_encoding_drug.pkl'
pickle_file = open(lc, 'wb')
pickle.dump(total_false, pickle_file)
pickle_file.close()

lc = 'morgan_encoding.pkl'
pickle_file = open(lc, 'wb')
pickle.dump(drug_morgan, pickle_file)
pickle_file.close()

lc = 'pubchem_encoding.pkl'
pickle_file = open(lc, 'wb')
pickle.dump(drug_pubchem, pickle_file)
pickle_file.close()

lc = 'espf_encoding.pkl'
pickle_file = open(lc, 'wb')
pickle.dump(drug_espf, pickle_file)
pickle_file.close()

print('finished')
