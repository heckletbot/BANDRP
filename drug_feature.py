import os
import pickle

import deepchem as dc
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors


def drug_feature(smile):
    mol = Chem.MolFromSmiles(smile)
    featurizer = dc.feat.MolGraphConvFeaturizer(use_edges=True)
    mol_object = featurizer.featurize(mol)
    try:
        # 原子特征30维 边特征11维 边以两个向量表示，两个向量相同位置的值表示两个原子之间存在化学键
        atom_feature = mol_object[0].node_features
        edge_feature = mol_object[0].edge_features
        edge_list = mol_object[0].edge_index
        return [atom_feature, edge_feature, edge_list]
    except:
        print(smile)
        return False


# 药物文件路径和保存文件的路径，药物文件应有一列名为smiles序列
def get_drug_feature(drug_file, save_path):
    drug_df = pd.read_csv(drug_file,index_col=0)
    drug = drug_df.drop_duplicates(subset=["smiles"])
    #pre_process.to_csv('C:/Users/hp/Documents/WeChat Files/wxid_juc0ikbw516922/FileStorage/File/2022-11/Drug_smiles.csv')
    # with open(drug_file, 'rb') as f:
    #     pre_process = pickle.load(f)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for index,row in drug.iterrows():
        if drug_feature(row['smiles']) == False:
            continue
        else:
            features = drug_feature(row['smiles'])
            atom_feature = features[0]
            edge_feature = features[1]
            edge_list = features[2]
        save_dir = '%s/%s.pkl' % (save_path, row['drug_id'])
        pickle.dump([atom_feature, edge_feature, edge_list], open(save_dir, 'wb'))


def calcproperties(smiles_list):
    # 14个性质的计算
    df = pd.DataFrame(
        columns=['drug_id', 'mol_weight', 'XLogP', 'HydrogenBondDonorCount', 'HydrogenBondAcceptorCount',
                 'PolarSurfaceArea', 'NumRings', 'RotatableBondCount', 'Refractivity', 'FormalCharge',
                 'RO5', 'Ghose Filter', 'Veber Rule'
                 ])
    finger_print = {}
    for smiles in smiles_list:
        # 将 SMILES 转化为分子对象
        mol = Chem.MolFromSmiles(smiles[1])
        # 计算摩根指纹
        fpgen = AllChem.GetMorganGenerator(radius=2)
        fp = fpgen.GetFingerprintAsNumPy(mol)
        finger_print[smiles[0]] = fp
        # 计算全部能直接计算的性质
        vals = Descriptors.CalcMolDescriptors(mol)
        formal_charge = Chem.GetFormalCharge(mol)
        # 计算各个物理化学性质
        smile_property = []
        smile_property.append(smiles[0])
        smile_property.append(vals['MolWt'])
        smile_property.append(vals['MolLogP'])
        smile_property.append(vals['NumHDonors'])
        smile_property.append(vals['NumHAcceptors'])
        smile_property.append(vals['TPSA'])
        smile_property.append(vals['RingCount'])
        smile_property.append(vals['NumRotatableBonds'])
        smile_property.append(vals['MolMR'])
        smile_property.append(formal_charge)

        # 计算 RO5 的参数
        MW = Descriptors.MolWt(mol)
        HBA = Descriptors.NOCount(mol)
        HBD = Descriptors.NHOHCount(mol)
        LogP = Descriptors.MolLogP(mol)

        # 计算 Ghose Filter 的参数
        num_rot_bonds = Descriptors.NumRotatableBonds(mol)
        num_atoms = mol.GetNumAtoms()
        mol_formula = Chem.rdMolDescriptors.CalcMolFormula(mol)

        # 计算 Veber Rule 的参数
        psa = Descriptors.TPSA(mol)

        # 计算 wQED 的参数
        formal_charge = Chem.GetFormalCharge(mol)
        logp = Descriptors.MolLogP(mol)
        hbd = Descriptors.NumHDonors(mol)
        hba = Descriptors.NumHAcceptors(mol)
        # 计算 RO5
        conditions = [MW <= 500, HBA <= 10, HBD <= 5, LogP <= 5]
        ro5_pass = 0
        if conditions.count(True) >= 3:
            ro5_pass = 1

        # 计算 Ghose Filter
        ghose_pass = 1
        if num_rot_bonds <= 10 and 20 <= num_atoms <= 70 and 0.25 <= MW / num_atoms <= 2.0 and 0.1 <= mol_formula.count(
                'C') / num_atoms <= 0.9:
            ghose_pass = 0

        # 计算 Veber Rule
        veber_pass = 1
        if num_rot_bonds <= 10 and psa <= 140 and formal_charge == 0:
            veber_pass = 0

        smile_property.append(ro5_pass)
        smile_property.append(ghose_pass)
        smile_property.append(veber_pass)
        df.loc[len(df)] = smile_property

    # 全部性质计算
    mols = [Chem.MolFromSmiles(smiles[1]) for smiles in smiles_list]
    descrs = [Descriptors.CalcMolDescriptors(mol) for mol in mols]
    df_all = pd.DataFrame(descrs)
    df_all.insert(0, 'drug_id', df['drug_id'])

    return df, df_all, finger_print


if __name__ == '__main__':
    # 构图特征计算
    get_drug_feature('D:/drug_response_predict/drug_smiles.csv',
                     'D:/drug_response_predict/smiles')

    # 物理特征计算
    # with open('C:/Users/hp/Documents/WeChat Files/wxid_juc0ikbw516922\FileStorage\File/2023-06/ALL_smiles.pickle', 'rb') as f:
    #     file = pickle.load(f)
    #
    #d = pd.read_csv('D:/drug_response_predict/data/Drug/GDSC1_drug.csv')
    # smiles_list = []
    # for id, smiles in zip(d['drug_id'], d['IsomericSMILES']):
    #     smiles_list.append([id, smiles])
    #drug_id = list(range(len(file)))
    # smiles_list = []
    # for id, smiles in zip(drug_id, file):
    #     smiles_list.append([id, smiles])
    # df, df_all, fps = calcproperties(smiles_list)
    # df.to_csv('D:/property.csv', index=False)
    #df_all.to_csv('D:/drug_response_predict/data/Drug/GDSC1_all_property.csv', index=False)
    #with open('D:/drug_response_predict/data/Drug/GDSC1_MorganFP.pkl', 'wb') as f:
    #    pickle.dump(fps, f)
