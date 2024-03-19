import numpy as np
import pandas as pd
import torch.utils.data as Data
import torch
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

class my_dataloader(Data.Dataset):
    def __init__(self, drug_data, expression, mutation, methylation, pathway, pair, position):
        'Initialization'
        self.pair = pair
        self.drug_data = drug_data
        self.expression = expression
        self.mutation = mutation
        self.methylation = methylation
        self.pathway = pathway
        self.position = position

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.position)

    def __getitem__(self, index):
        'Generates one sample of data'
        index = self.pair[self.position[index]]
        drug_data = [self.drug_data[i][index[1]] for i in range(len(self.drug_data))]
        expression = self.expression[index[0]]
        mutation = self.mutation[index[0]]
        methylation = self.methylation[index[0]]
        pathway = self.pathway[index[0]]
        label = index[2]
        return drug_data, expression, mutation, methylation, pathway, label


def collate_fn(batch):
    fp1 = torch.stack([i[0][0] for i in batch], 0)
    fp2 = torch.stack([i[0][1] for i in batch], 0)
    fp3 = torch.stack([i[0][2] for i in batch], 0)
    drug_data = [fp1,fp2,fp3]

    exp = torch.stack([i[1] for i in batch], 0)
    mut = torch.stack([i[2] for i in batch], 0)
    methy = torch.stack([i[3] for i in batch], 0)
    path = torch.stack([i[4] for i in batch], 0)
    label = torch.tensor([i[5] for i in batch], dtype=torch.float32)
    return [drug_data, exp, mut, methy, path, label]


def data_process(drug_feature, mut_feature, exp_feature, methy_feature, pathway_feature, pair, cellline_id, drug_id):
    # cell_line drug ic50 pairs
    cellline_id.sort()
    drug_id.sort()
    cell_map = list(zip(cellline_id, list(range(len(cellline_id)))))
    drug_map = list(zip(drug_id, list(range(len(drug_id)))))
    cell_dict = {i[0]: i[1] for i in cell_map}
    drug_dict = {i[0]: i[1] for i in drug_map}
    all_pairs = []
    for i in pair:
        all_pairs.append([cell_dict[i[0]], drug_dict[i[1]], i[2]])

    # drug_feature
    drug_feature_num = len(drug_feature[drug_id[0]])
    drug_feature_df = pd.DataFrame(index = drug_id,columns=list(range(drug_feature_num)))
    for index in drug_id:
        for j in range(drug_feature_num):
            drug_feature_df.loc[index,j] = drug_feature[index][j]
    drug_data = [torch.from_numpy(np.array(list(drug_feature_df.iloc[:, i]), dtype='float32')) for i in range(drug_feature_num)]
   
    # cell lines feature
    # mutation expression methylation
    mutation = mut_feature.loc[cellline_id]
    expression = exp_feature.loc[cellline_id]
    methylation = methy_feature.loc[cellline_id]

    pathway = pathway_feature.loc[cellline_id]
    mutation = torch.from_numpy(np.array(mutation, dtype='float32'))
    expression = torch.from_numpy(np.array(expression, dtype='float32'))
    methylation = torch.from_numpy(np.array(methylation, dtype='float32'))
    pathway = torch.from_numpy(np.array(pathway, dtype='float32'))

    # compile train and test
    params = {'batch_size': 128,
              'shuffle': True,
              'num_workers': 4,
              'drop_last': False,
              'collate_fn': collate_fn}

    train_index, temp_index = train_test_split(range(len(pair)), test_size=0.2, random_state=42)
    val_index, test_index = train_test_split(temp_index, test_size=0.5, random_state=42)
    train_set=Data.DataLoader(my_dataloader(drug_data, expression, mutation, methylation, pathway, all_pairs, train_index),
            **params)
    test_set= Data.DataLoader(my_dataloader(drug_data, expression, mutation, methylation, pathway, all_pairs, test_index),
            **params)
    val_set= Data.DataLoader(my_dataloader(drug_data, expression, mutation, methylation, pathway, all_pairs, val_index),
            **params)
    return train_set, test_set, val_set

