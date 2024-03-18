from yacs.config import CfgNode as CN

_C = CN()

# data path
_C.path = CN()
rootpath = "/homec/caocheng/drug_response_predict"
_C.path.savedir = rootpath+"/output_dir"
_C.path.response = rootpath+"/data/GDSC2_IC50.csv"
_C.path.pathway = rootpath+"/data/CELL/pathway_cosmic.csv"
_C.path.mutation = rootpath+"/data/CELL/geo_mutation_cosmic.csv"
_C.path.methylation = rootpath+"/data/CELL/geo_methylation_cosmic.csv"
_C.path.expression = rootpath+"/data/CELL/geo_expression_cosmic.csv"
_C.path.morgan = rootpath+"/data/Drug/morgan_encoding.pkl"
_C.path.espf = rootpath+"/data/Drug/espf_encoding.pkl"
_C.path.psfp = rootpath+"/data/Drug/pubchem_encoding.pkl"

# model params
_C.model = CN()
_C.model.lr = 0.001
_C.model.weight_decay = 0
_C.model.epoch = 150
_C.model.cuda_id = 0


# Drug
_C.drug = CN()
_C.drug.drug_out_dim = 100

# Cell
_C.cell = CN()
_C.cell.cell_out_dim = 100

# Ban
_C.ban = CN()
_C.ban.ban_heads = 3

# Mlp
_C.mlp = CN()
_C.mlp.mlp_in_dim = 256
_C.mlp.mlp_hidden_dim = [512, 128]


def get_cfg_defaults():
    return _C.clone()