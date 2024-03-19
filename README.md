# BANDRP
An end-to-end deep learning model based on multiple omics data of cancer cell lines and multiple molecular fingerprints of drugs for anti-cancer drug prediction, named BANDRP. 
https://raw.githubusercontent.com/heckletbot/BANDRP/main/framework.svg?sanitize=true


# Requirements
* python == 3.9
* pytorch == 1.11.0
* Numpy == 1.21.5
* scikit-learn == 1.0.2
* pandas == 1.4.1
* rdkit == 2023.3.3
* scipy == 1.7.3



# Files:

1.data

**geo_expression.csv**: Gene expression data for cancer cell lines.

**geo_mutation.csv**: Genomic mutation data for cancer cell lines.

**geo_methylation.csv**: DNA methylation data for cancer cell lines.

**pathway.csv**: Pathway enrichment scores for cance cell lines.

**ECFP.pkl**: Extended Connectivity Fingerprint for drugs.

**ESPF.pkl**: Explainable Substructure Partition Fingerprint for drugs.

**PSFP.pkl**: PubChem Substructure Fingerprint for drugs.

**GDSC2_ic50.csv**: The IC50 values matrix between 169 drugs and 536 cells.

**cell_line_info.csv**:  Specific information of Cancer cell lines.

**drug_info.csv**: Specific information of drugs.

GDSC2_ic50.csv: The ic50 values collected from GDSC v2 database (https://www.cancerrxgene.org/downloads/bulk_download). The row of the file represents cancer cell lines and the column of the file represents drugs. The file data are IC50 values.

geo_expression.csv, geo_mutation.csv, geo_methylation.csv: These files are all collected from the CCLE database (https://depmap.org/portal/download/all/). The row of the files represents cancer cell lines and the column of the files represents genes.

pathway.csv: The pathway enrichment scores of cancer cell lines are calculated by the GSVA R package. The row of the files represents cancer cell lines and the column of the files represents pathways.

ECFP.pkl, ESPF.pkl, PSFP.pkl: All molecular fingerprints of the drug were calculated using the Rdkit python package. These files are stored as Python dictionaries, where the keys represent the PubChem IDs of drugs and the values represent their fingerprints.


2.Code

**main.py**: This function is used to train and test BANDRP.

**config.py**: This function is used to control the hyperparameters of BANDRP.

**drug_fingerprint.py**: This function is used to calculate the molecular fingerprint of drugs. It input the SMILES strings of drugs and output the molecular fingerprint of drugs.

**cell_process.r**: This function is used to calculate pathway enrichment scores for cancer cell lines. It inputs gene expression data of cancer cell lines and outputs pathway enrichment scores of cancer cell lines.

**data_load.py**: This function is used to load the data of cancer cell lines, drugs and IC50 values.

**data_process.py**: This function is used to process input data.

**model.py**: This function contains the BANDRP model components.

**utils.py**: This function contains the necessary processing subroutines.



# Require input files
You should prepare gene expression, genomic mutation, DNA methylation, and Pathway enrichment scores for cancer cell lines, ECFP, ESPF, and PSFP for drugs, and IC50 values between cancer cell lines and drugs. You can calculate pathway enrichment scores through **cell_process.r** and calculate molecular fingerprints of drugs through **drug_fingerprint.py**.



# Train and test folds
Before running the model, you need to set hyperparameters in **config.py**, including the following:

rootpath: All code and input data should be placed in the folder of this path. (The data folder we uploaded contains all the required data.)

savedir: Define the path to save the model.

cuda_id: Select the GPU to use, otherwise the cpu will be used.

epoch: Define the maximum number of epochs.

batch_size: Define the number of batch size for training and testing.

ban_heads: Define the num of muti-head attention.

All files of Data and Code should be stored in the same folder to run the model.

After setting up, you can run BANDRP through *python main.py*.
