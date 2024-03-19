rm(list = ls())
options(stringsAsFactors = F)
library(GSVA)
library(GSEABase)
library(msigdbr)
library(clusterProfiler)
library(org.Hs.eg.db)
library(enrichplot)
library(limma)
library(GSEABase)


remove(list = ls())
# Read the gene expression matrix
mydata <- read.table("../gene_expressions.csv", sep=",",
                    header=T, row.names=1,check.names=F)
# columns are genes, rows are Cell Lines
mydata <- t(mydata)
mydata= as.matrix(mydata)
mydata[1:5, 1:5]
# import MsigDb data set
msigdb <- "../c2.cp.v2022.1.Hs.symbols.gmt"
# read gmt file
geneset <- getGmt(file.path(msigdb))
# GSVA analysis
es.max <- gsva(mydata, geneset, min.sz=5,
               mx.diff=FALSE, verbose=T, kcdf="Gaussian",
               parallel.sz = parallel::detectCores())
head(es.max[1:10,1:2])
write.table(es.max, file="../GSVA_deepcdr_result.txt",sep="\t",
            quote=F,row.names = T)
save(es.max, file = "../GSVA_cosmic_result.Rda")

