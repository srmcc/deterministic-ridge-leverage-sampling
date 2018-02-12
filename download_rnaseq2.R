#!/usr/bin/env Rscript
args = commandArgs(trailingOnly=TRUE)

 # test if there is at least one argument: if not, return an error
if (length(args)==0) {
  stop("At least one argument must be supplied (disease_type).n", call.=FALSE)
} else if (length(args)==1) {
  # default output file
  args[2] = ""
}

disease_type= args[1]
file_loc= args[2]

## program...
library(TCGA2STAT)

if (disease_type=="LGG") {
data_types=c("RNASeq2")
}

data_list=list()


for (i in 1:length(data_types)) {
  if (data_types[i]=="RNASeq2"){ data_list[[i]] = getTCGA(disease=disease_type, data.type=data_types[i], clinical=FALSE) }
}

for (i in 1:length(data_types)) {
    num_samp= toString(dim(data_list[[i]]$dat)[2])
    write.table(data_list[[i]]$dat, file= paste(file_loc, paste("tcga", disease_type, data_types[i], num_samp, sep='_'), ".csv", sep=''), sep = ',')
    write.table(colnames(data_list[[i]]$dat), file= paste(file_loc, paste("tcga", disease_type, data_types[i], "barcodes", sep='_'), ".csv", sep=''), sep = ',')
  }
