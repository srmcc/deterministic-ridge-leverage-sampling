#!/usr/bin/env Rscript

# Survival Factor Model, Copyright (C) 2017,  Shannon McCurdy and Regents of the University of California.

# Contact:  Shannon McCurdy, smccurdy@berkeley.edu.

#     This file is part of Survival Factor Model.

#     Survival Factor Model is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.

#     Survival Factor Model is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU General Public License for more details.

#     You should have received a copy of the GNU General Public License
#     along with Survival Factor Model program.  If not, see <http://www.gnu.org/licenses/>.



# The TCGA2STAT package and the underlying code in the package are distributed under the GPL2.0 license. Users of the package are free to use and redistribute the package but editing is prohibited. The TCGA2STAT package is available from The Comprehensive R Archive Network (CRAN) and the package’s Github.
# Usage of this package to import the TCGA data and employ the data in downstream analysis constitutes to agreement to TCGA data usage policy.


# The TCGA2STAT package uses the CNTOOLS package.  The CNTOOLS package is licensed under the LGPL License.


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
data_types=c("RNASeq2","miRNASeq", "Mutation","CNV_SNP", "CNA_SNP", "Methylation")
methyl_type="450K"
}

if (disease_type=="GBM") {
data_types=c("RNASeq2", "Mutation","CNV_SNP", "CNA_SNP", "Methylation")
methyl_type="27K"
}

if (disease_type=="LUAD") {
data_types=c("RNASeq2","miRNASeq", "Mutation","CNV_SNP", "CNA_SNP")
}

if (disease_type=="LUSC") {
data_types=c("RNASeq2", "Mutation","CNV_SNP", "CNA_SNP", "Methylation")
methyl_type="27K"
}

data_list=list()

for (i in 1:length(data_types)) {
	if (data_types[i]=="RNASeq2"){ data_list[[i]] = getTCGA(disease=disease_type, data.type=data_types[i], clinical=TRUE) }
	if (data_types[i] =="miRNASeq"){ data_list[[i]] = getTCGA(disease=disease_type, data.type=data_types[i],  type="rpmmm", clinical=FALSE) }
	if (data_types[i] %in% c( "Mutation", "CNA_SNP")){ data_list[[i]] = getTCGA(disease=disease_type, data.type=data_types[i], clinical=FALSE) }
	if (data_types[i] =="CNV_SNP"){ data_list[[i]] = getTCGA(disease = disease_type, data.type = data_types[i], type = "", filter = "Y", p = getOption("mc.cores", 10L), clinical = FALSE, cvars = "OS")}
	if (data_types[i]== "Methylation"){data_list[[i]] = getTCGA(disease = disease_type, data.type = data_types[i], type = methyl_type, filter = "Y", p = getOption("mc.cores", 10L), clinical = FALSE, cvars = "OS")}
}


for (i in 2:(length(data_types)+1)){
	if (i==2){
		bind= OMICSBind(dat1=data_list[[1]]$dat, dat2=data_list[[i]]$dat)
	}
	if (i %in% 3:length(data_types)){ bind = OMICSBind(dat1=bind$X, dat2=data_list[[i]]$dat)}
	if (i== length(data_types)+1){ 
		bind =  OMICSBind(dat1=bind$X, dat2=t(data_list[[1]]$clinical))
		num_samp= toString(dim(bind$X)[2])
		write.table(bind$X, file= paste(file_loc, paste("tcga", disease_type, data_types[1], num_samp, sep='_'), ".csv", sep=''), sep = ',')
		write.table(bind$Y, file= paste(file_loc, paste("tcga", disease_type, "Clinical", num_samp, sep='_'), ".csv", sep=''), sep = ',')
		write.table(colnames(data_list[[1]]$dat), file= paste(file_loc, paste("tcga", disease_type, data_types[1], "barcodes", sep='_'), ".csv", sep=''), sep = ',')
	}
}

for (i in 2:length(data_types)){
		new_bind =  OMICSBind(dat1=bind$X, dat2= data_list[[i]]$dat)
		write.table(new_bind$Y, file= paste(file_loc, paste("tcga", disease_type, data_types[i], num_samp, sep='_'), ".csv", sep=''), sep = ',')
}
