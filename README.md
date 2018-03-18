# Determinstic-Ridge-Leverage-Sampling
This repository has been developed to reproduce the results in McCurdy (2018, arXiv/XXXX).  We implement the deterministic ridge leverage sampling algorithm that comes with a (1 + \epsilon) error column subset selection, (1 + \epsilon) error projection-cost preservation, an additive-multiplicative spectral bound, and a (1 + \epsilon) bound on the statistical risk for ridge regression.  The code in this repository will download the biological data for you, perform the analysis, and make the figures.

# Installing the dependencies
To download the TCGA lower-grade glioma (LGG) tumor multi-omic data collected by the [TCGA Research Network](http://cancergenome.nih.gov/), you will first need to install [CNTools](http://bioconductor.org/packages/CNTools/) (Zhang, 2015) and [TCGA2STAT](http://www.liuzlab.org/TCGA2STAT/) (Wan et al., 2016) in R.  To install the R packages, open an R environment and type the following:
```R
## try http:// if https:// URLs are not supported
source("https://bioconductor.org/biocLite.R")
biocLite("CNTools")
install.packages("TCGA2STAT_1.0.tar.gz", repos = NULL, type = "source")
```
The rest of this code repository is in python.  We used the anaconda package manager which can be obtained at https://www.continuum.io/downloads, and we include a .yml file (ridge.yml) that we use for the analysis.  You can simply create an "conda env" with the .yml file by 
```bash
conda env create -f ridge.yml 
#
# To activate this environment, use:
# $ source activate ridge
#
# To deactivate this environment, use:
# $ source deactivate
#
```
# Getting Started
In order to run the code, first clone this repository.

```bash
git clone https://github.com/srmcc/determinstic-ridge-leverage-sampling.git
```
Then activate your conda environment:

```bash
source activate ridge
```
Change directory to the cloned repository:
```bash
cd to /path/to/Determinstic-Ridge-Leverage-Sampling
```
And run the analysis file.
```bash
python ridge.py
```

