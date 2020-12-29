# Gene-Expression classification using TDA and Machine Learning

# Description
We utilize some recent developments in TDA (Topological Data Analysis) to curate gene expression data. We use persistent-1 and persistent-2 cycles to
1.  Select genes that help classifying the phenotype labels.
2. Curating cohort data.


## Instructions
1. Clone persistent-1 cycle repo from https://github.com/Sayan-m90/Persloop-viewer.
2. Clone persistent 2-cycle repo from https://github.com/Sayan-m90/Minimum-Persistent-Cycles.
3. Clone this repository.


## Files

**baseline.py**: Shows the baseline experiments using Decision-Tree and Naive-Bayes. It uses logistic regression also.


**baseline_nn.py**: Shows the baseline experiments described in the paper with the Neural nets.


**droso_breeding_genex.npy**: Has the Droso-breeding dataset described in the paper. It is an NumPy nd-array whose rows are cohorts and columns are gene-expression values.


**droso_breeding_labels.npy**: We assign label 0 to control, label 1 to the Drosophilas bred on Aspergillus nidulans mutant laeA, and label 2 to both the Drosophilas bred on wild Aspergillusnidulans and sterigmatocystin.


## Requirements

For persistent-1 cycle and persistent-2 cycles follow their corresponding repos. For this repo requirements are 
1. pytorch 1.4.0
2. scikit-learn
3. numpy

## Paper
The paper is submitted in APBC 2021. Full link and submission will be available soon.
