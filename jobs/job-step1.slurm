#!/bin/bash
#SBATCH --output=R-%x.%j.out
#SBATCH --error=R-%x.%j.err
#SBATCH -J Drugbank
#SBATCH -D .
#SBATCH --ntasks=1

# kmeans, bisecting, agglomerative, dbscan, hdbscan, optics, birch
# datasets/rdkit_Drugbank.csv

dataset=$1
algorithm=$2

# Drugbank clusterization
python algorithms/runner.py ${dataset} queries/rdkit_cancer.csv ${algorithm}
