#!/bin/bash
#SBATCH --output=R-%x.%j.out
#SBATCH --error=R-%x.%j.err
#SBATCH -J Drugbank
#SBATCH -D .
#SBATCH --ntasks=1

# Drugbank clusterization
python algorithms/runner.py datasets/rdkit_Drugbank.csv queries/rdkit_cancer.csv kmeans
#python algorithms/runner.py datasets/rdkit_Drugbank.csv queries/rdkit_cancer.csv bisecting
#python algorithms/runner.py datasets/rdkit_Drugbank.csv queries/rdkit_cancer.csv agglomerative
#python algorithms/runner.py datasets/rdkit_Drugbank.csv queries/rdkit_cancer.csv dbscan
#python algorithms/runner.py datasets/rdkit_Drugbank.csv queries/rdkit_cancer.csv hdbscan
#python algorithms/runner.py datasets/rdkit_Drugbank.csv queries/rdkit_cancer.csv optics
#python algorithms/runner.py datasets/rdkit_Drugbank.csv queries/rdkit_cancer.csv birch

# Drugbank evaluation
#python algorithms/runner-clustering.py datasets/rdkit_Drugbank.csv queries/rdkit_cancer.csv kmeans
#python algorithms/runner-clustering.py datasets/rdkit_Drugbank.csv queries/rdkit_cancer.csv bisecting
#python algorithms/runner-clustering.py datasets/rdkit_Drugbank.csv queries/rdkit_cancer.csv agglomerative
#python algorithms/runner-clustering.py datasets/rdkit_Drugbank.csv queries/rdkit_cancer.csv dbscan
#python algorithms/runner-clustering.py datasets/rdkit_Drugbank.csv queries/rdkit_cancer.csv hdbscan
#python algorithms/runner-clustering.py datasets/rdkit_Drugbank.csv queries/rdkit_cancer.csv optics
#python algorithms/runner-clustering.py datasets/rdkit_Drugbank.csv queries/rdkit_cancer.csv birch

