#!/bin/bash
#SBATCH --output=R-%x.%j.out
#SBATCH --error=R-%x.%j.err
#SBATCH -D .
#SBATCH --ntasks=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=112
#SBATCH -q gp_resa

export SRUN_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}

module load singularity

dataset=$1
algorithm=$2
query=queries/rdkit_cancer.csv
ulimit -s unlimited

export OMP_NUM_THREADS=64
singularity exec clustering.sif python3 algorithms/runner.py ${dataset} ${query} ${algorithm}

# algorithm: agglomerative, bisecting, birch, dbscan, featureagg, hdbscan, kmeans, minibatch, optics
