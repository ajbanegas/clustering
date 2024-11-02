#!/bin/bash
#SBATCH --output=R-%x.%j.out
#SBATCH --error=R-%x.%j.err
#SBATCH -D .
#SBATCH --ntasks=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH -q gp_resa

export SRUN_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}
export PYTHONUNBUFFERED=1

module load singularity

dataset=$1
query=$2
algorithm=$3
ulimit -s unlimited

export OMP_NUM_THREADS=64 # 64
singularity exec clustering.sif python3 algorithms/runner-clustering.py ${dataset} ${query} ${algorithm}

# algorithm: agglomerative, bisecting, birch, dbscan, hdbscan, kmeans, meanshift, minibatch, optics

#ds_file = sys.argv[1]
#q_file = sys.argv[2]
#alg_key = sys.argv[3]

