#!/bin/bash

#declare -a arr=("agglomerative" "bisecting" "birch" "dbscan" "meanshift" "hdbscan" "kmeans" "optics" "minibatch")
declare -a arr=("canberra" "chebyshev" "dice" "euclidean" "hamming" "manhattan")
declare -a dataset=("biosynth" "chebi" "chembl" "chemspaceqed" "drugbank" "enamine")

for alg in "${arr[@]}"
do
        for ds in "${dataset[@]}"
        do
                singularity exec clustering.sif python recoger-tanimoto.py salida-${alg}-${ds}-cancer-*txt
		singularity exec clustering.sif python recoger-tanimoto.py salida-${alg}-${ds}-covid19-*txt
		singularity exec clustering.sif python recoger-tanimoto.py salida-${alg}-${ds}-diabetes-*txt
		singularity exec clustering.sif python recoger-tanimoto.py salida-${alg}-${ds}-malaria-*txt
		singularity exec clustering.sif python recoger-tanimoto.py salida-${alg}-${ds}-rheumatoid_arthritis-*txt
        done
done
