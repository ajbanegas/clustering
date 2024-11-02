#!/bin/bash
#SBATCH --output=R-%x.%j.out
#SBATCH --error=R-%x.%j.err
#SBATCH -D .
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=112
#SBATCH -q gp_resa

algorithm=$1
database=$2
lisica_path="/home/ucam/user/clustering/lisica"

# for each file containing the candidates according to an algorithm...
for foo in compounds/${algorithm}-${database}*.txt
do
	# format: manhattan-enamine-malaria-8
	foobase=$(basename ${foo})
	foobase=$(echo ${foobase%.*})

	# ... parse the filename to extract the information
	database=$(echo ${foobase} | cut -d"-" -f2)
	queryset=$(echo ${foobase} | cut -d"-" -f3)
	queryid=$(echo ${foobase} | cut -d"-" -f4)

	suffix="${algorithm}-${database}-${queryset}-${queryid}"

	# read the list of candidates' ids from the file
	while read -r line
	do
		querypath="queries/${queryset}/${queryset}${queryid}.mol2"
		ligandpath="databases/${database}/${line}.mol2"
		timeout 10m ${lisica_path} -R ${querypath} -T ${ligandpath} -d 3 -w 500 -n 1 -m 2.5 -f /dev/null | head -1 >> salida-${suffix}.txt
	done < ${foo}
done

