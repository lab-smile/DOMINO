#!/bin/bash
#SBATCH --job-name=build_container  
#SBATCH --mail-type=END,FAIL          
#SBATCH --mail-user=<email>
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=30gb
#SBATCH --time=08:00:00
#SBATCH --output=%x.%j.out
date;hostname;pwd

module load singularity

# build a Singularity sandbox container (container in a writable directory) from MONAI Core docker image
singularity build --sandbox <directory location> docker://projectmonai/monai:1.1.0

# check nsys environment
singularity exec --nv <> nsys status -e