#!/bin/bash
#SBATCH --job-name=monai_test
#SBATCH --mail-type=END,FAIL          
#SBATCH --mail-user=<Enter Email>    	
#SBATCH --ntasks=1                   
#SBATCH --cpus-per-task=4
#SBATCH --distribution=block:block 
#SBATCH --partition=hpg-ai
#SBATCH --gres=gpu:a100:2
#SBATCH --mem=30gb                     
#SBATCH --time=72:00:00               
#SBATCH --output=%x_%j.log 

#module load pytorch
module load singularity
#module load monai

# Check if cuda enabled
#python -c "import torch; print(torch.cuda.is_available())"
singularity exec --nv <Enter path to MONAI container>/monaicore081 python3 -c "import torch; print(torch.cuda.is_available())"

#run code
#python test.py --num_gpu 2 --data_dir "<Data Directory>" --N_classes 12 --model_load_name "unetr_v5_bfc.pth" --dataparallel "True"

singularity exec --nv --bind <Enter path to testing file>:/mnt <Enter path to MONAI container>/monaicore081 python3 /mnt/test.py --num_gpu 2 --data_dir '/mnt/<data directory>/' --model_load_name "unetr_v5_06-19-22.pth" --N_classes 12 --dataparallel "True" --a_max_value 255 --spatial_size 64