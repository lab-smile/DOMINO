#!/bin/bash
#SBATCH --job-name=train
#SBATCH --mail-type=END,FAIL          
#SBATCH --mail-user=<Enter Email Name Here>     	
#SBATCH --ntasks=1                   
#SBATCH --cpus-per-task=4
#SBATCH --partition=hpg-ai
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=90gb                     
#SBATCH --time=72:00:00          
#SBATCH --output=%x_%j.log 

#module load pytorch
module load singularity
#module load monai

# Check if cuda enabled
singularity exec --nv <Enter path to MONAI container>/monaicore081 python3 -c "import torch; print(torch.cuda.is_available())"

#run code
#singularity exec --nv <Enter path to MONAI container>/monaicore08 python3 <Enter path to train file>/train_domino.py --num_gpu 3
#python3 <Enter path to train file>/train_domino.py --num_gpu 3 --a_min_value 0 --a_max_value 255

singularity exec --nv --bind <Enter path to train file>:/mnt <Enter path to MONAI container>/monaicore081 python3 /mnt/train_domino.py --num_gpu 1 --data_dir '/mnt/<data folder name>/' --model_save_name "unetr_v5_domino_06-20-22" --N_classes 12 --max_iteration 100