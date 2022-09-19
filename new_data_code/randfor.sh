#!/bin/sh
#SBACTH --job-name=“randomforest”
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --constraint=A100
#SBATCH --time=0-20:00
#SBATCH --mem=200gb
#SBATCH --partition=short
#SBATCH --mail-user=jspitaels@wpi.edu
#SBATCH --mail-type=ALL
#SBATCH --output='/work/shared/DEVCOM-SC21/Network/code/new_data_code/randomforest.out'
eval "$(conda shell.bash hook)"
conda activate deepLearning
srun python RandomForest.py