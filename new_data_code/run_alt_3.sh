#!/bin/sh
#SBACTH --job-name=“Dataset3_shartarc_alt”
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --constraint=A100
#SBATCH --time=0-18:00
#SBATCH --mem=200gb
#SBATCH --partition=short
#SBATCH --mail-user=gmalabanti@wpi.edu
#SBATCH --mail-type=ALL
#SBATCH --output='/work/shared/DEVCOM-SC21/Network/code/new_data_code/Dataset3_shortarc_alt.out'
eval "$(conda shell.bash hook)"
conda activate deepLearning
srun python run_code_newarc_shorter_alt_3.py