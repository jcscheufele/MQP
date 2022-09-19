#!/bin/sh
#SBACTH --job-name=“Joe Test”
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --constraint=A100
#SBATCH --time=0-02:00
#SBATCH --mem=160gb
#SBATCH --partition=short
#SBATCH --mail-user=jcscheufele@wpi.edu
#SBATCH --mail-type=ALL
#SBATCH --output='/work/shared/DEVCOM-SC21/Network/code/new_data_code/joe_test.out'
eval "$(conda shell.bash hook)"
conda activate deepLearning
srun python run_code_conv.py