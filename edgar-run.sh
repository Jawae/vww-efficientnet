#!/bin/zsh
#SBATCH --cpus-per-task 88
#SBATCH --gres=gpu:1
#SBATCH --job-name=mbed-exporter
source ~/.zprofile
cd ~/mbed-exporter
srun pipenv run python quantizer.py $*
