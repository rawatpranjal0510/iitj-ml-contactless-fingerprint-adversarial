#!/bin/bash
#SBATCH --job-name=fp_attack
#SBATCH --partition=dgx
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=logs/attack_%j.log
#SBATCH --error=logs/attack_%j.log

cd /iitjhome/m24ma2008/Pranjal/MLProject/Codes
PY=~/.conda/envs/fp_env/bin/python
export PYTHONUNBUFFERED=1
echo "Node: $(hostname)  Start: $(date)  Args: $@"
$PY attack.py "$@"
echo "End: $(date)"
