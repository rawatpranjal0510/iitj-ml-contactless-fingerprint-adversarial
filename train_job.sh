#!/bin/bash
#SBATCH --job-name=fp_arcface
#SBATCH --partition=dgx
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/train_%j.log
#SBATCH --error=logs/train_%j.log

cd /iitjhome/m24ma2008/Pranjal/MLProject/Codes
PY=~/.conda/envs/fp_env/bin/python
export PYTHONUNBUFFERED=1

echo "Node: $(hostname)  Start: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv
$PY -c "import torch; print('CUDA available:', torch.cuda.is_available())"

$PY train.py --tag v3 --select eer --workers 4 || exit 1
$PY calibrate.py --ckpt v3
$PY diag.py
echo "End: $(date)"
