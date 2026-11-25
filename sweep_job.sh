#!/bin/bash
#SBATCH --job-name=fp_sweep
#SBATCH --partition=dgx
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/sweep_%j.log
#SBATCH --error=logs/sweep_%j.log

cd /iitjhome/m24ma2008/Pranjal/MLProject/Codes
PY=~/.conda/envs/fp_env/bin/python
export PYTHONUNBUFFERED=1
echo "Node: $(hostname)  Start: $(date)"

# Model-selection sweep: each run picks its best epoch by VAL cross EER. No calibrate/diag here (keeps tau.json untouched).
while read tag args; do
  echo; echo "===== $tag : $args ====="
  $PY train.py --tag $tag --select eer --workers 4 $args || echo "!! $tag failed"
done << 'RUNS'
v3_s1  --seed 1
v3_s2  --seed 2
v3_m50 --m 0.5
v3_m30 --m 0.3
v3_reg --wd 0.1 --dropout 0.5
v3_lr3 --lr_bb 3e-4
RUNS
echo "End: $(date)"
