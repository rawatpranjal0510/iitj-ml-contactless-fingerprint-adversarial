#!/bin/bash
# Full pipeline in order. Usage: ./run_all.sh <stage>
#   phase1 | sweep | quality_prep | quality_merge | attacks | analysis
# GPU stages submit SLURM jobs; wait for each to finish before the next stage.
set -e; cd /iitjhome/m24ma2008/Pranjal/MLProject/Codes; PY=~/.conda/envs/fp_env/bin/python
case "$1" in
  phase1)        $PY data.py; $PY -m src.pairs; sbatch train_job.sh ;;           # E0 train v3 (select by val cross EER) + E1 calibrate + diag
  sweep)         sbatch sweep_job.sh ;;                                         # optional: 6-run model-selection sweep (kept v3)
  quality_prep)  $PY quality.py prep ;;                                         # -> out/qimg.zip; score on Windows (README), copy CSVs to out/
  quality_merge) $PY quality.py merge ;;                                        # -> out/quality.csv, out/quality_cutoffs.json (val tertiles)
  attacks)       sbatch attack_job.sh --splits test; sbatch attack_job.sh --splits val ;;
  analysis)      $PY analyze.py --file attacks_test; $PY analyze.py --file attacks_val
                 $PY policy.py; $PY policy.py --metric mclfiq
                 $PY policy.py --primary 0.02 --suffix _eps0.02
                 $PY policy.py --far_budget 0.0012 --suffix _far0.12; $PY policy.py --far_budget 0.002 --suffix _far0.20
                 $PY make_figs.py ;;
  *) echo "usage: $0 {phase1|sweep|quality_prep|quality_merge|attacks|analysis}"; exit 1 ;;
esac
