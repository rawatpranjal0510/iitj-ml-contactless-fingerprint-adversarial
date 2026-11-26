# Reproducibility

## Declaration of reproducibility

We declare that all results, tables and figures reported in our paper (`reports/paper.pdf`) were produced by the code in this repository at release tag **`v1.0`**, using the environment in `environment.yml` / `requirements.txt`, the trained checkpoint `checkpoints/v3.pt` (SHA-256 below), and the subject-disjoint split recorded in `results/split.json` and `results/data/images.csv`. All random seeds are fixed. Every decision that could influence results (model selection, thresholds, quality tiers, policy parameters) was made on the validation split only; the test split was used once, for final evaluation. Raw per-pair outputs, analysis tables and logs of the reported run are included under `results/`, so every number in the paper can be checked against a file. The only external input not redistributed is the PolyU Contactless 2D database, which is licensed by its owners (see Data). The quality-scoring step uses the official NIST NFIQ2 Windows binaries; its outputs (`results/data/quality.csv`) are included so the rest of the pipeline can be reproduced without it.

## What is provided

| Artifact | Location |
|---|---|
| Code (pipeline scripts, library) | `*.py`, `src/` |
| SLURM job scripts | `jobs/` |
| One-command stage runner | `run_all.sh` |
| Environment | `environment.yml`, `requirements.txt` (pinned), `requirements-freeze.txt` (full original freeze) |
| Trained checkpoint (45 MB) | `checkpoints/v3.pt`, checksum in `checkpoints/SHA256SUMS` |
| Split definition | `results/split.json`, `results/data/images.csv` |
| Frozen thresholds and quality cutoffs | `results/tau.json`, `results/quality_cutoffs.json` |
| Quality scores (NFIQ2 v2.3, MCLFIQ under v2.2) | `results/data/quality.csv` |
| Per-pair attack results | `results/data/attacks_{test,val}.csv`, `results/data/attacks_test_strong.csv` |
| All analysis tables (CSV + LaTeX) | `results/tables/` |
| All figures (PNG + PDF) | `results/figures/` |
| Training, sweep, attack, analysis and policy logs | `results/runs/`, `results/logs/` |
| Paper source and compiled PDF | `paper/`, `reports/paper.pdf` |

Checkpoint SHA-256: `6d770c64a84a38bb1f271d1bf7c8571efb17428decc5fe302626aa87343feff4  v3.pt`

## Environment

- Python 3.10.22, PyTorch 2.5.1 (CUDA 12.4), torchvision 0.20.1, NumPy 1.26.4, pandas 2.3.3, SciPy 1.15.3, scikit-learn 1.7.2, statsmodels 0.15.0, matplotlib 3.10.9.
- `conda env create -f environment.yml && conda activate fp_env` (or `pip install -r requirements.txt` in a Python 3.10 environment).
- Hardware used: one NVIDIA A100-SXM4-40GB (SLURM `--partition=dgx --gres=gpu:1 --cpus-per-task=8 --mem=32G`), RHEL 8.6.
- Quality scoring: NIST NFIQ2 v2.3.0 and v2.2.0 Windows installers (`nfiq2-2.3.0-win64.msi`, `nfiq2-2.2.0-win64.msi`) and the MCLFIQ model (Priesnitz et al.).

## Data

The PolyU Contactless 2D Fingerprint Database is not redistributed (licence of The Hong Kong Polytechnic University; request access from the PolyU Biometrics Research Centre). Place it at the path set as `DATA_ROOT` in `config.py`, with this layout:

```
processed_contactless_2d_fingerprint_images/
  first_session/p1/p1.bmp ... p6.bmp   (336 finger folders)
  second_session/p1/p1.bmp ... p6.bmp  (160 finger folders, same IDs as session 1)
```

`data.py` rebuilds the split deterministically (seed 42); the result must match `results/data/images.csv` and `results/split.json` (202 / 84 / 50 fingers; 1,368 / 1,008 / 600 images).

## Seeds and determinism

- Split: `SEED = 42` (`config.py`). Training: `--seed 0`. Attacks: `--seed 0`. Bootstrap: seed 0, 1,000 resamples.
- `cudnn.benchmark` and TF32 are enabled, so scores can differ between GPU models at about 1e-3. `attack.py` checks recomputed clean scores against `calibrate.py` with a tolerance of 1e-2 and prints the observed difference. Results were produced on one A100.
- Retraining reproduces the reported metrics to within seed-level variation (validation cross EER 7.55-8.34% over six runs); exact reproduction of every reported number uses the released checkpoint.

## How to reproduce

All commands run from the repository root. Set `DATA_ROOT` in `config.py` first. `PY` is your environment's Python.

### Option A: from the released checkpoint (exact numbers)

```bash
PY=python                                    # Python of the fp_env environment
$PY data.py                                  # split -> out/images.csv, out/split.json (compare with results/)
$PY -m src.pairs                             # pairs -> out/pairs_{val,test}.csv
mkdir -p out/ckpt && cp checkpoints/v3.pt out/ckpt/v3.pt
$PY calibrate.py --ckpt v3                   # thresholds -> out/tau.json (tau=0.6837, tau_floor=0.6413), scores_*.csv
$PY diag.py                                  # EER / TAR by pair kind
cp results/data/quality.csv results/quality_cutoffs.json out/   # or re-score quality (Option C)
sbatch jobs/attack_job.sh --splits test      # PGD + eps* + noise -> out/attacks_test.csv  (about 18 min on an A100)
sbatch jobs/attack_job.sh --splits val       # -> out/attacks_val.csv                       (about 28 min)
$PY analyze.py --file attacks_test           # Stage 1 tables/figures (test)
$PY analyze.py --file attacks_val            # Stage 1 replication (val)
$PY policy.py                                # Stage 2 (+ ablations, see run_all.sh)
$PY make_figs.py                             # publication figures -> out/figs/paper/
```

Without SLURM, run `python attack.py --splits test` (and `--splits val`) directly on a GPU.

### Option B: full pipeline including training

```bash
./run_all.sh phase1         # data, pairs, train (SLURM, about 5 min), calibrate, diag
./run_all.sh sweep          # optional six-run sensitivity sweep
./run_all.sh quality_prep   # then Option C, then:
./run_all.sh quality_merge
./run_all.sh attacks        # two SLURM jobs
./run_all.sh analysis       # Stage 1, Stage 2, ablations, figures
./make_release.sh           # copy outputs into results/ and paper/
```

### Option C: re-score quality (Windows, NIST binaries)

1. `python quality.py prep` writes `out/qimg.zip` (contact-like conversion: local mean/std normalisation sigma = 8, inversion, PNG tagged 500 DPI).
2. On Windows, unzip it, write `qbatch.txt` (one PNG path per line) and run:
   - `"C:\Program Files\NFIQ 2\bin\nfiq2.exe" -F -j 8 -o q_nfiq2.csv -f qbatch.txt`
   - `nfiq22\NFIQ 2\bin\nfiq2.exe -F -j 8 -o q_mclfiq_v22.csv -m MCLFIQ.txt -f qbatch.txt`, where `nfiq22` is the v2.2 installer extracted with `msiexec /a nfiq2-2.2.0-win64.msi /qn TARGETDIR=...` and `MCLFIQ.txt` is a model-info file (Name, Trainer, Description, Version, Path, Hash = MD5 of the MCLFIQ yaml).
3. Copy both CSVs to `out/` and run `python quality.py merge`.

## Expected key numbers (test split unless stated)

| Quantity | Value | File |
|---|---|---|
| Cross-session EER / TAR at FAR 0.1% | 10.50% / 30.7% | `results/logs`, `calibrate.py` output |
| tau_fixed / tau_floor; test FAR | 0.6837 / 0.6413; 0.076% / 0.180% | `results/tau.json` |
| ASR at eps = 0.05/255; median eps* | 69.6%; 0.034/255 | `results/tables/b3_test.csv`, `results/logs/attack_test.log` |
| eps* ratio per +0.1 Q, log-margin control | 1.068 (1.037-1.095); val 1.038 (1.016-1.063) | `results/tables/regression_{test,val}.csv` |
| Same, with finger fixed effects | 1.032 (p = 0.08); val 1.016 (p = 0.47) | `results/tables/finger_fe_{test,val}.csv` |
| Adaptive threshold: Q1 TAR gain vs fixed / vs equal-FAR global | +9.1 / +2.3 points | `results/tables/policy_ci_test_nfiq2.csv` |
| Attack-strength check (5 x 100 steps) | median eps* ratio 1.000 | `results/data/attacks_test_strong.csv` |

Bootstrap intervals are reproduced exactly with the fixed seed; regenerated files can be diffed against `results/`.
