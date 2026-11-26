# Margin-Controlled Adversarial Vulnerability & Quality-Adaptive Thresholding (PolyU Contactless 2D)

**Kalpana Yadav, Pranjal Rawat, Vaibhav Kunte** · Machine Learning course project, Indian Institute of Technology Jodhpur

📄 **Paper:** [reports/paper.pdf](reports/paper.pdf) · LaTeX source: [paper/](paper/) · Results: [results/](results/)

## What this project asks
Low-quality contactless fingerprint probes look more vulnerable to adversarial attacks, but they also sit closer to the acceptance threshold. **Does image quality predict adversarial vulnerability *beyond* the clean verification margin M = S − τ?** And can a quality-adaptive threshold recover genuine acceptance for poor captures without loosening security?

## Key findings (held-out test set, replicated on validation)
- **The matcher is fragile:** white-box L∞ PGD breaks every accepted cross-session pair at ε = 1/255; random noise of the same budget does not.
- **Margin dominates:** the clean margin explains about half of the raw quality–robustness association (robustness radius ε\* vs margin: Spearman 0.95).
- **A small residual remains, but it is between fingers:** ε\* ratio per +0.1 NFIQ2 quality after log-margin control is 1.068 (95% CI 1.037–1.095) on test and 1.038 on validation, and it disappears within fingers (p = 0.08 and 0.47).
- **The adaptive threshold redistributes acceptance:** it raises low-quality TAR by 9.1 points at 0.12% FAR, but matches a single threshold at equal FAR overall.
- **Margin beats quality for rejection:** at ε = 0.02/255, rejecting the 20% smallest-margin genuine pairs prevents 65% of attacks, against 30% for quality-based rejection.

## Setup at a glance
PolyU Contactless 2D (S1 enrolled vs S2 probe) · subject-disjoint split 202/84/50 fingers · ResNet-18 + ArcFace · NFIQ2 v2.3 (MCLFIQ as ablation) · PGD with robustness radius ε\* · Firth logistic + log ε\* regression with 1,000-resample finger-clustered bootstrap.

## Pipeline (`./run_all.sh <stage>`)
| Stage | Scripts | Output |
|---|---|---|
| phase1 | `data.py`, `src/pairs.py`, `train_job.sh` (`train.py` -> `calibrate.py` -> `diag.py`) | `out/images.csv`, `split.json`, `pairs_*.csv`, `ckpt/v3.pt`, `tau.json`, `scores_*.csv` |
| sweep (optional) | `sweep_job.sh` | `out/runs/v3_*.json` |
| quality_prep / merge | `quality.py prep`, Windows NFIQ2 (below), `quality.py merge` | `out/quality.csv`, `quality_cutoffs.json` |
| attacks | `attack_job.sh` (`attack.py`) | `out/attacks_{test,val}.csv` |
| analysis | `analyze.py`, `policy.py`, `make_figs.py` | `out/tables/*`, `out/figs/paper/*` |

Environment: `~/.conda/envs/fp_env` (PyTorch 2.5.1, statsmodels 0.15); SLURM `--partition=dgx --gres=gpu:1`.

## Quality scoring on Windows (NFIQ2 has no RHEL 8 v2.3 build)
1. Install `nfiq2-2.3.0-win64.msi` (NFIQ2 v2.3). Extract `nfiq2-2.2.0-win64.msi` without installing:
   `msiexec /a nfiq2-2.2.0-win64.msi /qn TARGETDIR=%USERPROFILE%\Desktop\nfiq22`
2. MCLFIQ model (`MCLFIQ_model.zip`, Priesnitz et al.) + model-info file `MCLFIQ.txt` (Name/Trainer/Description/Version/Path/Hash=MD5 of the yaml).
3. Copy and unzip `out/qimg.zip`, write a batch file (one PNG path per line), then:
   - NFIQ2 v2.3: `nfiq2.exe -F -j 8 -o q_nfiq2.csv -f qbatch.txt`
   - MCLFIQ under v2.2: `nfiq22\NFIQ 2\bin\nfiq2.exe -F -j 8 -o q_mclfiq_v22.csv -m MCLFIQ.txt -f qbatch.txt`
4. Copy both CSVs to `out/` and run `quality.py merge`.

## Protocol decisions (and deviations from the proposal)
- **Split**: subject-disjoint, SEED=42: 202/84/50 fingers = 1,368/1,008/600 images (proposal table listed 2,424/2,016/1,200).
- **Input size**: 224x352 (H x W), native 350x225 aspect preserved; no ROI crop (fingertip fills the frame).
- **Impostors (val/test)**: all cross-session pairs (S1 enrolled vs S2 probe, different fingers), so tau controls FAR in the attacked scenario.
- **Thresholds**: `tau_fixed` at FAR 0.1%, `tau_floor` at FAR 0.2%, both from val impostors, conservative quantile (`method="higher"`).
- **Model selection**: by val EER on cross pairs (`--select eer`); a 6-run sweep (seeds, ArcFace margin, regularisation, LR) stayed within seed noise, so v3 was kept.
- **Quality**: contactless -> contact-like (local mean/std normalisation sigma=8, inverted, PNG tagged 500 DPI; ridge period ~8.6 px is already about 500 PPI). NFIQ2 v2.3 is primary. MCLFIQ runs under NFIQ2 v2.2 (it was trained on v2.2 features; under v2.3 it inverted). Tier cutoffs = val S2-probe tertiles. Failures: Q=0, flagged.
- **Attack budgets**: PGD saturates at the proposal grid {1,2,4,8}/255 (100% ASR), so a fine grid {0.01..0.15}/255 and a per-pair robustness radius eps* (bisection) were added. Sub-1/255 budgets are robustness distances in continuous pixel space, not storable 8-bit attacks.
- **Stage 1**: primary outcome log eps* (OLS); success at fixed eps via Firth logistic (separation). 1,000-resample finger-clustered bootstrap. Finger fixed-effects check added. Replicated on val.
- **Stage 2**: tau(Q) = clip(tau_fixed - alpha*max(0, Q_high - Q), tau_floor, tau_fixed); NFIQ2 failures keep tau_fixed. alpha and Q_high tuned on val (max Q1 TAR under attack at eps=0.05/255, val FAR <= 0.15%). Added baseline B2-global = one threshold at B2's val FAR.

## Repository layout
- `./`: pipeline scripts (see Pipeline); `src/`: model, dataset, pairs, evaluation
- `results/`: outputs of the run reported in the paper (rebuild with `./make_release.sh`)
  - `tables/`: CSV + LaTeX tables (Stage 1 `*_test`/`*_val`, `policy_*`, `b4_*`, `policy_params_*.json`)
  - `figures/`: publication figures (PNG + PDF)
  - `data/`: `images.csv` (split), `quality.csv`, `attacks_{test,val}.csv`, `attacks_test_strong.csv`
  - `runs/`: training and sweep summaries; `logs/`: analysis, policy and attack logs
  - `tau.json`, `quality_cutoffs.json`, `split.json`
- `paper/`: IEEE paper (`main.tex`, `refs.bib`, `figures/`); zip the folder and upload it to Overleaf

Not in git: the PolyU dataset (licence), checkpoints (`out/ckpt`), large regenerable files (`pairs_*`, `scores_*`, `qimg`).

## Additional checks
- Attack budget (Stage 1): after log-margin control, the NFIQ2 odds ratio for success is below 1 at eps = 0.02, 0.03, 0.05, 0.075/255 on test (all CIs exclude 1) and at 3 of 4 on validation.
- Attack strength: 5 restarts x 100 steps on 100 random accepted test pairs left eps* unchanged (median ratio 1.000, IQR 0.995-1.000, Spearman 1.000; ASR 73.0% vs 73.0%).
- attack.py checks recomputed clean scores against calibrate.py (tolerance 1e-2; GPU models differ at about 1e-3 because of TF32).
