# Margin-Controlled Adversarial Vulnerability & Quality-Adaptive Thresholding (PolyU Contactless 2D)

**Kalpana Yadav, Pranjal Rawat, Vaibhav Kunte** · Machine Learning course project, Indian Institute of Technology Jodhpur

📄 **Report:** [reports/paper.pdf](reports/paper.pdf) · LaTeX source: [paper/](paper/) · Results: [results/](results/) · **Reproduce:** [REPRODUCIBILITY.md](REPRODUCIBILITY.md) · Release: tag `v1.0`

## What this project asks
Low-quality contactless fingerprint probes look more vulnerable to adversarial attacks, but they also sit closer to the acceptance threshold. **Does image quality predict adversarial vulnerability *beyond* the clean verification margin M = S − τ?** And can a quality-adaptive threshold recover genuine acceptance for poor captures without loosening security?

## Key findings (held-out test set, replicated on validation)
- **The matcher is fragile:** white-box L∞ PGD breaks every accepted cross-session pair at ε = 1/255; random noise of the same budget does not.
- **Margin dominates:** the clean margin explains about half of the raw quality–robustness association (robustness radius ε\* vs margin: Spearman 0.95).
- **Two mechanisms of similar size:** low-quality probes start about 1.3× closer to τ and lose similarity about 1.5× faster under the same attack (mean drop at ε = 0.05/255: 0.158 for Q1 vs 0.106 for Q3).
- **The residual is between fingers:** ε\* ratio per +0.1 NFIQ2 quality after log-margin control is 1.068 (95% CI 1.037–1.095) on test and 1.038 on validation, and it disappears within fingers (p = 0.08 and 0.47).
- **The adaptive threshold redistributes acceptance:** it raises low-quality TAR by 9.1 points at 0.12% FAR, but matches a single threshold at equal FAR overall.
- **Margin beats quality for rejection:** at ε = 0.02/255, rejecting the 20% smallest-margin genuine pairs prevents 65% of attacks, against 30% for quality-based rejection.

## Repository structure
```
.
├── config.py              paths, seed, image size, FAR targets
├── data.py                image table + subject-disjoint split
├── train.py               ResNet-18 + ArcFace training (checkpoint selection by val cross EER)
├── calibrate.py           freeze tau_fixed (FAR 0.1%) and tau_floor (FAR 0.2%) on val impostors
├── diag.py                EER / TAR by pair kind, score histograms
├── quality.py             contact-like conversion for NFIQ2 (prep) and score merging (merge)
├── attack.py              L-inf PGD dodging, eps* bisection, random-noise control (B3)
├── analyze.py             Stage 1: margin bands, regressions, bootstrap, finger fixed effects
├── policy.py              Stage 2: adaptive threshold, B1/B2/B2-global, B4 rejection curves
├── make_figs.py           publication figures from saved tables
├── src/                   library: dataset, model (Embedder, ArcFace), pairs, evaluation
├── jobs/                  SLURM scripts: train_job.sh, sweep_job.sh, attack_job.sh, progress.sh
├── run_all.sh             run the pipeline stage by stage
├── make_release.sh        copy outputs into results/ and paper/
├── checkpoints/           v3.pt (trained model, 45 MB) + SHA256SUMS
├── results/               outputs of the reported run
│   ├── tables/            CSV + LaTeX tables
│   ├── figures/           PNG + PDF figures
│   ├── data/              images.csv, quality.csv, attacks_{test,val}.csv, attacks_test_strong.csv
│   ├── runs/  logs/       training/sweep summaries; attack, analysis and policy logs
│   └── tau.json  quality_cutoffs.json  split.json
├── paper/                 report LaTeX source (CVPR format) + figures
├── reports/paper.pdf      compiled report
├── environment.yml  requirements.txt  requirements-freeze.txt
├── REPRODUCIBILITY.md     declaration, environment, data, seeds, step-by-step instructions
└── README.md
```

## Quick start
```bash
conda env create -f environment.yml && conda activate fp_env      # or: pip install -r requirements.txt (Python 3.10)
# set DATA_ROOT in config.py to the PolyU database, then:
python data.py && python -m src.pairs
mkdir -p out/ckpt && cp checkpoints/v3.pt out/ckpt/v3.pt && python calibrate.py --ckpt v3
```
Full instructions (from the checkpoint, from scratch, and quality re-scoring): **[REPRODUCIBILITY.md](REPRODUCIBILITY.md)**.

## Checkpoint
`checkpoints/v3.pt` (45 MB) is the trained ResNet-18 + ArcFace embedder used for every reported result (epoch 22, val cross EER 7.84%). SHA-256: `6d770c64a84a38bb1f271d1bf7c8571efb17428decc5fe302626aa87343feff4`. Load with `src.model.load_model("checkpoints/v3.pt", device)`.

## Data
The PolyU Contactless 2D Fingerprint Database is **not included** (licensed by The Hong Kong Polytechnic University; request access from the PolyU Biometrics Research Centre). The exact split is defined by `results/split.json` and `results/data/images.csv`. Expected layout: see REPRODUCIBILITY.md.

## Setup at a glance
PolyU Contactless 2D (S1 enrolled vs S2 probe) · subject-disjoint split 202/84/50 fingers (seed 42) · ResNet-18 + ArcFace (s = 30, m = 0.40), 224×352 input · NFIQ2 v2.3 (MCLFIQ under v2.2 as ablation) · PGD (50 steps, step 2.5ε/50) with robustness radius ε\* · pre-specified primary test: OLS of log ε\* with log-margin control · Firth logistic for success at fixed ε · 1,000-resample finger-clustered bootstrap.

## Protocol decisions (and deviations from the proposal)
- **Split**: subject-disjoint, SEED=42: 202/84/50 fingers = 1,368/1,008/600 images (proposal table listed 2,424/2,016/1,200).
- **Input size**: 224x352 (H x W), native 350x225 aspect preserved; no ROI crop (fingertip fills the frame).
- **Impostors (val/test)**: all cross-session pairs (S1 enrolled vs S2 probe, different fingers): 250,992 val / 88,200 test, so tau controls FAR in the attacked scenario.
- **Thresholds**: `tau_fixed` at FAR 0.1%, `tau_floor` at FAR 0.2%, both from val impostors, conservative quantile (`method="higher"`).
- **Model selection**: by val EER on cross pairs (`--select eer`); a 6-run sweep (seeds, ArcFace margin, regularisation, LR) stayed within seed noise, so v3 was kept.
- **Quality**: contactless -> contact-like (local mean/std normalisation sigma=8, inverted, PNG tagged 500 DPI; ridge period ~8.6 px is already about 500 PPI). NFIQ2 v2.3 is primary. MCLFIQ runs under NFIQ2 v2.2 (it was trained on v2.2 features; under v2.3 it inverted). Tier cutoffs = val S2-probe tertiles. Failures: Q=0, flagged.
- **Attack budgets**: PGD saturates at the proposal grid {1,2,4,8}/255 (100% ASR), so a fine grid {0.01..0.15}/255 and a per-pair robustness radius eps* (bisection) were added. Sub-1/255 budgets are robustness distances in continuous pixel space, not storable 8-bit attacks.
- **Stage 1**: pre-specified primary test = log eps* (OLS) with log-margin control (from eps* ≈ M/||grad S||_1); Firth logistic for success at fixed eps and the other margin controls are robustness checks. 1,000-resample finger-clustered bootstrap. Finger fixed-effects check. Replicated on val.
- **Stage 2**: tau(Q) = clip(tau_fixed - alpha*max(0, Q_high - Q), tau_floor, tau_fixed); NFIQ2 failures keep tau_fixed. alpha and Q_high tuned on val (max Q1 TAR under attack at eps=0.05/255, val FAR <= 0.15%). Added baseline B2-global = one threshold at B2's val FAR.

## Additional checks
- Attack budget (Stage 1): after log-margin control, the NFIQ2 odds ratio for success is below 1 at eps = 0.02, 0.03, 0.05, 0.075/255 on test (all CIs exclude 1) and at 3 of 4 on validation.
- Attack strength: 5 restarts x 100 steps on 100 random accepted test pairs left eps* unchanged (median ratio 1.000, IQR 0.995-1.000, Spearman 1.000; ASR 73.0% vs 73.0%).
- attack.py checks recomputed clean scores against calibrate.py (tolerance 1e-2; GPU models differ at about 1e-3 because of TF32).

## Commit history
The full history with hashes and timestamps is listed in the report (appendix) and in `git log`. The release used for the report is tag **`v1.0`**.
