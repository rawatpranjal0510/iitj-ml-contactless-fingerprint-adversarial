from pathlib import Path
DATA_ROOT = Path("/iitjhome/m24ma2008/Pranjal/MLProject/DataSet/extracted_images/processed_contactless_2d_fingerprint_images")
OUT = Path(__file__).resolve().parent / "out"
SEED, BATCH = 42, 64
IMG_SIZE = (224, 352)                     # (H, W): keeps the native 350x225 aspect ratio
N_VAL, N_TEST = 84, 50
FAR = 1e-3                                # tau_fixed: FAR 0.1% on val impostors
FAR_FLOOR = 2e-3                          # tau_floor: safety floor, FAR <= 0.2% on val impostors
for d in [OUT, OUT / "ckpt", OUT / "figs", OUT / "runs"]:
    d.mkdir(parents=True, exist_ok=True)
