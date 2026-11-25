import numpy as np, pandas as pd, torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.v2 as T
from config import DATA_ROOT, IMG_SIZE

def load_img(path):
    im = Image.open(DATA_ROOT / path).convert("L").resize(IMG_SIZE[::-1], Image.BILINEAR)
    return torch.from_numpy(np.array(im)).unsqueeze(0)      # np.array copies -> writable, no warning; uint8 1xHxW

def u(a, b):                                                # uniform sample from the torch RNG (safe with DataLoader workers)
    return float(torch.empty(1).uniform_(a, b))

class Strong:
    """Contactless-style variation: pose, defocus, illumination, contrast, sensor noise, occlusion.
    No flips: a mirrored ridge pattern is a different finger, not the same one."""
    def __init__(self):
        self.geo = T.RandomAffine(degrees=15, translate=(0.08, 0.08), scale=(0.9, 1.1), fill=0)
        self.blur = T.RandomApply([T.GaussianBlur(5, sigma=(0.1, 2.0))], p=0.3)
        self.erase = T.RandomErasing(p=0.25, scale=(0.02, 0.12), value=0)
    def __call__(self, x):                                  # x: 1xHxW float in [0,1]
        x = self.blur(self.geo(x)).clamp(0, 1)
        x = x ** u(0.7, 1.4)                                # gamma (illumination)
        m = x.mean(); x = (x - m) * u(0.7, 1.3) + m         # contrast
        x = x * u(0.8, 1.2)                                 # brightness
        x = x + torch.randn_like(x) * u(0.0, 0.03)          # sensor noise
        return self.erase(x.clamp(0, 1))

class Light:                                                # the original augmentation, kept for ablation
    def __call__(self, x):
        x = torch.roll(x, shifts=(int(u(-8, 8)), int(u(-8, 8))), dims=(1, 2))
        return (x * u(0.85, 1.15)).clamp(0, 1)

AUGS = {"none": None, "light": Light, "strong": Strong}

class Imgs(Dataset):
    """Loads the chosen rows of images.csv into memory once. Returns float in [0,1]."""
    def __init__(self, df, rows, labels=None, aug="none"):
        self.rows = list(rows)
        self.x = torch.stack([load_img(df.path[i]) for i in self.rows])
        self.y = None if labels is None else torch.tensor([labels[df.finger[i]] for i in self.rows])
        self.aug = AUGS[aug]() if AUGS[aug] else None
    def __len__(self): return len(self.rows)
    def __getitem__(self, k):
        x = self.x[k].float() / 255
        if self.aug: x = self.aug(x)
        return (x, self.y[k]) if self.y is not None else x