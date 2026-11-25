import math, torch, torch.nn as nn, torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights

class Embedder(nn.Module):
    def __init__(self, dim=128, pretrained=True, dropout=0.0, bn=False):
        super().__init__()
        self.net = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
        if dropout > 0 or bn:                                  # regularised embedding head
            layers = [nn.Dropout(dropout), nn.Linear(512, dim)] + ([nn.BatchNorm1d(dim)] if bn else [])
            self.net.fc = nn.Sequential(*layers)
        else:                                                  # same layout as the first checkpoint
            self.net.fc = nn.Linear(512, dim)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
    def forward(self, x):                                      # x: Bx1xHxW in [0,1]  (the attack works in this space)
        x = (x.repeat(1, 3, 1, 1) - self.mean) / self.std
        return F.normalize(self.net(x), dim=1)

class ArcFace(nn.Module):
    def __init__(self, dim, n_classes, s=30.0, m=0.40):
        super().__init__()
        self.W = nn.Parameter(torch.randn(n_classes, dim) * 0.01)
        self.s, self.m = s, m
    def forward(self, emb, y, m=None):
        m = self.m if m is None else m
        cos = F.linear(emb, F.normalize(self.W)).clamp(-1 + 1e-7, 1 - 1e-7)
        phi = torch.cos(torch.acos(cos) + m)
        phi = torch.where(cos > math.cos(math.pi - m), phi, cos - m * math.sin(m))
        onehot = F.one_hot(y, cos.size(1)).bool()
        return self.s * torch.where(onehot, phi, cos)

def load_model(path, device):
    """Loads either the old checkpoint (plain state_dict) or the new one ({'state', 'cfg'})."""
    ck = torch.load(path, map_location=device, weights_only=True)   # tensors + plain cfg dict only
    state, cfg = (ck["state"], ck["cfg"]) if isinstance(ck, dict) and "state" in ck else (ck, {})
    m = Embedder(pretrained=False, **cfg).to(device)
    m.load_state_dict(state)
    return m.eval()