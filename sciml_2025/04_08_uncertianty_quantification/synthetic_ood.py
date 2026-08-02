# synthetic_ood.py
# One-split, MNIST-compatible synthetic OOD dataset (labels are always -1).
# Usage:
#   from synthetic_ood import OODSet
#   from torchvision import transforms
#   from torch.utils.data import DataLoader
#
#   transform = transforms.Compose([
#       transforms.ToTensor(),
#       transforms.Normalize((0.5,), (0.5,))  # or (0.1307,), (0.3081,) for MNIST stats
#   ])
#
#   oodset = OODSet(root="./data", download=True, kinds=["gaussian","horizontal_lines","spiral","checkerboard"],
#                   size_per_kind=1000, base_seed=0, transform=transform)
#   ood_loader = DataLoader(oodset, batch_size=64, shuffle=False)

import math
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.datasets.vision import VisionDataset


# ---------- Primitive generators (return float tensor in [0,1], shape [H,W]) ----------
def gen_gaussian(g: torch.Generator, H: int = 28, W: int = 28) -> torch.Tensor:
    x = torch.randn(H, W, generator=g) * 0.25 + 0.5
    return x.clamp(0.0, 1.0)


def gen_horizontal_lines(g: torch.Generator, H: int = 28, W: int = 28) -> torch.Tensor:
    x = torch.zeros(H, W)
    num = torch.randint(3, 10, (1,), generator=g).item()
    thick = torch.randint(1, 3, (1,), generator=g).item()
    for _ in range(num):
        row = torch.randint(0, H, (1,), generator=g).item()
        val = 0.7 + 0.3 * torch.rand((), generator=g).item()
        x[row : row + thick, :] = val

    # Mild blur to reduce aliasing
    k = torch.tensor([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=torch.float32)
    k = k / k.sum()
    x = F.conv2d(x.view(1, 1, H, W), k.view(1, 1, 3, 3), padding=1).squeeze()
    return x.clamp(0.0, 1.0)


def gen_spiral(g: torch.Generator, H: int = 28, W: int = 28) -> torch.Tensor:
    x = torch.zeros(H, W)
    cx, cy = (W - 1) / 2.0, (H - 1) / 2.0
    turns = 3.0
    T = 800
    max_r = min(H, W) / 2.0 - 2.0

    for t in torch.linspace(0, turns * 2 * math.pi, T):
        tf = float(t)
        r = (tf / (turns * 2 * math.pi)) * max_r
        px = cx + r * math.cos(tf)
        py = cy + r * math.sin(tf)
        ix, iy = int(round(px)), int(round(py))
        if 0 <= ix < W and 0 <= iy < H:
            x[iy, ix] = 1.0
            # Thicken stroke a bit
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    jx, jy = ix + dx, iy + dy
                    if 0 <= jx < W and 0 <= jy < H:
                        x[jy, jx] = max(x[jy, jx], torch.tensor(0.8))

    # Light blur to reduce aliasing
    k = torch.tensor([[0, 1, 0], [1, 4, 1], [0, 1, 0]], dtype=torch.float32)
    k = k / k.sum()
    x = F.conv2d(x.view(1, 1, H, W), k.view(1, 1, 3, 3), padding=1).squeeze()
    return x.clamp(0.0, 1.0)


def gen_checkerboard(g: torch.Generator, H: int = 28, W: int = 28, cell: Optional[int] = None) -> torch.Tensor:
    x = torch.zeros(H, W, dtype=torch.float32)
    cell = cell or torch.randint(2, 6, (1,), generator=g).item()  # 2..5 px
    phase_x = torch.randint(0, 2, (1,), generator=g).item()
    phase_y = torch.randint(0, 2, (1,), generator=g).item()
    yy, xx = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    tiles = ((xx // cell + phase_x) + (yy // cell + phase_y)) % 2
    bright = 0.8 + 0.2 * torch.rand((), generator=g).item()
    dark = 0.05 + 0.15 * torch.rand((), generator=g).item()
    x = torch.where(tiles == 0, torch.full_like(x, bright), torch.full_like(x, dark))

    # Mild blur
    k = torch.tensor([[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]], dtype=torch.float32)
    k = k / k.sum()
    x = F.conv2d(x.view(1, 1, H, W), k.view(1, 1, 3, 3), padding=1).squeeze()
    return x.clamp(0.0, 1.0)


GENS = {
    "gaussian": gen_gaussian,
    "horizontal_lines": gen_horizontal_lines,
    "spiral": gen_spiral,
    "checkerboard": gen_checkerboard,
}


class OODSet(VisionDataset):
    """
    Single-split synthetic OOD dataset (all labels are -1), MNIST-compatible.

    Directory layout:
      root/
        SyntheticOOD/
          processed/
            ood.pt

    The .pt file stores a tuple:
      (data_uint8 [N,28,28], targets_long [N])

    Parameters
    ----------
    root : str
        Root directory for storage.
    download : bool
        If True, generate and save to disk if needed.
    kinds : List[str]
        Which generators to include. Default: all four kinds.
    size_per_kind : int
        Number of samples per kind (total N = len(kinds) * size_per_kind).
    base_seed : int
        Base seed used to derive per-kind seeds for determinism.
    transform, target_transform : torchvision transforms
        Same semantics as MNIST. Images are returned as PIL, then transformed.
    """

    ood_file = "ood.pt"

    def __init__(
        self,
        root: str = "./data",
        download: bool = False,
        kinds: Optional[List[str]] = None,
        size_per_kind: int = 1000,
        base_seed: int = 0,
        transform=None,
        target_transform=None,
    ):
        kinds = kinds or ["gaussian", "horizontal_lines", "spiral", "checkerboard"]
        for k in kinds:
            if k not in GENS:
                raise ValueError(f"Unknown kind: {k}. Allowed: {list(GENS.keys())}")

        super().__init__(root, transform=transform, target_transform=target_transform)
        self.kinds = kinds
        self.size_per_kind = int(size_per_kind)
        self.base_seed = int(base_seed)

        self._base_folder = Path(root) / "SyntheticOOD"
        self.processed_folder = self._base_folder / "processed"
        self.processed_folder.mkdir(parents=True, exist_ok=True)

        if download:
            self.download()

        path = self.processed_folder / self.ood_file
        if not path.exists():
            raise RuntimeError(f"{path} not found. Pass download=True to generate.")
        self.data, self.targets = torch.load(path, map_location="cpu")

    def __len__(self) -> int:
        return self.data.size(0)

    def __getitem__(self, index: int):
        img_u8 = self.data[index]  # uint8 [28,28]
        target = int(self.targets[index].item())  # always -1

        # MNIST returns a PIL Image; do the same
        img = Image.fromarray(img_u8.numpy(), mode="L")

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    # ---------- generation & saving ----------
    def download(self) -> None:
        ood_path = self.processed_folder / self.ood_file
        if ood_path.exists():
            return  # already generated

        data, targets = self._generate_all()
        torch.save((data, targets), ood_path)

    def _generate_all(self):
        H = W = 28
        all_imgs = []
        all_labels = []

        for kind_idx, kind in enumerate(self.kinds):
            # Stable seed per kind
            seed = self.base_seed + 97 * kind_idx
            g = torch.Generator().manual_seed(seed)

            imgs = self._gen_many(kind, self.size_per_kind, g, H=H, W=W)  # float [N,28,28] in [0,1]
            imgs_u8 = (imgs * 255.0).round().clamp(0, 255).to(torch.uint8)  # like MNIST storage
            all_imgs.append(imgs_u8)

            labels = torch.full((self.size_per_kind,), -1, dtype=torch.long)  # all -1
            all_labels.append(labels)

        data = torch.cat(all_imgs, dim=0)      # [N_total, 28, 28]
        targets = torch.cat(all_labels, dim=0) # [N_total]
        return data, targets

    @staticmethod
    def _gen_many(kind: str, n: int, g: torch.Generator, H: int, W: int) -> torch.Tensor:
        fn = GENS[kind]
        imgs = []
        for _ in range(n):
            x = fn(g, H=H, W=W)  # float [H,W] in [0,1]
            imgs.append(x.unsqueeze(0))
        return torch.cat(imgs, dim=0)  # [N, H, W]


__all__ = ["OODSet"]
