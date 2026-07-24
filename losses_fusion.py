import os
from typing import Optional, List, Tuple

from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


class VisIrPairedDataset(Dataset):
    """Paired dataset for Stage-2 Fusion (no GT required).

    Each sample:
      - vis_low: RGB (B,3,H,W) in [0,1]
      - ir_noisy: grayscale (B,1,H,W) in [0,1]

    Pairing rule: match by *stem* (filename without extension).
    Cropping: same random crop applied to both to preserve alignment.
    """

    def __init__(
        self,
        vis_low_dir: str,
        ir_noisy_dir: str,
        transform=None,
        crop_size: Optional[int] = None,
        training: bool = True,
        strict_filename_match: bool = False,
    ):
        self.vis_low_dir = vis_low_dir
        self.ir_noisy_dir = ir_noisy_dir
        self.transform = transform
        self.crop_size = crop_size
        self.training = training

        def list_files(d: str) -> List[str]:
            return [f for f in os.listdir(d) if os.path.isfile(os.path.join(d, f))]

        def stem(f: str) -> str:
            return os.path.splitext(f)[0]

        vis_files = list_files(vis_low_dir)
        ir_files = list_files(ir_noisy_dir)

        map_v = {stem(f): f for f in vis_files}
        map_i = {stem(f): f for f in ir_files}

        common = sorted(set(map_v) & set(map_i))
        if len(common) == 0:
            raise RuntimeError(
                "No common filenames across vis_low/ir_noisy. "
                "Make sure the 2 folders contain paired images with matching base filenames."
            )

        if strict_filename_match and set(map_v) != set(map_i):
            missing_v = sorted(set(map_i) - set(map_v))
            missing_i = sorted(set(map_v) - set(map_i))
            raise RuntimeError(
                "Strict filename match failed:\n"
                f"  stems in IR but not VIS: {missing_v[:10]} ...\n"
                f"  stems in VIS but not IR: {missing_i[:10]} ..."
            )

        self.keys = common
        self.vis_list = [map_v[k] for k in self.keys]
        self.ir_list = [map_i[k] for k in self.keys]

    def __len__(self) -> int:
        return len(self.keys)

    @staticmethod
    def _norm01(t: torch.Tensor) -> torch.Tensor:
        t = t.float()
        mx = float(t.max()) if t.numel() > 0 else 1.0
        if mx > 1.5:
            denom = 65535.0 if mx > 300 else 255.0
            t = t / denom
        return t.clamp(0.0, 1.0)

    @staticmethod
    def _hw(x: torch.Tensor) -> Tuple[int, int]:
        return (int(x.shape[-2]), int(x.shape[-1]))

    def __getitem__(self, idx: int):
        vis_path = os.path.join(self.vis_low_dir, self.vis_list[idx])
        ir_path = os.path.join(self.ir_noisy_dir, self.ir_list[idx])

        vis = Image.open(vis_path).convert("RGB")
        ir_img = Image.open(ir_path)
        ir = ir_img if ir_img.mode in ["L", "I;16", "I"] else ir_img.convert("L")

        if self.transform is not None:
            vis = self.transform(vis)
            ir = self.transform(ir)

        vis = self._norm01(vis)
        ir = self._norm01(ir)

        # Ensure spatial match before cropping
        if self._hw(vis) != self._hw(ir):
            raise RuntimeError(
                f"Size mismatch for key '{self.keys[idx]}': vis{self._hw(vis)} vs ir{self._hw(ir)}"
            )

        if self.training and self.crop_size is not None:
            i, j, h, w = transforms.RandomCrop.get_params(vis, output_size=(self.crop_size, self.crop_size))
            vis = transforms.functional.crop(vis, i, j, h, w)
            ir = transforms.functional.crop(ir, i, j, h, w)

        return vis, ir


def create_fusion_dataloader(
    vis_low_dir: str,
    ir_noisy_dir: str,
    crop_size: int = 256,
    batch_size: int = 8,
    num_workers: int = 4,
    training: bool = True,
    strict_filename_match: bool = False,
):
    tf = transforms.Compose([transforms.ToTensor()])

    dataset = VisIrPairedDataset(
        vis_low_dir=vis_low_dir,
        ir_noisy_dir=ir_noisy_dir,
        transform=tf,
        crop_size=crop_size if training else None,
        training=training,
        strict_filename_match=strict_filename_match,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=training,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=training,
    )
    return loader
