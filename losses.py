import os
from typing import Optional
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import random

class PairedDataset(Dataset):
    def __init__(self, low_dir, high_dir, transform=None, crop_size=None, training=True):
        self.low_dir = low_dir
        self.high_dir = high_dir
        self.transform = transform
        self.crop_size = crop_size
        self.training = training

        self.low_images = sorted([f for f in os.listdir(low_dir) if os.path.isfile(os.path.join(low_dir, f))])
        self.high_images = sorted([f for f in os.listdir(high_dir) if os.path.isfile(os.path.join(high_dir, f))])

        assert len(self.low_images) == len(self.high_images), "Mismatch in number of images"

    def __len__(self):
        return len(self.low_images)

    def __getitem__(self, idx):
        low_image_path = os.path.join(self.low_dir, self.low_images[idx])
        high_image_path = os.path.join(self.high_dir, self.high_images[idx])

        low_image = Image.open(low_image_path).convert('RGB')
        high_image = Image.open(high_image_path).convert('RGB')

        if self.transform:
            low_image = self.transform(low_image)
            high_image = self.transform(high_image)

        if self.training and self.crop_size:
            i, j, h, w = transforms.RandomCrop.get_params(low_image, output_size=(self.crop_size, self.crop_size))
            low_image = transforms.functional.crop(low_image, i, j, h, w)
            high_image = transforms.functional.crop(high_image, i, j, h, w)

        return low_image, high_image


class TriPairedDataset(Dataset):
    """Paired dataset for dual-branch restoration.

    Each sample contains:
      - low-light visible RGB (vis_low)
      - normal-light visible RGB ground truth (vis_high)
      - noisy infrared (ir_noisy) (grayscale)
      - clean infrared ground truth (ir_clean) (grayscale)

    All items are cropped with the SAME random crop (if enabled) to preserve alignment.
    """

    def __init__(
        self,
        vis_low_dir: str,
        vis_high_dir: str,
        ir_noisy_dir: str,
        ir_clean_dir: str,
        transform=None,
        crop_size: Optional[int] = None,
        training: bool = True,
        strict_filename_match: bool = False,
    ):
        self.vis_low_dir = vis_low_dir
        self.vis_high_dir = vis_high_dir
        self.ir_noisy_dir = ir_noisy_dir
        self.ir_clean_dir = ir_clean_dir
        self.transform = transform
        self.crop_size = crop_size
        self.training = training

        def list_files(d):
            return [f for f in os.listdir(d) if os.path.isfile(os.path.join(d, f))]

        vis_low = list_files(vis_low_dir)
        vis_high = list_files(vis_high_dir)
        ir_noisy = list_files(ir_noisy_dir)
        ir_clean = list_files(ir_clean_dir)

        def stem(f):
            return os.path.splitext(f)[0]

        map_low = {stem(f): f for f in vis_low}
        map_high = {stem(f): f for f in vis_high}
        map_in = {stem(f): f for f in ir_noisy}
        map_ic = {stem(f): f for f in ir_clean}

        common = sorted(set(map_low) & set(map_high) & set(map_in) & set(map_ic))
        if len(common) == 0:
            raise RuntimeError(
                "No common filenames across vis_low/vis_high/ir_noisy/ir_clean. "
                "Make sure the 4 folders contain paired images with matching base filenames."
            )

        if strict_filename_match:
            # Optional strict check: all folders must have the same number of files and the same stems.
            all_stems = [set(map_low), set(map_high), set(map_in), set(map_ic)]
            if not (all_stems[0] == all_stems[1] == all_stems[2] == all_stems[3]):
                raise RuntimeError(
                    "Strict filename match failed: the 4 folders do not contain identical filename stems. "
                    "Either fix the dataset or set strict_filename_match=False."
                )

        self.keys = common
        self.low_files = [map_low[k] for k in self.keys]
        self.high_files = [map_high[k] for k in self.keys]
        self.ir_noisy_files = [map_in[k] for k in self.keys]
        self.ir_clean_files = [map_ic[k] for k in self.keys]

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        vis_low_path = os.path.join(self.vis_low_dir, self.low_files[idx])
        vis_high_path = os.path.join(self.vis_high_dir, self.high_files[idx])
        ir_noisy_path = os.path.join(self.ir_noisy_dir, self.ir_noisy_files[idx])
        ir_clean_path = os.path.join(self.ir_clean_dir, self.ir_clean_files[idx])

        vis_low = Image.open(vis_low_path).convert('RGB')
        vis_high = Image.open(vis_high_path).convert('RGB')
        # Keep 16-bit infrared if present (e.g., mode "I;16"); otherwise convert to 8-bit grayscale.
        ir_noisy_img = Image.open(ir_noisy_path)
        ir_clean_img = Image.open(ir_clean_path)
        ir_noisy = ir_noisy_img if ir_noisy_img.mode in ["L", "I;16", "I"] else ir_noisy_img.convert('L')
        ir_clean = ir_clean_img if ir_clean_img.mode in ["L", "I;16", "I"] else ir_clean_img.convert('L')

        if self.transform:
            vis_low = self.transform(vis_low)
            vis_high = self.transform(vis_high)
            ir_noisy = self.transform(ir_noisy)
            ir_clean = self.transform(ir_clean)

        # ---------------------------------------------------------------------
        # Robust normalization to [0,1]
        # Some datasets store images as 8-bit but already converted to float,
        # or store infrared as 16-bit/32-bit. torchvision.ToTensor() handles
        # most PIL modes, but if your inputs are saved in unusual formats
        # (e.g., float PNG/TIF), values may be >1 and will explode the loss
        # terms (histogram, ms-ssim) into NaNs. We defensively rescale.
        # ---------------------------------------------------------------------
        def _norm01(t):
            t = t.float()
            mx = float(t.max())
            if mx > 1.5:
                # Heuristic: 8-bit-like vs 16-bit-like
                denom = 65535.0 if mx > 300 else 255.0
                t = t / denom
            return t.clamp(0.0, 1.0)

        vis_low = _norm01(vis_low)
        vis_high = _norm01(vis_high)
        ir_noisy = _norm01(ir_noisy)
        ir_clean = _norm01(ir_clean)

        # Ensure spatial sizes match for cropping (H,W)
        def hw(x):
            return (x.shape[-2], x.shape[-1])
        if hw(vis_low) != hw(vis_high) or hw(vis_low) != hw(ir_noisy) or hw(vis_low) != hw(ir_clean):
            raise RuntimeError(
                f"Size mismatch among paired inputs for key '{self.keys[idx]}': "
                f"vis_low{hw(vis_low)}, vis_high{hw(vis_high)}, ir_noisy{hw(ir_noisy)}, ir_clean{hw(ir_clean)}"
            )

        if self.training and self.crop_size:
            i, j, h, w = transforms.RandomCrop.get_params(vis_low, output_size=(self.crop_size, self.crop_size))
            vis_low = transforms.functional.crop(vis_low, i, j, h, w)
            vis_high = transforms.functional.crop(vis_high, i, j, h, w)
            ir_noisy = transforms.functional.crop(ir_noisy, i, j, h, w)
            ir_clean = transforms.functional.crop(ir_clean, i, j, h, w)

        return vis_low, vis_high, ir_noisy, ir_clean

def create_dataloaders(train_low, train_high, test_low, test_high, crop_size=256, batch_size=1):
    transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    train_loader = None
    test_loader = None
    
    if train_low and train_high:
        train_dataset = PairedDataset(train_low, train_high, transform=transform, crop_size=crop_size, training=True)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    if test_low and test_high:
        test_dataset = PairedDataset(test_low, test_high, transform=transform, training=False)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)

    return train_loader, test_loader


def create_dual_dataloaders(
    train_vis_low: str,
    train_vis_high: str,
    train_ir_noisy: str,
    train_ir_clean: str,
    test_vis_low: Optional[str] = None,
    test_vis_high: Optional[str] = None,
    test_ir_noisy: Optional[str] = None,
    test_ir_clean: Optional[str] = None,
    crop_size: int = 256,
    batch_size: int = 1,
    num_workers: int = 4,
):
    """Create dataloaders for dual-branch restoration training.

    For training you MUST provide 4 directories.
    For testing/validation, you may optionally provide 4 directories as well.
    """

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_dataset = TriPairedDataset(
        train_vis_low,
        train_vis_high,
        train_ir_noisy,
        train_ir_clean,
        transform=transform,
        crop_size=crop_size,
        training=True,
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    test_loader = None
    if all(p is not None for p in [test_vis_low, test_vis_high, test_ir_noisy, test_ir_clean]):
        test_dataset = TriPairedDataset(
            test_vis_low,
            test_vis_high,
            test_ir_noisy,
            test_ir_clean,
            transform=transform,
            crop_size=None,
            training=False,
        )
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader
