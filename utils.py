import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torchvision import transforms
from PIL import Image

from model import LYT_DualBranch
from fusion_net import FusionNet
from utils import norm01, load_state_dict_safely


# ======================================================================
# (MANUAL PATHS) - edit these 5 lines only
# ======================================================================
VIS_LOW_DIR = "/home/msia/Leeql/LYT-Net/filtered_datasets/Visible_LQ"      # RGB low-light visible images
IR_NOISY_DIR = "/home/msia/Leeql/LYT-Net/filtered_datasets/Infrared_gt_noisy1"    # grayscale noisy infrared images
RESTORER_CKPT = "/home/msia/Leeql/LYTFusion/V4.3.1/runs_1/best_model.pth"  # Stage-1 ckpt
FUSION_CKPT = "/home/msia/Leeql/LYTFusion/ugpf_stage2_lite/runs_fusion_stage2_2/checkpoints/fusion_last.pth"                   # Stage-2 ckpt
OUT_DIR = "./fusion_outputs_2"                                # output directory


def save_gray01(t: torch.Tensor, path: str):
    """Save a (1,1,H,W) or (1,H,W) tensor in [0,1] as 8-bit grayscale PNG."""
    if t.dim() == 4:
        t = t[0, 0]
    elif t.dim() == 3:
        t = t[0]
    t = t.detach().float().cpu().clamp(0, 1)
    arr = (t.numpy() * 255.0 + 0.5).astype(np.uint8)
    Image.fromarray(arr, mode="L").save(path)


@torch.no_grad()
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    out = Path(OUT_DIR)
    out.mkdir(parents=True, exist_ok=True)
    (out / "vis_hat").mkdir(parents=True, exist_ok=True)
    (out / "ir_hat").mkdir(parents=True, exist_ok=True)
    (out / "fused").mkdir(parents=True, exist_ok=True)
    (out / "fused_y").mkdir(parents=True, exist_ok=True)

    # ------------------- Load models -------------------
    restorer = LYT_DualBranch().to(device)
    load_state_dict_safely(restorer, RESTORER_CKPT)
    restorer.eval()

    # Lite conv-only FusionNet
    fuse_net = FusionNet(base_channels=32, res_scale=0.6, gate_smooth=True).to(device)
    load_state_dict_safely(fuse_net, FUSION_CKPT, key="fusion")
    fuse_net.eval()

    tf = transforms.ToTensor()

    vis_files = sorted([p for p in Path(VIS_LOW_DIR).iterdir() if p.is_file()])
    ir_map = {p.stem: p for p in Path(IR_NOISY_DIR).iterdir() if p.is_file()}

    matched = [vf for vf in vis_files if vf.stem in ir_map]
    if len(matched) == 0:
        raise RuntimeError("No matched filenames between VIS_LOW_DIR and IR_NOISY_DIR.")

    print(f"Matched pairs: {len(matched)}")

    for vf in matched:
        ip = ir_map[vf.stem]

        vis = tf(Image.open(vf).convert("RGB"))
        ir_img = Image.open(ip)
        ir = tf(ir_img if ir_img.mode in ["L", "I;16", "I"] else ir_img.convert("L"))

        vis = norm01(vis).unsqueeze(0).to(device)
        ir = norm01(ir).unsqueeze(0).to(device)

        vis_hat, ir_hat, vis_conf, ir_conf = restorer(vis, ir)
        fused_y, fused_rgb = fuse_net(vis_hat, ir_hat, vis_conf, ir_conf)

        # Preserve original extension
        ext = vf.suffix  # e.g. .png/.jpg
        name = vf.name   # full filename

        # Save outputs (name stays identical)
        # - For IR grayscale, always save PNG to avoid JPEG loss
        fused_path = out / "fused" / name
        vis_hat_path = out / "vis_hat" / name
        ir_hat_path = out / "ir_hat" / f"{vf.stem}.png"
        fused_y_path = out / "fused_y" / f"{vf.stem}.png"

        # VIS & Fused are RGB
        from torchvision.utils import save_image
        save_image(vis_hat.clamp(0, 1), str(vis_hat_path))
        save_image(fused_rgb.clamp(0, 1), str(fused_path))

        # IR outputs as grayscale PNG
        save_gray01(ir_hat, str(ir_hat_path))
        save_gray01(fused_y, str(fused_y_path))

    print(f"Done. Outputs saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
