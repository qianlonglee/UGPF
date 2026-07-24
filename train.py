import os
import torch
import torch.nn.functional as F
from torchmetrics.functional import structural_similarity_index_measure
from torchvision.utils import save_image

from model import LYT_DualBranch
from dataloader import TriPairedDataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms


def calculate_psnr(img1, img2, max_pixel_value=1.0, gt_mean=True):
    if gt_mean:
        img1_gray = img1.mean(axis=1)
        img2_gray = img2.mean(axis=1)
        mean_restored = img1_gray.mean()
        mean_target = img2_gray.mean()
        img1 = torch.clamp(img1 * (mean_target / (mean_restored + 1e-12)), 0, 1)
    mse = F.mse_loss(img1, img2, reduction='mean')
    if mse.item() == 0:
        return float('inf')
    psnr = 20 * torch.log10(max_pixel_value / torch.sqrt(mse))
    return psnr.item()


def calculate_ssim(img1, img2, max_pixel_value=1.0, gt_mean=True):
    if gt_mean:
        img1_gray = img1.mean(axis=1, keepdim=True)
        img2_gray = img2.mean(axis=1, keepdim=True)
        mean_restored = img1_gray.mean()
        mean_target = img2_gray.mean()
        img1 = torch.clamp(img1 * (mean_target / (mean_restored + 1e-12)), 0, 1)
    ssim_val = structural_similarity_index_measure(img1, img2, data_range=max_pixel_value)
    return ssim_val.item()


@torch.no_grad()
def evaluate(model, dataloader, device, out_dir=None):
    model.eval()
    psnr_v = 0.0
    ssim_v = 0.0
    psnr_i = 0.0
    n = 0

    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)

    for idx, (v_low, v_gt, i_noisy, i_gt) in enumerate(dataloader):
        v_low, v_gt = v_low.to(device), v_gt.to(device)
        i_noisy, i_gt = i_noisy.to(device), i_gt.to(device)
        v_hat, i_hat, _, _ = model(v_low, i_noisy)
        v_hat = torch.clamp(v_hat, 0, 1)
        i_hat = torch.clamp(i_hat, 0, 1)

        psnr_v += calculate_psnr(v_hat, v_gt)
        ssim_v += calculate_ssim(v_hat, v_gt)
        psnr_i += calculate_psnr(i_hat, i_gt, gt_mean=False)
        n += 1

        if out_dir is not None:
            save_image(v_hat, os.path.join(out_dir, f"vis_{idx:04d}.png"))
            save_image(i_hat, os.path.join(out_dir, f"ir_{idx:04d}.png"))

    return {
        "psnr_vis": psnr_v / max(n, 1),
        "ssim_vis": ssim_v / max(n, 1),
        "psnr_ir": psnr_i / max(n, 1),
    }


def main():
    # ---------------------------------------------------------------------
    # Test Paths (edit as needed)
    # ---------------------------------------------------------------------
    TEST_VIS_LOW_DIR = "/home/msia/Leeql/LYT-Net/filtered_datasets/Visible_LQ"
    TEST_VIS_HIGH_DIR = "/home/msia/Leeql/LYT-Net/filtered_datasets/Visible_HQ"
    TEST_IR_NOISY_DIR = "/home/msia/Leeql/LYT-Net/filtered_datasets/Infrared"
    TEST_IR_CLEAN_DIR = "/home/msia/Leeql/LYT-Net/filtered_datasets/Infrared_gt"

    weights_path = "runs_dual_restoration/best_model.pth"
    results_dir = "results_dual_restoration"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    transform = transforms.Compose([transforms.ToTensor()])
    test_dataset = TriPairedDataset(
        TEST_VIS_LOW_DIR,
        TEST_VIS_HIGH_DIR,
        TEST_IR_NOISY_DIR,
        TEST_IR_CLEAN_DIR,
        transform=transform,
        crop_size=None,
        training=False,
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)

    # Keep model config consistent with training.
    model = LYT_DualBranch(
        vis_filters=32,
        ir_filters=32,
        ir_n_resblocks=1,
        ir_init_scale=0.5,
        ir_max_scale=2.0,
    ).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    print(f"Loaded weights: {weights_path}")

    metrics = evaluate(model, test_loader, device, out_dir=results_dir)
    print(
        f"Test | PSNR(V)={metrics['psnr_vis']:.3f} SSIM(V)={metrics['ssim_vis']:.4f} | "
        f"PSNR(IR)={metrics['psnr_ir']:.3f}"
    )


if __name__ == "__main__":
    main()
