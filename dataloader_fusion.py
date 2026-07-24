import os
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchmetrics.functional import structural_similarity_index_measure
from torchvision.utils import save_image
import torchvision.transforms as transforms

from model import LYT_DualBranch
from losses import CombinedLoss, IRLoss, ConfidenceLoss, confidence_weighted_l1, confidence_weighted_grad_l1
from dataloader import TriPairedDataset
from torch.utils.data import DataLoader, Subset
import random


def calculate_psnr(img1, img2, max_pixel_value=1.0, gt_mean=True):
    """PSNR between two BCHW tensors in [0,1]."""
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
    """SSIM between two BCHW tensors in [0,1]."""
    if gt_mean:
        img1_gray = img1.mean(axis=1, keepdim=True)
        img2_gray = img2.mean(axis=1, keepdim=True)
        mean_restored = img1_gray.mean()
        mean_target = img2_gray.mean()
        img1 = torch.clamp(img1 * (mean_target / (mean_restored + 1e-12)), 0, 1)
    ssim_val = structural_similarity_index_measure(img1, img2, data_range=max_pixel_value)
    return ssim_val.item()


@torch.no_grad()
def validate(model, dataloader, device, gt_mean_vis=True, save_dir=None, save_first_n=0):
    model.eval()
    total_psnr_v = 0.0
    total_ssim_v = 0.0
    total_psnr_i = 0.0
    n = 0

    if save_dir is not None and save_first_n > 0:
        os.makedirs(save_dir, exist_ok=True)

    for idx, (v_low, v_gt, i_noisy, i_gt) in enumerate(dataloader):
        v_low, v_gt = v_low.to(device), v_gt.to(device)
        i_noisy, i_gt = i_noisy.to(device), i_gt.to(device)

        v_hat, i_hat, _, _ = model(v_low, i_noisy)
        v_hat = torch.clamp(v_hat, 0, 1)
        i_hat = torch.clamp(i_hat, 0, 1)

        total_psnr_v += calculate_psnr(v_hat, v_gt, gt_mean=gt_mean_vis)
        total_ssim_v += calculate_ssim(v_hat, v_gt, gt_mean=gt_mean_vis)
        total_psnr_i += calculate_psnr(i_hat, i_gt, gt_mean=False)
        n += 1

        if save_dir is not None and idx < save_first_n:
            save_image(v_hat, os.path.join(save_dir, f"vis_{idx:04d}.png"))
            save_image(i_hat, os.path.join(save_dir, f"ir_{idx:04d}.png"))

    return {
        "psnr_vis": total_psnr_v / max(n, 1),
        "ssim_vis": total_ssim_v / max(n, 1),
        "psnr_ir": total_psnr_i / max(n, 1),
    }


def main():
    # ---------------------------------------------------------------------
    # Dataset Paths (EDIT THESE 4 PATHS)
    # ---------------------------------------------------------------------
    TRAIN_VIS_LOW_DIR = "/home/msia/Leeql/EMS_full_dataset/EMS_dataset_train/vis_Low-Light/Visible_LQ"  # low-light visible images (RGB)
    TRAIN_VIS_HIGH_DIR = "/home/msia/Leeql/EMS_full_dataset/EMS_dataset_train/vis_Low-Light/Visible_HQ"  # normal-light visible GT (RGB)
    TRAIN_IR_NOISY_DIR = "/home/msia/Leeql/EMS_full_dataset/EMS_dataset_train/vis_Low-Light/Infrared_gt_noisy1"  # noisy infrared images (1ch)
    TRAIN_IR_CLEAN_DIR = "/home/msia/Leeql/EMS_full_dataset/EMS_dataset_train/vis_Low-Light/Infrared_gt"  # clean infrared GT (1ch)

    # (Optional) Validation paths: leave as None to validate on training set
    VAL_VIS_LOW_DIR = None
    VAL_VIS_HIGH_DIR = None
    VAL_IR_NOISY_DIR = None
    VAL_IR_CLEAN_DIR = None

    # ---------------------------------------------------------------------
    # Hyperparameters
    # ---------------------------------------------------------------------
    learning_rate = 2e-4
    num_epochs = 300
    crop_size = 256
    batch_size = 1
    lambda_ir = 0.5
    lambda_conf = 0.05  # weak supervision weight for confidence heads (keep small)
    ir_grad_weight = 1.0
    conf_alpha = 10.0   # larger => more peaky confidence target
    # ---------------- Second-round upgrades (robustness) ----------------
    # 1) Confidence-weighted reconstruction (reliability weighting). Keep small.
    lambda_w_vis = 0.05  # auxiliary weighted L1/grad for VIS
    lambda_w_ir = 0.05   # auxiliary weighted L1/grad for IR
    # 2) Content consistency regularization between two stochastic degradations.
    lambda_cons = 0.05   # suggested 0.05~0.1
    cons_sigma_vis = 0.02
    cons_sigma_ir = 0.02
    # Mixed precision (AMP) can easily produce NaNs with this model/loss
    # (ms-ssim, histogram soft assignment, and attention softmax).
    # Keep it OFF by default to match the original LYT-Net training.
    use_amp = False
    save_root = "runs_3"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}; LR: {learning_rate}; Epochs: {num_epochs}")

    # ---------------------------------------------------------------------
    # Dataloaders
    # ---------------------------------------------------------------------
        # ---------------------------------------------------------------------
    # Dataloaders (auto train/val split)
    # ---------------------------------------------------------------------
    # We keep ONLY 4 user-editable directories (the training set), and we
    # automatically split by filename stems into train/val. This avoids
    # validating on random-cropped training patches (which inflates PSNR)
    # and ensures the "best_model.pth" is selected using full-resolution images
    # comparable to test.py.
    VAL_RATIO = 0.1      # 10% for validation
    SPLIT_SEED = 42
    # NOTE: PSNR/SSIM for VIS uses mean-alignment by default in calculate_psnr/ssim.
    # Set GT_MEAN_VIS=False if you want standard PSNR/SSIM.
    GT_MEAN_VIS = True

    transform = transforms.Compose([transforms.ToTensor()])

    # Build deterministic full-res dataset to get stable ordering / keys
    base_dataset = TriPairedDataset(
        TRAIN_VIS_LOW_DIR, TRAIN_VIS_HIGH_DIR, TRAIN_IR_NOISY_DIR, TRAIN_IR_CLEAN_DIR,
        transform=transform, crop_size=None, training=False,
    )
    n = len(base_dataset)
    indices = list(range(n))
    rng = random.Random(SPLIT_SEED)
    rng.shuffle(indices)
    val_n = max(1, int(n * VAL_RATIO)) if VAL_RATIO > 0 else 0
    val_indices = indices[:val_n]
    train_indices = indices[val_n:]

    train_dataset = TriPairedDataset(
        TRAIN_VIS_LOW_DIR, TRAIN_VIS_HIGH_DIR, TRAIN_IR_NOISY_DIR, TRAIN_IR_CLEAN_DIR,
        transform=transform, crop_size=crop_size, training=True,
    )
    val_dataset = TriPairedDataset(
        TRAIN_VIS_LOW_DIR, TRAIN_VIS_HIGH_DIR, TRAIN_IR_NOISY_DIR, TRAIN_IR_CLEAN_DIR,
        transform=transform, crop_size=None, training=False,
    )

    train_loader = DataLoader(
        Subset(train_dataset, train_indices),
        batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        Subset(val_dataset, val_indices),
        batch_size=1, shuffle=False, num_workers=4, pin_memory=True
    )

    print(f"Train loader: {len(train_loader)}; Val loader: {len(val_loader)} (val_ratio={VAL_RATIO}, seed={SPLIT_SEED})")

    # ---------------------------------------------------------------------
    # Model & Optim
    # ---------------------------------------------------------------------
    # ------------------------------
    # Model capacity knobs (IR-Denoiser++ v4.3+)
    # ------------------------------
    VIS_FILTERS = 32
    IR_FILTERS = 32          # (B) increase IR denoiser width
    IR_N_RESBLOCKS = 1       # number of ResBlocks per scale (kept small)
    IR_INIT_SCALE = 0.5      # (A2) initial residual scale s in delta = s*tanh(.)
    IR_MAX_SCALE = 2.0       # safety clamp for s

    model = LYT_DualBranch(
        vis_filters=VIS_FILTERS,
        ir_filters=IR_FILTERS,
        ir_n_resblocks=IR_N_RESBLOCKS,
        ir_init_scale=IR_INIT_SCALE,
        ir_max_scale=IR_MAX_SCALE,
    ).to(device)
    # Visible loss MUST match the original LYT-Net prototype (same loss terms & perceptual backbone).
    # NOTE: In this repo, CombinedLoss forward signature is (y_true, y_pred).
    vis_criterion = CombinedLoss(device=device)
    ir_criterion = IRLoss(grad_weight=ir_grad_weight)
    conf_criterion = ConfidenceLoss(alpha=conf_alpha)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=(torch.cuda.is_available() and use_amp))

    run_dir = os.path.join(save_root)
    os.makedirs(run_dir, exist_ok=True)
    ckpt_path = os.path.join(run_dir, "best_model.pth")
    sample_dir = os.path.join(run_dir, "samples")
    os.makedirs(sample_dir, exist_ok=True)

    best_psnr = -1.0
    print("Training started.")

    def _stat(name, t):
        t = t.detach()
        finite = torch.isfinite(t).all().item()
        return f"{name}: dtype={t.dtype} min={float(t.min()):.4f} max={float(t.max()):.4f} mean={float(t.mean()):.4f} finite={finite}"

    
    # -----------------------------------------------------------------
    # Second-round: stochastic degradation for consistency regularization
    # -----------------------------------------------------------------
    def _stochastic_degrade_vis(x: torch.Tensor, sigma: float = 0.02) -> torch.Tensor:
        # x: (B,3,H,W) in [0,1]
        if x.numel() == 0:
            return x
        # random gamma & brightness (per-sample)
        b = 0.9 + 0.2 * torch.rand((x.size(0), 1, 1, 1), device=x.device, dtype=x.dtype)
        g = 0.9 + 0.2 * torch.rand((x.size(0), 1, 1, 1), device=x.device, dtype=x.dtype)
        y = torch.clamp(x * b, 0.0, 1.0)
        y = torch.clamp(y, 1e-6, 1.0) ** g
        # additive noise
        if sigma > 0:
            n = torch.randn_like(y) * sigma
            y = torch.clamp(y + n, 0.0, 1.0)
        return y

    def _stochastic_degrade_ir(x: torch.Tensor, sigma: float = 0.02) -> torch.Tensor:
        # x: (B,1,H,W) in [0,1]
        if x.numel() == 0:
            return x
        c = 0.9 + 0.2 * torch.rand((x.size(0), 1, 1, 1), device=x.device, dtype=x.dtype)
        y = torch.clamp(x * c, 0.0, 1.0)
        if sigma > 0:
            n = torch.randn_like(y) * sigma
            y = torch.clamp(y + n, 0.0, 1.0)
        return y

    def _normalize_feat(f: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        # normalize per-sample, per-channel across spatial dims
        mean = f.mean(dim=(2, 3), keepdim=True)
        std = f.std(dim=(2, 3), keepdim=True)
        return (f - mean) / (std + eps)

    for epoch in range(num_epochs):
        model.train()
        epoch_loss_total = 0.0
        epoch_loss_vis = 0.0
        epoch_loss_ir = 0.0
        epoch_loss_conf = 0.0
        epoch_loss_wvis = 0.0
        epoch_loss_wir = 0.0
        epoch_loss_cons = 0.0

        for batch in train_loader:
            v_low, v_gt, i_noisy, i_gt = batch
            v_low, v_gt = v_low.to(device), v_gt.to(device)
            i_noisy, i_gt = i_noisy.to(device), i_gt.to(device)

            # Quick sanity check (first iteration of first epoch)
            if epoch == 0 and epoch_loss_total == 0.0:
                print(_stat("v_low", v_low), "|", _stat("v_gt", v_gt))
                print(_stat("i_noisy", i_noisy), "|", _stat("i_gt", i_gt))

            optimizer.zero_grad(set_to_none=True)
            # Forward can be AMP if you want, but compute ALL losses in fp32 to avoid NaNs.
            with torch.cuda.amp.autocast(enabled=(torch.cuda.is_available() and use_amp)):
                if lambda_cons > 0:
                    v_hat, i_hat, v_conf, i_conf, v_feat, i_feat = model(v_low, i_noisy, return_feats=True)
                    # second view for consistency
                    v_low_2 = _stochastic_degrade_vis(v_low, sigma=cons_sigma_vis)
                    i_noisy_2 = _stochastic_degrade_ir(i_noisy, sigma=cons_sigma_ir)
                    _, _, _, _, v_feat2, i_feat2 = model(v_low_2, i_noisy_2, return_feats=True)
                else:
                    v_hat, i_hat, v_conf, i_conf = model(v_low, i_noisy)

            if lambda_cons > 0:
                finite_ok = (torch.isfinite(v_hat).all() and torch.isfinite(i_hat).all() and torch.isfinite(v_conf).all() and torch.isfinite(i_conf).all() and
                             torch.isfinite(v_feat).all() and torch.isfinite(i_feat).all() and torch.isfinite(v_feat2).all() and torch.isfinite(i_feat2).all())
            else:
                finite_ok = (torch.isfinite(v_hat).all() and torch.isfinite(i_hat).all() and torch.isfinite(v_conf).all() and torch.isfinite(i_conf).all())

            if not finite_ok:
                print("[ERROR] Model produced NaN/Inf.")
                print(_stat("v_hat", v_hat), "|", _stat("i_hat", i_hat))
                print(_stat("v_conf", v_conf), "|", _stat("i_conf", i_conf))
                raise RuntimeError("NaN/Inf in model outputs")

            # Keep visible branch training identical to LYT-Net: compute vis loss only on V_hat vs V_gt.
            # IMPORTANT: run loss in fp32 (disable autocast) for numerical stability.
            with torch.cuda.amp.autocast(enabled=False):
                v_hat_f = v_hat.float()
                i_hat_f = i_hat.float()
                v_gt_f = v_gt.float()
                i_gt_f = i_gt.float()
                loss_vis = vis_criterion(v_gt_f, v_hat_f)
                # IR branch residual denoising loss (L1 + grad L1)
                loss_ir = ir_criterion(i_hat_f, i_gt_f)
                # Weak confidence supervision (first/second-round): teach confidence heads a soft reliability map
                loss_conf = conf_criterion(v_hat_f, v_gt_f, v_conf.float(), i_hat_f, i_gt_f, i_conf.float())
                # Weak confidence supervision (does not weight reconstruction losses)
                # Second-round: confidence-weighted reconstruction (reliability weighting)
                # Use DETACHED confidence to avoid degenerate solutions (simply lowering confidence everywhere).
                loss_w_vis = confidence_weighted_l1(v_hat_f, v_gt_f, v_conf.float().detach()) + \
                             confidence_weighted_grad_l1(v_hat_f, v_gt_f, v_conf.float().detach())
                loss_w_ir = confidence_weighted_l1(i_hat_f, i_gt_f, i_conf.float().detach()) + \
                            confidence_weighted_grad_l1(i_hat_f, i_gt_f, i_conf.float().detach())

                # Second-round: content consistency (two stochastic degradations)
                if lambda_cons > 0:
                    v1 = _normalize_feat(v_feat.float())
                    v2 = _normalize_feat(v_feat2.float())
                    i1 = _normalize_feat(i_feat.float())
                    i2 = _normalize_feat(i_feat2.float())
                    loss_cons = 0.5 * (F.l1_loss(v1, v2.detach()) + F.l1_loss(v2, v1.detach())) + \
                                0.5 * (F.l1_loss(i1, i2.detach()) + F.l1_loss(i2, i1.detach()))
                else:
                    loss_cons = torch.zeros((), device=device, dtype=torch.float32)

                # Joint multi-task optimization
                loss = (loss_vis + lambda_ir * loss_ir + lambda_conf * loss_conf +
                        lambda_w_vis * loss_w_vis + lambda_w_ir * loss_w_ir + lambda_cons * loss_cons)

            # Fail fast if NaNs/Infs appear, and print useful diagnostics.
            if not torch.isfinite(loss).all():
                print("\n[ERROR] Non-finite loss detected. Dumping tensor stats for debugging:")
                print(_stat("v_low", v_low))
                print(_stat("v_gt", v_gt))
                print(_stat("i_noisy", i_noisy))
                print(_stat("i_gt", i_gt))
                print(_stat("v_hat", v_hat))
                print(_stat("i_hat", i_hat))
                print(f"loss_vis={loss_vis.item()} loss_ir={loss_ir.item()} loss_conf={loss_conf.item()} total={loss.item()}")
                raise RuntimeError("Non-finite loss encountered. Check input ranges/normalization.")

            if use_amp and torch.cuda.is_available():
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            epoch_loss_total += loss.item()
            epoch_loss_vis += float(loss_vis.detach().cpu().item())
            epoch_loss_ir += float(loss_ir.detach().cpu().item())
            epoch_loss_conf += float(loss_conf.detach().cpu().item())
            epoch_loss_wvis += float(loss_w_vis.detach().cpu().item())
            epoch_loss_wir += float(loss_w_ir.detach().cpu().item())
            epoch_loss_cons += float(loss_cons.detach().cpu().item())

        metrics = validate(
            model,
            val_loader,
            device,
            gt_mean_vis=GT_MEAN_VIS,
            save_dir=os.path.join(sample_dir, f"epoch_{epoch+1:04d}") if (epoch + 1) % 50 == 0 else None,
            save_first_n=4,
        )
        denom = max(len(train_loader), 1)
        avg_loss = epoch_loss_total / denom
        avg_loss_vis = epoch_loss_vis / denom
        avg_loss_ir = epoch_loss_ir / denom
        avg_loss_conf = epoch_loss_conf / denom
        avg_loss_wvis = epoch_loss_wvis / denom
        avg_loss_wir = epoch_loss_wir / denom
        avg_loss_cons = epoch_loss_cons / denom
        print(
            f"Epoch {epoch+1}/{num_epochs} | total={avg_loss:.4f} "
            f"(vis={avg_loss_vis:.4f}, ir={avg_loss_ir:.4f}, conf={avg_loss_conf:.4f}, "
            f"wvis={avg_loss_wvis:.4f}, wir={avg_loss_wir:.4f}, cons={avg_loss_cons:.4f}) | "
            f"lambdas(ir={lambda_ir}, conf={lambda_conf}, wvis={lambda_w_vis}, wir={lambda_w_ir}, cons={lambda_cons}) | "
            f"PSNR(V)={metrics['psnr_vis']:.3f} SSIM(V)={metrics['ssim_vis']:.4f} | "
            f"PSNR(IR)={metrics['psnr_ir']:.3f}"
        )

        scheduler.step()

        if metrics["psnr_vis"] > best_psnr:
            best_psnr = metrics["psnr_vis"]
            torch.save(model.state_dict(), ckpt_path)
            print(f"Saved best model to {ckpt_path} (PSNR(V)={best_psnr:.3f})")


if __name__ == "__main__":
    main()
