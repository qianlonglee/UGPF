import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from pytorch_msssim import ms_ssim
import torchvision.transforms as T

class VGGPerceptualLoss(nn.Module):
    def __init__(self, device):
        super(VGGPerceptualLoss, self).__init__()
        # Torchvision changed the VGG weights API across versions.
        # Also, some environments run without internet (unable to download weights).
        # In that case we safely disable the perceptual loss term.
        self.loss_model = None
        try:
            vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features[:16]  # Until block3_conv3
            self.loss_model = vgg.to(device).eval()
        except Exception:
            try:
                vgg = models.vgg19(pretrained=True).features[:16]
                self.loss_model = vgg.to(device).eval()
            except Exception:
                self.loss_model = None

        if self.loss_model is not None:
            for param in self.loss_model.parameters():
                param.requires_grad = False

    def forward(self, y_true, y_pred):
        if self.loss_model is None:
            # Perceptual loss disabled (e.g., weights unavailable)
            return y_true.new_tensor(0.0)
        y_true, y_pred = y_true.to(next(self.loss_model.parameters()).device), y_pred.to(next(self.loss_model.parameters()).device)
        return F.mse_loss(self.loss_model(y_true), self.loss_model(y_pred))


def color_loss(y_true, y_pred):
    return torch.mean(torch.abs(torch.mean(y_true, dim=[1, 2, 3]) - torch.mean(y_pred, dim=[1, 2, 3])))

def psnr_loss(y_true, y_pred):
    mse = F.mse_loss(y_true, y_pred)
    psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
    return 40.0 - torch.mean(psnr)

def smooth_l1_loss(y_true, y_pred):
    return F.smooth_l1_loss(y_true, y_pred)

def multiscale_ssim_loss(y_true, y_pred, max_val=1.0, power_factors=[0.5, 0.5]):
    return 1.0 - ms_ssim(y_true, y_pred, data_range=max_val, size_average=True)

def gaussian_kernel(x, mu, sigma):
    return torch.exp(-0.5 * ((x - mu) / sigma) ** 2)

def histogram_loss(y_true, y_pred, bins=256, sigma=0.01):
    
    bin_edges = torch.linspace(0.0, 1.0, bins, device=y_true.device)

    y_true_hist = torch.sum(gaussian_kernel(y_true.unsqueeze(-1), bin_edges, sigma), dim=0)
    y_pred_hist = torch.sum(gaussian_kernel(y_pred.unsqueeze(-1), bin_edges, sigma), dim=0)
    
    y_true_hist /= y_true_hist.sum()
    y_pred_hist /= y_pred_hist.sum()

    hist_distance = torch.mean(torch.abs(y_true_hist - y_pred_hist))
    return hist_distance

class CombinedLoss(nn.Module):
    def __init__(self, device):
        super(CombinedLoss, self).__init__()
        self.perceptual_loss_model = VGGPerceptualLoss(device)
        self.alpha1 = 1.00
        self.alpha2 = 0.06
        self.alpha3 = 0.05
        self.alpha4 = 0.5
        self.alpha5 = 0.0083
        self.alpha6 = 0.25

    def forward(self, y_true, y_pred):
        smooth_l1_l = smooth_l1_loss(y_true, y_pred)
        ms_ssim_l = multiscale_ssim_loss(y_true, y_pred)
        perc_l = self.perceptual_loss_model(y_true, y_pred)
        hist_l = histogram_loss(y_true, y_pred)
        psnr_l = psnr_loss(y_true, y_pred)
        color_l = color_loss(y_true, y_pred)

        total_loss = (self.alpha1 * smooth_l1_l + self.alpha2 * perc_l + 
                      self.alpha3 * hist_l + self.alpha5 * psnr_l + 
                      self.alpha6 * color_l + self.alpha4 * ms_ssim_l)

        return torch.mean(total_loss)


def _sobel_gradients(x: torch.Tensor) -> torch.Tensor:
    """Compute Sobel gradients magnitude (approx) for a BCHW tensor."""
    if x.dim() != 4:
        raise ValueError(f"Expected 4D tensor BCHW, got {x.dim()}D")
    # Apply per-channel Sobel; groups = C
    c = x.size(1)
    device = x.device
    dtype = x.dtype
    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=device, dtype=dtype).view(1, 1, 3, 3)
    ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], device=device, dtype=dtype).view(1, 1, 3, 3)
    kx = kx.repeat(c, 1, 1, 1)
    ky = ky.repeat(c, 1, 1, 1)
    gx = F.conv2d(x, kx, padding=1, groups=c)
    gy = F.conv2d(x, ky, padding=1, groups=c)
    g = torch.sqrt(gx * gx + gy * gy + 1e-12)
    return g


class IRLoss(nn.Module):
    """Simple infrared restoration loss: L1 + gradient L1."""

    def __init__(self, grad_weight: float = 1.0):
        super().__init__()
        self.grad_weight = grad_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if pred.shape != target.shape:
            raise ValueError(f"IRLoss shape mismatch: pred {tuple(pred.shape)} vs target {tuple(target.shape)}")
        l1 = F.l1_loss(pred, target)
        g1 = F.l1_loss(_sobel_gradients(pred), _sobel_gradients(target))
        return l1 + self.grad_weight * g1


class DualBranchLoss(nn.Module):
    """Total loss for dual-branch restoration (visible + infrared)."""

    def __init__(self, device, lambda_ir: float = 0.5, ir_grad_weight: float = 1.0):
        super().__init__()
        self.vis_loss = CombinedLoss(device)
        self.ir_loss = IRLoss(grad_weight=ir_grad_weight)
        self.lambda_ir = lambda_ir

    def forward(
        self,
        vis_pred: torch.Tensor,
        vis_target: torch.Tensor,
        ir_pred: torch.Tensor,
        ir_target: torch.Tensor,
    ) -> torch.Tensor:
        # CombinedLoss expects (y_true, y_pred) in this repo's implementation
        l_vis = self.vis_loss(vis_target, vis_pred)
        l_ir = self.ir_loss(ir_pred, ir_target)
        return l_vis + self.lambda_ir * l_ir


def confidence_soft_target(pred: torch.Tensor, target: torch.Tensor, alpha: float = 10.0) -> torch.Tensor:
    """Build a soft confidence target in [0,1] from absolute reconstruction error.

    Target formulation:
        T = exp(-alpha * |pred-target|)

    For RGB, error is averaged across channels to produce a 1-channel map.
    """
    if pred.shape != target.shape:
        raise ValueError(f"confidence_soft_target shape mismatch: pred {tuple(pred.shape)} vs target {tuple(target.shape)}")
    err = torch.abs(pred - target)
    if err.size(1) > 1:
        err = err.mean(dim=1, keepdim=True)
    return torch.exp(-float(alpha) * err).clamp(0.0, 1.0)



# -------------------------------------------------------------------------
# Confidence-weighted reconstruction losses (second-round upgrade)
# -------------------------------------------------------------------------

def _normalize_weight_map(w: torch.Tensor, eps: float = 1e-6, min_w: float = 0.2, max_w: float = 2.0) -> torch.Tensor:
    """Normalize a (B,1,H,W) reliability map to have per-sample mean ~1.

    We clamp to a safe range to avoid vanishing gradients when weights become
    too small (or exploding when too large). Designed for stability.
    """
    if w.dim() != 4 or w.size(1) != 1:
        raise ValueError(f"weight map must be (B,1,H,W), got {tuple(w.shape)}")
    mean = w.mean(dim=(1,2,3), keepdim=True)
    w = w / (mean + eps)
    return w.clamp(min_w, max_w)

def confidence_weighted_l1(pred: torch.Tensor, target: torch.Tensor, conf: torch.Tensor,
                           min_w: float = 0.2, max_w: float = 2.0) -> torch.Tensor:
    """Confidence-weighted L1 loss.

    `conf` is treated as a reliability map; we detach it by default upstream.
    """
    if pred.shape != target.shape:
        raise ValueError(f"weighted_l1 shape mismatch: pred {tuple(pred.shape)} vs target {tuple(target.shape)}")
    w = _normalize_weight_map(conf, min_w=min_w, max_w=max_w)
    if pred.size(1) != 1:
        w = w.expand(pred.size(0), pred.size(1), pred.size(2), pred.size(3))
    return (w * (pred - target).abs()).mean()

def confidence_weighted_grad_l1(pred: torch.Tensor, target: torch.Tensor, conf: torch.Tensor,
                                min_w: float = 0.2, max_w: float = 2.0) -> torch.Tensor:
    """Confidence-weighted gradient L1 (Sobel). Works for 1ch or 3ch preds."""
    if pred.shape != target.shape:
        raise ValueError(f"weighted_grad_l1 shape mismatch: pred {tuple(pred.shape)} vs target {tuple(target.shape)}")
    # Use grayscale for multi-channel to reduce color/texture bias.
    if pred.size(1) == 3:
        pred_g = pred.mean(dim=1, keepdim=True)
        target_g = target.mean(dim=1, keepdim=True)
    else:
        pred_g, target_g = pred, target
    w = _normalize_weight_map(conf, min_w=min_w, max_w=max_w)
    gp = _sobel_gradients(pred_g)
    gt = _sobel_gradients(target_g)
    return (w * (gp - gt).abs()).mean()

class ConfidenceLoss(nn.Module):
    """Weak supervision for pixel-wise confidence maps.

    This DOES NOT weight the reconstruction losses (first-round upgrade);
    it only teaches the confidence heads to predict a soft reliability map.
    """

    def __init__(self, alpha: float = 10.0):
        super().__init__()
        self.alpha = float(alpha)

    def forward(
        self,
        vis_pred: torch.Tensor,
        vis_target: torch.Tensor,
        vis_conf: torch.Tensor,
        ir_pred: torch.Tensor,
        ir_target: torch.Tensor,
        ir_conf: torch.Tensor,
    ) -> torch.Tensor:
        if vis_conf.dim() != 4 or vis_conf.size(1) != 1:
            raise ValueError(f"vis_conf must be (B,1,H,W), got {tuple(vis_conf.shape)}")
        if ir_conf.dim() != 4 or ir_conf.size(1) != 1:
            raise ValueError(f"ir_conf must be (B,1,H,W), got {tuple(ir_conf.shape)}")

        t_vis = confidence_soft_target(vis_pred, vis_target, alpha=self.alpha)
        t_ir = confidence_soft_target(ir_pred, ir_target, alpha=self.alpha)
        return F.l1_loss(vis_conf, t_vis) + F.l1_loss(ir_conf, t_ir)
