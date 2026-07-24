import torch
import torch.nn.functional as F


def rgb_to_y(rgb: torch.Tensor) -> torch.Tensor:
    """Convert RGB [0,1] to luminance Y [0,1], returns (B,1,H,W)."""
    if rgb.dim() != 4 or rgb.size(1) != 3:
        raise ValueError(f"Expected RGB BCHW, got {tuple(rgb.shape)}")
    r = rgb[:, 0:1]
    g = rgb[:, 1:2]
    b = rgb[:, 2:3]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    return y.clamp(0.0, 1.0)


def sobel_magnitude(x: torch.Tensor) -> torch.Tensor:
    """Sobel gradient magnitude for (B,1,H,W) or (B,C,H,W)."""
    if x.dim() != 4:
        raise ValueError(f"Expected BCHW, got {tuple(x.shape)}")

    c = x.size(1)
    device = x.device
    dtype = x.dtype

    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=device, dtype=dtype).view(1, 1, 3, 3)
    ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], device=device, dtype=dtype).view(1, 1, 3, 3)
    kx = kx.repeat(c, 1, 1, 1)
    ky = ky.repeat(c, 1, 1, 1)

    gx = F.conv2d(x, kx, padding=1, groups=c)
    gy = F.conv2d(x, ky, padding=1, groups=c)
    return torch.sqrt(gx * gx + gy * gy + 1e-12)


def Fusion_Loss_Base(
    vi: torch.Tensor,
    ir: torch.Tensor,
    fu: torch.Tensor,
    w_int: float = 1.0,
    w_grad: float = 1.0,
):
    """A clean, standard Stage-2 fusion loss.

    Args:
      vi: enhanced visible RGB, (B,3,H,W) in [0,1]
      ir: denoised IR, (B,1,H,W) in [0,1]
      fu: fused RGB, (B,3,H,W) in [0,1]
      w_int: intensity weight
      w_grad: gradient weight

    Returns:
      loss_total, loss_int, loss_grad

    Design (classic):
      - Intensity: fused luminance should follow the stronger signal among (Y_vis, IR)
      - Gradient: fused gradients should follow the stronger edges among (grad(Y_vis), grad(IR))

    This is widely used in unsupervised IVIF fusion training.
    """
    if vi.size(1) != 3 or fu.size(1) != 3 or ir.size(1) != 1:
        raise ValueError(
            f"Shape mismatch: vi{tuple(vi.shape)} ir{tuple(ir.shape)} fu{tuple(fu.shape)}"
        )

    vi_y = rgb_to_y(vi)
    fu_y = rgb_to_y(fu)

    # intensity target: per-pixel max
    target_int = torch.max(vi_y, ir)
    loss_int = F.l1_loss(fu_y, target_int)

    # gradient target: per-pixel max of gradient magnitude
    grad_vi = sobel_magnitude(vi_y)
    grad_ir = sobel_magnitude(ir)
    grad_target = torch.max(grad_vi, grad_ir)

    grad_fu = sobel_magnitude(fu_y)
    loss_grad = F.l1_loss(grad_fu, grad_target)

    loss_total = float(w_int) * loss_int + float(w_grad) * loss_grad
    return loss_total, loss_int, loss_grad


def MaxFeatureSemanticLoss(
    vis_feat: torch.Tensor,
    ir_feat: torch.Tensor,
    fused_feat: torch.Tensor,
    eps: float = 1e-6,
):
    """Max-Feature Semantic Loss (UGPF).

    Goal: fused features keep the *strongest* semantic activation among
    VIS/IR, avoiding gray-collapse when modalities conflict.

    We measure per-pixel feature energy (L2 norm over channels):
        s(x) = ||x||_2
    and enforce:
        s(fused) ≈ max( s(vis), s(ir) )

    Args:
        vis_feat, ir_feat, fused_feat: (B,C,H,W)

    Returns:
        scalar loss
    """
    if vis_feat.dim() != 4 or ir_feat.dim() != 4 or fused_feat.dim() != 4:
        raise ValueError("Expected features in BCHW.")

    sv = torch.sqrt(torch.sum(vis_feat * vis_feat, dim=1, keepdim=True) + eps)
    si = torch.sqrt(torch.sum(ir_feat * ir_feat, dim=1, keepdim=True) + eps)
    sf = torch.sqrt(torch.sum(fused_feat * fused_feat, dim=1, keepdim=True) + eps)

    target = torch.max(sv, si)
    return F.l1_loss(sf, target)
