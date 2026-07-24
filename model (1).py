from typing import Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


###############################################
# YCbCr <-> RGB (keep stable & deterministic)
###############################################

def RGB2YCrCb(rgb_image: torch.Tensor):
    """Convert RGB in [0,1] to Y, Cb, Cr in [0,1].

    We detach chroma channels by default (common practice in IVIF fusion)
    to reduce color drifting when optimizing luminance-focused objectives.

    Args:
        rgb_image: (B,3,H,W) in [0,1]

    Returns:
        Y:  (B,1,H,W) in [0,1]
        Cb: (B,1,H,W) in [0,1] (detached)
        Cr: (B,1,H,W) in [0,1] (detached)
    """
    R = rgb_image[:, 0:1]
    G = rgb_image[:, 1:2]
    B = rgb_image[:, 2:3]

    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cr = (R - Y) * 0.713 + 0.5
    Cb = (B - Y) * 0.564 + 0.5

    Y = Y.clamp(0.0, 1.0)
    Cr = Cr.clamp(0.0, 1.0).detach()
    Cb = Cb.clamp(0.0, 1.0).detach()
    return Y, Cb, Cr


def YCbCr2RGB(Y: torch.Tensor, Cb: torch.Tensor, Cr: torch.Tensor):
    """Convert Y, Cb, Cr in [0,1] back to RGB in [0,1]."""
    ycrcb = torch.cat([Y, Cr, Cb], dim=1)  # [B,3,H,W]
    B, _, H, W = ycrcb.shape

    im_flat = ycrcb.permute(0, 2, 3, 1).reshape(-1, 3)

    mat = torch.tensor(
        [[1.0, 1.0, 1.0],
         [1.403, -0.714, 0.0],
         [0.0, -0.344, 1.773]],
        device=Y.device,
        dtype=Y.dtype,
    )
    bias = torch.tensor([0.0, -0.5, -0.5], device=Y.device, dtype=Y.dtype)

    temp = (im_flat + bias).mm(mat)
    out = temp.reshape(B, H, W, 3).permute(0, 3, 1, 2)
    return out.clamp(0.0, 1.0)


###############################################
# Lightweight conv-only building blocks
###############################################


def _gn(ch: int) -> nn.GroupNorm:
    # small batch friendly
    groups = 8
    while groups > 1 and (ch % groups) != 0:
        groups //= 2
    return nn.GroupNorm(groups, ch)


class DSConv(nn.Module):
    """Depthwise-Separable Conv block (conv-only, very few params)."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        stride: int = 1,
        dilation: int = 1,
        act: bool = True,
    ):
        super().__init__()
        pad = dilation
        self.dw = nn.Conv2d(
            in_ch,
            in_ch,
            kernel_size=3,
            stride=stride,
            padding=pad,
            dilation=dilation,
            groups=in_ch,
            bias=False,
        )
        self.pw = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.norm = _gn(out_ch)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dw(x)
        x = self.pw(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class LiteEncoder(nn.Module):
    """3-stage tiny encoder."""

    def __init__(self, in_ch: int = 1, base: int = 24):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, base, 3, padding=1, bias=False),
            _gn(base),
            nn.SiLU(inplace=True),
        )
        self.s1 = nn.Sequential(
            DSConv(base, base, stride=1),
            DSConv(base, base, stride=1),
        )
        self.s2 = nn.Sequential(
            DSConv(base, base * 2, stride=2),
            DSConv(base * 2, base * 2, stride=1),
        )
        self.s3 = nn.Sequential(
            DSConv(base * 2, base * 4, stride=2),
            DSConv(base * 4, base * 4, stride=1, dilation=2),
        )

    def forward(self, x: torch.Tensor):
        x = self.stem(x)
        f1 = self.s1(x)   # H
        f2 = self.s2(f1)  # H/2
        f3 = self.s3(f2)  # H/4
        return f1, f2, f3


class ReliabilityGate(nn.Module):
    """Conv-only reliability gate.

    It converts (vis_conf, ir_conf) into soft weights via a tiny 3x3 conv + softmax.
    This keeps the whole Stage-2 as pure convolutional computation.

    Input conf maps should be in [0,1].
    """

    def __init__(self, smooth: bool = True):
        super().__init__()
        self.smooth = smooth
        if smooth:
            self.refine = nn.Conv2d(2, 2, kernel_size=3, padding=1, bias=True)
        else:
            self.refine = nn.Identity()

    def forward(
        self,
        vis_feat: torch.Tensor,
        ir_feat: torch.Tensor,
        vis_conf: Optional[torch.Tensor],
        ir_conf: Optional[torch.Tensor],
    ):
        """Return gated features and weights."""
        B, _, H, W = vis_feat.shape

        if vis_conf is None or ir_conf is None:
            wv = torch.full((B, 1, H, W), 0.5, device=vis_feat.device, dtype=vis_feat.dtype)
            wi = 1.0 - wv
        else:
            vc = F.interpolate(vis_conf, size=(H, W), mode="bilinear", align_corners=False)
            ic = F.interpolate(ir_conf, size=(H, W), mode="bilinear", align_corners=False)
            w = torch.cat([vc, ic], dim=1)  # (B,2,H,W)
            w = self.refine(w)
            w = torch.softmax(w, dim=1)
            wv = w[:, 0:1]
            wi = w[:, 1:2]

        return vis_feat * wv, ir_feat * wi, wv, wi


class LiteDecoder(nn.Module):
    """Very small decoder with additive skips (cheaper than concat)."""

    def __init__(self, base: int = 24):
        super().__init__()
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(base * 4, base * 2, 1, bias=False),
            _gn(base * 2),
            nn.SiLU(inplace=True),
            DSConv(base * 2, base * 2),
        )
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(base * 2, base, 1, bias=False),
            _gn(base),
            nn.SiLU(inplace=True),
            DSConv(base, base),
        )
        self.head = nn.Conv2d(base, 1, 3, padding=1)

    def forward(self, f1: torch.Tensor, f2: torch.Tensor, f3: torch.Tensor):
        x = self.up2(f3)  # -> base*2, H/2
        x = x + f2
        x = self.up1(x)   # -> base, H
        x = x + f1
        out = self.head(x)
        return out


###############################################
# Main: Lightweight UGPF FusionNet (conv-only)
###############################################


class FusionNet(nn.Module):
    """UGPF Stage-2 (Lite): conv-only reliability-gated luminance fusion.

    Why it's fast/small:
      - 3-scale encoder/decoder (H, H/2, H/4)
      - Depthwise-separable conv blocks
      - Additive skip connections (no big concat)
      - Optional confidence-gated feature flow (ReliabilityGate)

    Input:
      rgb: (B,3,H,W) enhanced VIS in [0,1]
      ir:  (B,1,H,W) denoised IR in [0,1]
      vis_conf: (B,1,H,W) in [0,1]
      ir_conf:  (B,1,H,W) in [0,1]

    Output:
      fused_y:   (B,1,H,W) in [0,1]
      fused_rgb: (B,3,H,W) in [0,1]

    If return_feats=True:
      returns a dict with mid-level features for semantic loss.
    """

    def __init__(self, base_channels: int = 24, res_scale: float = 0.6, gate_smooth: bool = True):
        super().__init__()
        self.base = int(base_channels)
        self.res_scale = float(res_scale)

        self.vis_enc = LiteEncoder(in_ch=1, base=self.base)
        self.ir_enc = LiteEncoder(in_ch=1, base=self.base)

        self.gate1 = ReliabilityGate(smooth=gate_smooth)
        self.gate2 = ReliabilityGate(smooth=gate_smooth)
        self.gate3 = ReliabilityGate(smooth=gate_smooth)

        self.fuse1 = DSConv(self.base * 2, self.base, stride=1)
        self.fuse2 = DSConv(self.base * 4, self.base * 2, stride=1)
        self.fuse3 = DSConv(self.base * 8, self.base * 4, stride=1)

        self.dec = LiteDecoder(base=self.base)

    def forward(
        self,
        rgb: torch.Tensor,
        ir: torch.Tensor,
        vis_conf: Optional[torch.Tensor] = None,
        ir_conf: Optional[torch.Tensor] = None,
        return_feats: bool = False,
    ):
        if ir.size(1) != 1:
            ir = ir[:, :1]

        # fuse only luminance; keep chroma from VIS to preserve natural colors
        vis_y, cb, cr = RGB2YCrCb(rgb)

        r1, r2, r3 = self.vis_enc(vis_y)
        t1, t2, t3 = self.ir_enc(ir)

        # reliability gating at each scale
        r1g, t1g, _, _ = self.gate1(r1, t1, vis_conf, ir_conf)
        r2g, t2g, _, _ = self.gate2(r2, t2, vis_conf, ir_conf)
        r3g, t3g, _, _ = self.gate3(r3, t3, vis_conf, ir_conf)

        # fuse features (concat then cheap DSConv)
        f1 = self.fuse1(torch.cat([r1g, t1g], dim=1))
        f2 = self.fuse2(torch.cat([r2g, t2g], dim=1))
        f3 = self.fuse3(torch.cat([r3g, t3g], dim=1))

        # decode -> luminance residual
        delta = self.dec(f1, f2, f3)
        delta = torch.tanh(delta) * self.res_scale
        fused_y = (vis_y + delta).clamp(0.0, 1.0)

        fused_rgb = YCbCr2RGB(fused_y, cb, cr)

        if return_feats:
            feats: Dict[str, torch.Tensor] = {
                "r2": r2g,
                "t2": t2g,
                "f2": f2,
            }
            return fused_y, fused_rgb, feats

        return fused_y, fused_rgb
