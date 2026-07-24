import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

class LayerNormalization(nn.Module):
    def __init__(self, dim):
        super(LayerNormalization, self).__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, return_feat: bool = False):
        # Rearrange the tensor for LayerNorm (B, C, H, W) to (B, H, W, C)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        # Rearrange back to (B, C, H, W)
        return x.permute(0, 3, 1, 2)

class SEBlock(nn.Module):
    def __init__(self, input_channels, reduction_ratio=16):
        super(SEBlock, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(input_channels, input_channels // reduction_ratio)
        self.fc2 = nn.Linear(input_channels // reduction_ratio, input_channels)
        self._init_weights()

    def forward(self, x):
        batch_size, num_channels, _, _ = x.size()
        y = self.pool(x).reshape(batch_size, num_channels)
        y = F.relu(self.fc1(y))
        y = torch.tanh(self.fc2(y))
        y = y.reshape(batch_size, num_channels, 1, 1)
        return x * y
    
    def _init_weights(self):
        init.kaiming_uniform_(self.fc1.weight, a=0, mode='fan_in', nonlinearity='relu')
        init.kaiming_uniform_(self.fc2.weight, a=0, mode='fan_in', nonlinearity='relu')
        init.constant_(self.fc1.bias, 0)
        init.constant_(self.fc2.bias, 0)

class MSEFBlock(nn.Module):
    def __init__(self, filters):
        super(MSEFBlock, self).__init__()
        self.layer_norm = LayerNormalization(filters)
        self.depthwise_conv = nn.Conv2d(filters, filters, kernel_size=3, padding=1, groups=filters)
        self.se_attn = SEBlock(filters)
        self._init_weights()

    def forward(self, x):
        x_norm = self.layer_norm(x)
        x1 = self.depthwise_conv(x_norm)
        x2 = self.se_attn(x_norm)
        x_fused = x1 * x2
        x_out = x_fused + x
        return x_out
    
    def _init_weights(self):
        init.kaiming_uniform_(self.depthwise_conv.weight, a=0, mode='fan_in', nonlinearity='relu')
        init.constant_(self.depthwise_conv.bias, 0)


class ResBlock(nn.Module):
    """Lightweight residual block: Conv-ReLU-Conv with identity skip.

    This is intentionally tiny (no norm) to keep training stable and preserve
    the original LYT-Net behavior while improving denoising capacity.
    """

    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=pad)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=pad)
        self._init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.conv1(x), inplace=True)
        out = self.conv2(out)
        return x + out

    def _init_weights(self):
        for layer in [self.conv1, self.conv2]:
            init.kaiming_uniform_(layer.weight, a=0, mode='fan_in', nonlinearity='relu')
            if layer.bias is not None:
                init.constant_(layer.bias, 0)

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_size, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_size = embed_size
        self.num_heads = num_heads
        assert embed_size % num_heads == 0
        self.head_dim = embed_size // num_heads
        self.query_dense = nn.Linear(embed_size, embed_size)
        self.key_dense = nn.Linear(embed_size, embed_size)
        self.value_dense = nn.Linear(embed_size, embed_size)
        self.combine_heads = nn.Linear(embed_size, embed_size)
        self._init_weights()

    def split_heads(self, x, batch_size):
        x = x.reshape(batch_size, -1, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)

    def forward(self, x):
        batch_size, _, height, width = x.size()
        x = x.reshape(batch_size, height * width, -1)

        query = self.split_heads(self.query_dense(x), batch_size)
        key = self.split_heads(self.key_dense(x), batch_size)
        value = self.split_heads(self.value_dense(x), batch_size)
        
        attention_weights = F.softmax(torch.matmul(query, key.transpose(-2, -1)) / (self.head_dim ** 0.5), dim=-1)
        attention = torch.matmul(attention_weights, value)
        attention = attention.permute(0, 2, 1, 3).contiguous().reshape(batch_size, -1, self.embed_size)
        
        output = self.combine_heads(attention)
        
        return output.reshape(batch_size, height, width, self.embed_size).permute(0, 3, 1, 2)

    def _init_weights(self):
        init.xavier_uniform_(self.query_dense.weight)
        init.xavier_uniform_(self.key_dense.weight)
        init.xavier_uniform_(self.value_dense.weight)
        init.xavier_uniform_(self.combine_heads.weight)
        init.constant_(self.query_dense.bias, 0)
        init.constant_(self.key_dense.bias, 0)
        init.constant_(self.value_dense.bias, 0)
        init.constant_(self.combine_heads.bias, 0)

class Denoiser(nn.Module):
    def __init__(self, num_filters, kernel_size=3, activation='relu'):
        super(Denoiser, self).__init__()
        self.conv1 = nn.Conv2d(1, num_filters, kernel_size=kernel_size, padding=1)
        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size=kernel_size, stride=2, padding=1)
        self.conv3 = nn.Conv2d(num_filters, num_filters, kernel_size=kernel_size, stride=2, padding=1)
        self.conv4 = nn.Conv2d(num_filters, num_filters, kernel_size=kernel_size, stride=2, padding=1)
        self.bottleneck = MultiHeadSelfAttention(embed_size=num_filters, num_heads=4)
        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up3 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up4 = nn.Upsample(scale_factor=2, mode='nearest')
        self.output_layer = nn.Conv2d(1, 1, kernel_size=kernel_size, padding=1)
        self.res_layer = nn.Conv2d(num_filters, 1, kernel_size=kernel_size, padding=1)
        self.activation = getattr(F, activation)
        self._init_weights()

    def forward(self, x, return_feat: bool = False):
        x1 = self.activation(self.conv1(x))
        x2 = self.activation(self.conv2(x1))
        x3 = self.activation(self.conv3(x2))
        x4 = self.activation(self.conv4(x3))
        x = self.bottleneck(x4)
        feat = x  # (B, C, H/8, W/8)
        x = self.up4(x)
        x = self.up3(x + x3)
        x = self.up2(x + x2)
        x = x + x1
        x = self.res_layer(x)
        out = torch.tanh(self.output_layer(x + x))
        if return_feat:
            return out, feat
        return out
    
    def _init_weights(self):
        for layer in [self.conv1, self.conv2, self.conv3, self.conv4, self.output_layer, self.res_layer]:
            init.kaiming_uniform_(layer.weight, a=0, mode='fan_in', nonlinearity='relu')
            if layer.bias is not None:
                init.constant_(layer.bias, 0)


def _inv_softplus(y: float) -> float:
    """Inverse of softplus for initialization: softplus(x)=log(1+exp(x))."""
    # y must be > 0
    import math
    return math.log(math.expm1(y))


class DenoiserPlus(nn.Module):
    """Stronger denoiser used for the IR branch (v4.3+).

    Upgrades over the original Denoiser:
      (A2) Residual amplitude control: delta = s * tanh(out), where s is a
           learnable positive scalar (initialized conservatively).
      (B) Capacity boost: per-scale ResBlock to improve denoising strength
          (especially for speckle/strong random noise) with small overhead.

    Notes:
      - API-compatible with Denoiser (supports return_feat).
      - Decoder is kept minimal (same as original) to avoid over-smoothing.
    """

    def __init__(
        self,
        num_filters: int,
        kernel_size: int = 3,
        init_scale: float = 0.5,
        max_scale: float = 2.0,
        n_resblocks: int = 1,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(1, num_filters, kernel_size=kernel_size, padding=1)
        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size=kernel_size, stride=2, padding=1)
        self.conv3 = nn.Conv2d(num_filters, num_filters, kernel_size=kernel_size, stride=2, padding=1)
        self.conv4 = nn.Conv2d(num_filters, num_filters, kernel_size=kernel_size, stride=2, padding=1)

        # Per-scale residual blocks (lightweight)
        self.res1 = nn.Sequential(*[ResBlock(num_filters, kernel_size=kernel_size) for _ in range(n_resblocks)])
        self.res2 = nn.Sequential(*[ResBlock(num_filters, kernel_size=kernel_size) for _ in range(n_resblocks)])
        self.res3 = nn.Sequential(*[ResBlock(num_filters, kernel_size=kernel_size) for _ in range(n_resblocks)])
        self.res4 = nn.Sequential(*[ResBlock(num_filters, kernel_size=kernel_size) for _ in range(n_resblocks)])

        self.bottleneck = MultiHeadSelfAttention(embed_size=num_filters, num_heads=4)
        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up3 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up4 = nn.Upsample(scale_factor=2, mode='nearest')

        self.output_layer = nn.Conv2d(1, 1, kernel_size=kernel_size, padding=1)
        self.res_layer = nn.Conv2d(num_filters, 1, kernel_size=kernel_size, padding=1)

        # Learnable positive residual scale s (A2)
        self.max_scale = float(max_scale)
        s0 = float(init_scale)
        self._s_raw = nn.Parameter(torch.tensor(_inv_softplus(s0), dtype=torch.float32))

        self._init_weights()

    def _scale(self) -> torch.Tensor:
        s = F.softplus(self._s_raw)
        if self.max_scale is not None:
            s = torch.clamp(s, 0.0, self.max_scale)
        return s

    def forward(self, x: torch.Tensor, return_feat: bool = False):
        x1 = F.relu(self.conv1(x), inplace=True)
        x1 = self.res1(x1)
        x2 = F.relu(self.conv2(x1), inplace=True)
        x2 = self.res2(x2)
        x3 = F.relu(self.conv3(x2), inplace=True)
        x3 = self.res3(x3)
        x4 = F.relu(self.conv4(x3), inplace=True)
        x4 = self.res4(x4)

        x = self.bottleneck(x4)
        feat = x  # (B, C, H/8, W/8)

        x = self.up4(x)
        x = self.up3(x + x3)
        x = self.up2(x + x2)
        x = x + x1

        x = self.res_layer(x)
        out = torch.tanh(self.output_layer(x + x))
        delta = self._scale() * out
        if return_feat:
            return delta, feat
        return delta

    def _init_weights(self):
        for layer in [self.conv1, self.conv2, self.conv3, self.conv4, self.output_layer, self.res_layer]:
            init.kaiming_uniform_(layer.weight, a=0, mode='fan_in', nonlinearity='relu')
            if layer.bias is not None:
                init.constant_(layer.bias, 0)

class LYT(nn.Module):
    """LYT-Net visible restoration backbone (prototype-compatible).

    v4.2 (first-round upgrade):
      1) IR-guided illumination gating: a lightweight IR encoder generates a
         soft gate to modulate the Y-branch (illumination) features. This is a
         *gentle* guidance (scaled by `ir_gate_eps`) to avoid introducing
         artifacts when IR is noisy/misaligned.
      2) Optional pixel-wise confidence head. For safety, confidence is
         predicted from *detached* internal features by default, so the
         confidence supervision does not perturb the original LYT restoration
         path (helps preserve/avoid degrading baseline performance).
    """

    def __init__(self, filters=32, enable_ir_guidance: bool = True, ir_gate_eps: float = 0.2):
        super(LYT, self).__init__()
        self.process_y = self._create_processing_layers(filters)
        self.process_cb = self._create_processing_layers(filters)
        self.process_cr = self._create_processing_layers(filters)

        self.denoiser_cb = Denoiser(filters // 2)
        self.denoiser_cr = Denoiser(filters // 2)
        self.lum_pool = nn.MaxPool2d(8)
        self.lum_mhsa = MultiHeadSelfAttention(embed_size=filters, num_heads=4)
        self.lum_up = nn.Upsample(scale_factor=8, mode='nearest')
        self.lum_conv = nn.Conv2d(filters, filters, kernel_size=1, padding=0)
        self.ref_conv = nn.Conv2d(filters * 2, filters, kernel_size=1, padding=0)
        self.msef = MSEFBlock(filters)
        self.recombine = nn.Conv2d(filters * 2, filters, kernel_size=3, padding=1)
        self.final_adjustments = nn.Conv2d(filters, 3, kernel_size=3, padding=1)

        # ------------------------------
        # IR-guided illumination gating
        # ------------------------------
        self.enable_ir_guidance = enable_ir_guidance
        self.ir_gate_eps = float(ir_gate_eps)
        if self.enable_ir_guidance:
            # Keep it tiny: one conv + activation, then a 1x1 gate.
            self.ir_proj = nn.Sequential(
                nn.Conv2d(1, filters, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
            )
            self.ir_gate = nn.Conv2d(filters * 2, filters, kernel_size=1, padding=0)

        # ------------------------------
        # Confidence head (1-channel)
        # ------------------------------
        self.vis_conf_head = nn.Conv2d(filters, 1, kernel_size=3, padding=1)
        self._init_weights()

    def _create_processing_layers(self, filters):
        return nn.Sequential(
            nn.Conv2d(1, filters, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
    
    def _rgb_to_ycbcr(self, image):
        r, g, b = image[:, 0, :, :], image[:, 1, :, :], image[:, 2, :, :]
    
        y = 0.299 * r + 0.587 * g + 0.114 * b
        u = -0.14713 * r - 0.28886 * g + 0.436 * b + 0.5
        v = 0.615 * r - 0.51499 * g - 0.10001 * b + 0.5
        
        yuv = torch.stack((y, u, v), dim=1)
        return yuv

    def forward(self, inputs, ir_hint: torch.Tensor = None, return_conf: bool = False, return_feats: bool = False):
        ycbcr = self._rgb_to_ycbcr(inputs)
        y, cb, cr = torch.split(ycbcr, 1, dim=1)
        cb = self.denoiser_cb(cb) + cb
        cr = self.denoiser_cr(cr) + cr

        y_processed = self.process_y(y)
        cb_processed = self.process_cb(cb)
        cr_processed = self.process_cr(cr)

        ref = torch.cat([cb_processed, cr_processed], dim=1)
        lum = y_processed

        # IR-guided gating (illumination branch only)
        if self.enable_ir_guidance and (ir_hint is not None):
            if ir_hint.dim() != 4 or ir_hint.size(1) != 1:
                raise ValueError(f"LYT expects ir_hint as (B,1,H,W), got {tuple(ir_hint.shape)}")
            if ir_hint.shape[-2:] != lum.shape[-2:]:
                raise ValueError(
                    f"LYT ir_hint spatial mismatch: ir_hint{tuple(ir_hint.shape[-2:])} vs lum{tuple(lum.shape[-2:])}"
                )
            ir_feat = self.ir_proj(ir_hint)
            gate = torch.sigmoid(self.ir_gate(torch.cat([lum, ir_feat], dim=1)))
            lum = lum * (1.0 + self.ir_gate_eps * gate)

        lum_1 = self.lum_pool(lum)
        lum_1 = self.lum_mhsa(lum_1)
        lum_ctx = lum_1  # low-res illumination content feature (B,C,H/8,W/8)
        lum_1 = self.lum_up(lum_1)
        lum = lum + lum_1

        ref = self.ref_conv(ref)
        shortcut = ref
        ref = ref + 0.2 * self.lum_conv(lum)
        ref = self.msef(ref)
        ref = ref + shortcut

        recombined = self.recombine(torch.cat([ref, lum], dim=1))
        output = self.final_adjustments(recombined)

        out = torch.sigmoid(output)

        if return_conf:
            # Predict confidence from *detached* features to avoid disturbing
            # the original LYT restoration behavior.
            conf = torch.sigmoid(self.vis_conf_head(recombined.detach()))
            if return_feats:
                return out, conf, lum_ctx
            return out, conf
        if return_feats:
            return out, lum_ctx
        return out
    
    def _init_weights(self):
        for module in self.children():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                init.kaiming_uniform_(module.weight, a=0, mode='fan_in', nonlinearity='relu')
                if module.bias is not None:
                    init.constant_(module.bias, 0)


class IRRestorer(nn.Module):
    """A lightweight infrared restoration/denoising branch.

    Uses the existing Denoiser block to predict a residual (delta), then applies
    residual learning with clamping to [0, 1].
    """
    def __init__(self, filters: int = 32, n_resblocks: int = 1, init_scale: float = 0.5, max_scale: float = 2.0):
        """Infrared denoising/restoration branch.

        v4.3+ upgrade (A2+B):
          - Use DenoiserPlus with per-scale ResBlocks to boost capacity.
          - Use a learnable residual scale s * tanh(.) for stable yet stronger
            residual prediction.
        """
        super().__init__()
        self.denoiser = DenoiserPlus(
            num_filters=filters,
            n_resblocks=n_resblocks,
            init_scale=init_scale,
            max_scale=max_scale,
        )
        # Confidence head (1-channel). We feed it a detached residual estimate
        # to keep the denoising behavior stable.
        self.ir_conf_head = nn.Conv2d(1, 1, kernel_size=3, padding=1)

    def forward(self, ir_noisy: torch.Tensor, return_conf: bool = False, return_feats: bool = False):
        if ir_noisy.dim() != 4 or ir_noisy.size(1) != 1:
            raise ValueError(f"IRRestorer expects shape (B,1,H,W), got {tuple(ir_noisy.shape)}")
        if return_feats:
            delta, ir_ctx = self.denoiser(ir_noisy, return_feat=True)
        else:
            delta = self.denoiser(ir_noisy)
            ir_ctx = None
        ir_hat = torch.clamp(ir_noisy + delta, 0.0, 1.0)
        if return_conf:
            conf = torch.sigmoid(self.ir_conf_head(delta.detach()))
            if return_feats:
                return ir_hat, conf, ir_ctx
            return ir_hat, conf
        if return_feats:
            return ir_hat, ir_ctx
        return ir_hat


class LYT_DualBranch(nn.Module):
    """Dual-branch restoration template.

    - Visible branch: original LYT-Net (kept unchanged)
    - Infrared branch: lightweight IRRestorer
    """
    def __init__(
        self,
        vis_filters: int = 32,
        ir_filters: int = 32,
        ir_gate_eps: float = 0.2,
        ir_n_resblocks: int = 1,
        ir_init_scale: float = 0.5,
        ir_max_scale: float = 2.0,
    ):
        super().__init__()
        self.vis = LYT(filters=vis_filters, enable_ir_guidance=True, ir_gate_eps=ir_gate_eps)
        self.ir = IRRestorer(
            filters=ir_filters,
            n_resblocks=ir_n_resblocks,
            init_scale=ir_init_scale,
            max_scale=ir_max_scale,
        )

    def forward(self, vis_low: torch.Tensor, ir_noisy: torch.Tensor, return_feats: bool = False):
        # Visible enhancement is gently guided by IR via Y-branch gating.
        if return_feats:
            vis_hat, vis_conf, vis_ctx = self.vis(vis_low, ir_hint=ir_noisy, return_conf=True, return_feats=True)
            ir_hat, ir_conf, ir_ctx = self.ir(ir_noisy, return_conf=True, return_feats=True)
            return vis_hat, ir_hat, vis_conf, ir_conf, vis_ctx, ir_ctx
        vis_hat, vis_conf = self.vis(vis_low, ir_hint=ir_noisy, return_conf=True)
        ir_hat, ir_conf = self.ir(ir_noisy, return_conf=True)
        return vis_hat, ir_hat, vis_conf, ir_conf
                    