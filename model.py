import torch
from torchprofile import profile_macs

from model import LYT_DualBranch


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = LYT_DualBranch(
        vis_filters=32,
        ir_filters=32,
        ir_n_resblocks=1,
        ir_init_scale=0.5,
        ir_max_scale=2.0,
    ).to(device)
    model.eval()

    vis_input = torch.randn(1, 3, 256, 256).to(device)
    ir_input = torch.randn(1, 1, 256, 256).to(device)

    # torchprofile may not support multiple inputs for a wrapper module reliably.
    # We profile each branch separately and sum the MACs.
    macs_vis = profile_macs(model.vis, vis_input)
    macs_ir = profile_macs(model.ir, ir_input)
    macs_total = macs_vis + macs_ir
    flops_total = macs_total * 2

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    gmacs = macs_total / (1024 * 1024 * 1024)
    gflops = flops_total / (1024 * 1024 * 1024)

    print(f"Visible MACs (G): {macs_vis / (1024*1024*1024):.4f}")
    print(f"Infrared MACs (G): {macs_ir / (1024*1024*1024):.4f}")
    print(f"Total MACs (G): {gmacs:.4f}")
    print(f"Total FLOPs (G): {gflops:.4f}")
    print(f"Params (M): {num_params / 1e6:.4f}")
    print(f"Params: {num_params}")


if __name__ == "__main__":
    main()
