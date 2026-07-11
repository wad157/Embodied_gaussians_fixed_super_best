import torch
from gsplat.rendering import rasterization


def main():
    device = "cuda"
    means = torch.tensor([[0.0, 0.0, 2.0]], device=device, dtype=torch.float32)
    quats = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device, dtype=torch.float32)
    scales = torch.tensor([[0.1, 0.1, 0.1]], device=device, dtype=torch.float32)
    opacities = torch.tensor([0.9], device=device, dtype=torch.float32)
    colors = torch.tensor([[1.0, 0.0, 0.0]], device=device, dtype=torch.float32)
    viewmats = torch.eye(4, device=device, dtype=torch.float32)[None]
    Ks = torch.tensor(
        [[[100.0, 0.0, 32.0], [0.0, 100.0, 32.0], [0.0, 0.0, 1.0]]],
        device=device,
        dtype=torch.float32,
    )

    render_colors, render_alphas, _ = rasterization(
        means,
        quats,
        scales,
        opacities,
        colors,
        viewmats,
        Ks,
        width=64,
        height=64,
    )
    torch.cuda.synchronize()
    print("gsplat rasterization ok", tuple(render_colors.shape), tuple(render_alphas.shape))


if __name__ == "__main__":
    main()
