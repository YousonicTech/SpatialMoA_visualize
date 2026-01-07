import math
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt


def reduce_last_dim(x: torch.Tensor, reduce: str = "mean") -> torch.Tensor:
    """
    x: [M, 8] or [N, 8]
    return: [M] or [N]
    """
    if x.dim() != 2 or x.size(1) != 8:
        raise ValueError(f"Expect shape [*, 8], got {tuple(x.shape)}")
    if reduce == "mean":
        return x.mean(dim=1)
    elif reduce == "sum":
        return x.sum(dim=1)
    else:
        raise ValueError("reduce must be 'mean' or 'sum'")


def safe_mean_std(x: torch.Tensor, eps: float = 1e-12):
    mu = x.mean()
    sigma = x.std(unbiased=False)
    return mu, sigma, sigma / (mu.abs() + eps)  # CV uses abs(mean) for safety


def normalized_entropy(x: torch.Tensor, eps: float = 1e-12) -> float:
    """
    Convert x (length L) into a probability distribution and compute normalized entropy in [0,1].
    Higher => more uniform.
    Works even if x has negative values (shift to be non-negative).
    """
    x = x.float()
    L = x.numel()
    if L <= 1:
        return 1.0

    # shift to non-negative
    x = x - x.min()
    x = x + eps
    p = x / x.sum()

    H = -(p * (p + eps).log()).sum()
    H_norm = (H / math.log(L)).clamp(0.0, 1.0)
    return float(H_norm.item())


def main(
    data_dir: str,
    reduce: str = "mean",
    start: int = 0,
    end: int = 999,
    key_a: str = "A",
    key_b: str = "B",
    out_dir: str = "./attn_plots",
):
    data_dir = Path(data_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pooled_a_norm = []
    pooled_b_norm = []

    a_mean_list, a_std_list, a_cv_list, a_ent_list = [], [], [], []
    b_mean_list, b_std_list, b_cv_list, b_ent_list = [], [], [], []

    missing = 0
    loaded = 0

    for i in range(start, end + 1):
        p = data_dir / f"{i}.pt"
        if not p.exists():
            missing += 1
            continue

        obj = torch.load(p, map_location="cpu")
        if key_a not in obj or key_b not in obj:
            raise KeyError(f"{p.name} missing keys. Found keys: {list(obj.keys())}")

        A = obj[key_a]
        B = obj[key_b]
        if not isinstance(A, torch.Tensor):
            A = torch.as_tensor(A)
        if not isinstance(B, torch.Tensor):
            B = torch.as_tensor(B)

        a_vec = reduce_last_dim(A, reduce=reduce)  # [M]
        b_vec = reduce_last_dim(B, reduce=reduce)  # [N]

        # ---- per-sample stats (scale-sensitive) ----
        a_mu, a_sigma, a_cv = safe_mean_std(a_vec)
        b_mu, b_sigma, b_cv = safe_mean_std(b_vec)

        a_mean_list.append(float(a_mu.item()))
        a_std_list.append(float(a_sigma.item()))
        a_cv_list.append(float(a_cv.item()))
        a_ent_list.append(normalized_entropy(a_vec))

        b_mean_list.append(float(b_mu.item()))
        b_std_list.append(float(b_sigma.item()))
        b_cv_list.append(float(b_cv.item()))
        b_ent_list.append(normalized_entropy(b_vec))

        # ---- scale-invariant normalization for pooled distribution ----
        eps = 1e-12
        a_norm = a_vec / (a_vec.mean().abs() + eps)  # around 1 if uniform
        b_norm = b_vec / (b_vec.mean().abs() + eps)

        pooled_a_norm.append(a_norm.numpy())
        pooled_b_norm.append(b_norm.numpy())

        loaded += 1

    if loaded == 0:
        raise RuntimeError("No .pt files loaded. Check data_dir and filename range.")

    pooled_a_norm = np.concatenate(pooled_a_norm, axis=0)
    pooled_b_norm = np.concatenate(pooled_b_norm, axis=0)

    a_mean = np.array(a_mean_list); a_std = np.array(a_std_list); a_cv = np.array(a_cv_list); a_ent = np.array(a_ent_list)
    b_mean = np.array(b_mean_list); b_std = np.array(b_std_list); b_cv = np.array(b_cv_list); b_ent = np.array(b_ent_list)

    print(f"Loaded files: {loaded}, missing: {missing}")
    print(f"A: CV mean={a_cv.mean():.4f}, median={np.median(a_cv):.4f} | Entropy mean={a_ent.mean():.4f}, median={np.median(a_ent):.4f}")
    print(f"B: CV mean={b_cv.mean():.4f}, median={np.median(b_cv):.4f} | Entropy mean={b_ent.mean():.4f}, median={np.median(b_ent):.4f}")

    # ---------------- Plot 1: pooled mean-normalized distribution ----------------
    plt.figure()
    bins = 80
    plt.hist(pooled_a_norm, bins=bins, alpha=0.5, density=True, label="A: x/mean(x)")
    plt.hist(pooled_b_norm, bins=bins, alpha=0.5, density=True, label="B: x/mean(x)")
    plt.axvline(1.0, linewidth=1.0)
    plt.title(f"Pooled normalized distribution (reduce={reduce})")
    plt.xlabel("x / mean(x)  (per-sample normalization)")
    plt.ylabel("density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"pooled_norm_hist_{reduce}.png", dpi=200)

    # ---------------- Plot 2: per-sample CV distribution (scale-invariant) ----------------
    plt.figure()
    bins = 60
    plt.hist(a_cv, bins=bins, alpha=0.5, density=True, label="A: CV=std/|mean| (per-sample)")
    plt.hist(b_cv, bins=bins, alpha=0.5, density=True, label="B: CV=std/|mean| (per-sample)")
    plt.title("Per-sample coefficient of variation (smaller => more uniform)")
    plt.xlabel("CV")
    plt.ylabel("density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "cv_hist.png", dpi=200)

    # ---------------- Plot 3: per-sample normalized entropy (higher => more uniform) ----------------
    plt.figure()
    bins = 60
    plt.hist(a_ent, bins=bins, alpha=0.5, density=True, label="A: normalized entropy")
    plt.hist(b_ent, bins=bins, alpha=0.5, density=True, label="B: normalized entropy")
    plt.title("Per-sample normalized entropy (higher => more uniform)")
    plt.xlabel("Entropy in [0,1]")
    plt.ylabel("density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "entropy_hist.png", dpi=200)

    # ---------------- Plot 4: mean vs std scatter (diagnose scale effect) ----------------
    plt.figure()
    plt.scatter(a_mean, a_std, s=12, alpha=0.6, label="A")
    plt.scatter(b_mean, b_std, s=12, alpha=0.6, label="B")
    plt.title("Per-sample mean vs std (scale diagnostic)")
    plt.xlabel("mean")
    plt.ylabel("std")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "mean_std_scatter.png", dpi=200)

    print(f"Saved plots to: {out_dir.resolve()}")


if __name__ == "__main__":
    # 改成你的目录，例如 "./pt_files"
    main(
        data_dir="./baseline_pt",
        reduce="mean",      # 或 "sum"
        start=0,
        end=999,
        key_a="t2m_att",
        key_b="l2m_att",
        out_dir="./attn_plots"
    )
