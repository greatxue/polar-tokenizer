"""
核心设计
--------
* **完全可选** – 传给 `compute_metrics()` 你手里有的张量/路径，它就算
  得到；没有依赖就跳过。
* **双曲一致** – 任何依赖几何的误差，都用 Lorentz 距离。
* **速度监控** – 装饰器 `@timed` 帮你测毫秒级函数时延。

Quick example
-------------
>>> from metrics import compute_metrics, timed
>>> q, commit_ms = timed(vq_module)(latent)   # 计时量化器
>>> m = compute_metrics(
        recon_img=recon, target_img=img,       # PSNR / LPIPS
        token_ids=ids, codebook_size=K,        # Active / Entropy
        hyper_orig=latent_h, hyper_quant=q,    # Commit (Lorentz)
        real_dir='png/real', gen_dir='png/gen',# FID
        radial_bins_v=r_vis, radial_bins_t=r_txt,
        want=['psnr','lpips','active','entropy','gini',
              'commit','radial_kl','fid'] )
>>> m['lookup_ms'] = commit_ms
>>> wandb.log(m)

Required pip packages
---------------------
metric        extra deps            pip install …
------------  -------------------  -----------------------------
PSNR/MSE     纯 torch              –
LPIPS         lpips + torchvision  lpips
FID           pytorch-fid          pytorch-fid

"""
from __future__ import annotations

import math, time
from pathlib import Path
from typing import Dict, Sequence, Any, Iterable, Union

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Lorentz helpers (no external deps) ----------------------------------------
# ---------------------------------------------------------------------------

def _lorentz_inner(u: torch.Tensor, v: torch.Tensor):
    return -u[..., 0] * v[..., 0] + torch.sum(u[..., 1:] * v[..., 1:], dim=-1)

def lorentz_distance(u: torch.Tensor, v: torch.Tensor):
    inner = torch.clamp(-_lorentz_inner(u, v), min=1.0 + 1e-5)
    return torch.acosh(inner)

# ---------------------------------------------------------------------------
# Optional heavy deps (lazy) -------------------------------------------------
# ---------------------------------------------------------------------------

_lpips_handle = None

def _get_lpips():
    global _lpips_handle
    if _lpips_handle is None:
        try:
            import lpips  # type: ignore
        except ImportError as e:
            raise ImportError('LPIPS metric requested but `pip install lpips` missing') from e
        _lpips_handle = lpips.LPIPS(net='vgg').eval().cuda() if torch.cuda.is_available() else lpips.LPIPS(net='vgg').eval()
    return _lpips_handle


def _fid_given_paths(real: str | Path, gen: str | Path, batch: int, device: str):
    try:
        from pytorch_fid.fid_score import calculate_fid_given_paths  # type: ignore
    except ImportError as e:
        raise ImportError('FID requested but `pip install pytorch-fid` missing') from e
    return calculate_fid_given_paths([str(real), str(gen)], batch, device, dims=2048)

# ---------------------------------------------------------------------------
# Codebook utilisation helpers ----------------------------------------------
# ---------------------------------------------------------------------------

def _hits(ids: torch.Tensor, K: int):
    if ids.dtype not in (torch.int32, torch.int64):
        ids = ids.long()
    return torch.bincount(ids.flatten(), minlength=K).float()

def _entropy(h: torch.Tensor, K: int):
    p = h / h.sum().clamp(min=1)
    return -(p * (p + 1e-12).log()).sum().item() / math.log(K)

def _gini(h: torch.Tensor):
    h_sorted = h.sort()[0]
    cum = h_sorted.cumsum(0)
    return 1 - 2 * (cum / cum[-1]).mean().item()

# ---------------------------------------------------------------------------
# Main entry -----------------------------------------------------------------
# ---------------------------------------------------------------------------

def compute_metrics(*,
                    recon_img: torch.Tensor | None = None,
                    target_img: torch.Tensor | None = None,
                    token_ids: torch.Tensor | None = None,
                    codebook_size: int | None = None,
                    hyper_orig: torch.Tensor | None = None,
                    hyper_quant: torch.Tensor | None = None,
                    radial_bins_v: torch.Tensor | None = None,
                    radial_bins_t: torch.Tensor | None = None,
                    real_dir: str | Path | None = None,
                    gen_dir: str | Path | None = None,
                    batch_size_fid: int = 50,
                    device_fid: str | None = None,
                    want: Iterable[str] | None = None) -> Dict[str, Any]:
    """Flexible metric computation.  Only metrics listed in *want* are returned."""
    want = set(want or ['psnr', 'active', 'entropy'])
    out: Dict[str, Any] = {}

    # -------------------------------------------------- reconstruction domain
    if recon_img is not None and target_img is not None:
        if {'psnr', 'mse'} & want:
            mse = F.mse_loss(recon_img, target_img).item()
            if 'mse' in want:
                out['mse'] = mse
            if 'psnr' in want:
                out['psnr'] = 10 * math.log10(1. / (mse + 1e-12))
        if 'lpips' in want:
            lp = _get_lpips()
            with torch.no_grad():
                out['lpips'] = lp(recon_img, target_img).mean().item()

    # ------------------------------------------- codebook utilisation metrics
    if token_ids is not None and codebook_size is not None and {'active','entropy','gini'} & want:
        hits = _hits(token_ids, codebook_size)
        if 'active' in want:
            out['active'] = (hits > 0).float().mean().item()
        if 'entropy' in want:
            out['entropy'] = _entropy(hits, codebook_size)
        if 'gini' in want:
            out['gini'] = _gini(hits)

    # ---------------------------------------------------- commit (hyperbolic)
    if hyper_orig is not None and hyper_quant is not None and 'commit' in want:
        out['commit'] = lorentz_distance(hyper_orig, hyper_quant).pow(2).mean().item()

    # --------------------------------------------------- radial KL (layer-align)
    if radial_bins_v is not None and radial_bins_t is not None and 'radial_kl' in want:
        vmax = int(max(radial_bins_v.max(), radial_bins_t.max()).item()) + 1
        pv = torch.bincount(radial_bins_v.flatten(), minlength=vmax).float() + 1e-6
        pt = torch.bincount(radial_bins_t.flatten(), minlength=vmax).float() + 1e-6
        pv /= pv.sum(); pt /= pt.sum()
        out['radial_kl'] = (pv * (pv / pt).log()).sum().item()

    # ------------------------------------------------------------- FID
    if 'fid' in want and real_dir and gen_dir:
        out['fid'] = _fid_given_paths(real_dir, gen_dir, batch_size_fid, device_fid or ('cuda' if torch.cuda.is_available() else 'cpu'))

    return out

# ---------------------------------------------------------------------------
# Timing decorator -----------------------------------------------------------
# ---------------------------------------------------------------------------

def timed(fn):
    """Wrap a function to also measure wall-time (ms). Usage:  result, ms = timed(fn)(*args)"""
    def _wrap(*a, **kw):
        t0 = time.perf_counter(); out = fn(*a, **kw); return out, (time.perf_counter()-t0)*1000
    return _wrap


