import os
import re

import numpy as np
import torch


def _parse_method(method):
    parts = method.split(":")
    name = parts[0]
    opts = {}
    for part in parts[1:]:
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        opts[key] = value
    return name, opts


def _eye_like(mat):
    return torch.eye(mat.shape[-1], dtype=mat.dtype, device=mat.device)


def _top_residual_projection(mean_vec, residual_cols, rank):
    if rank <= 0 or residual_cols.numel() == 0:
        return torch.zeros_like(mean_vec)

    gram = residual_cols.T @ residual_cols
    eigvals, eigvecs = torch.linalg.eigh(gram)
    order = torch.argsort(eigvals, descending=True)
    rank = min(rank, int((eigvals > 1e-8).sum().item()), eigvecs.shape[1])
    if rank <= 0:
        return torch.zeros_like(mean_vec)

    top_vals = eigvals[order[:rank]].clamp_min(1e-8)
    top_vecs = eigvecs[:, order[:rank]]
    basis = residual_cols @ (top_vecs / torch.sqrt(top_vals).unsqueeze(0))
    return basis @ (basis.T @ mean_vec)


def _delta_null_layer(layer_mats, rank, gamma=0.0, shrink=1.0):
    identity = _eye_like(layer_mats[0])
    deltas = torch.stack([m - identity for m in layer_mats], dim=0)
    mean_delta = deltas.mean(dim=0)
    residuals = deltas - mean_delta.unsqueeze(0)

    mean_vec = mean_delta.reshape(-1)
    residual_cols = residuals.reshape(residuals.shape[0], -1).T.contiguous()
    projected = mean_vec - _top_residual_projection(mean_vec, residual_cols, rank)
    fused = projected.reshape_as(mean_delta)

    if gamma > 0:
        consensus = torch.sign(deltas).mean(dim=0).abs().pow(gamma)
        fused = fused * consensus

    return (shrink * fused).cpu().numpy()


def _delta_consensus_layer(layer_mats, gamma=1.0, shrink=1.0):
    identity = _eye_like(layer_mats[0])
    deltas = torch.stack([m - identity for m in layer_mats], dim=0)
    consensus = torch.sign(deltas).mean(dim=0).abs().pow(gamma)
    return (shrink * deltas.mean(dim=0) * consensus).cpu().numpy()


def _mean_delta_layer(layer_mats, shrink=1.0):
    identity = _eye_like(layer_mats[0])
    deltas = torch.stack([m - identity for m in layer_mats], dim=0)
    return (shrink * deltas.mean(dim=0)).cpu().numpy()


def _trimmed_mean_layer(layer_mats, trim=1, shrink=1.0):
    identity = _eye_like(layer_mats[0])
    deltas = torch.stack([m - identity for m in layer_mats], dim=0)
    if trim <= 0 or deltas.shape[0] <= 2 * trim:
        return (shrink * deltas.mean(dim=0)).cpu().numpy()
    sorted_vals, _ = torch.sort(deltas, dim=0)
    return (shrink * sorted_vals[trim:-trim].mean(dim=0)).cpu().numpy()


def _central_layer(layer_mats, drop=1, use_delta=False, shrink=1.0):
    base = torch.stack(layer_mats, dim=0)
    if use_delta:
        base = base - _eye_like(layer_mats[0]).unsqueeze(0)
    center = base.mean(dim=0)
    residual_norm = (base - center.unsqueeze(0)).flatten(1).norm(dim=1)
    keep = max(1, base.shape[0] - int(drop))
    keep_idx = torch.argsort(residual_norm)[:keep]
    return (shrink * base[keep_idx].mean(dim=0)).cpu().numpy()


def _soft_central_layer(layer_mats, temp=1.0, use_delta=False, shrink=1.0):
    base = torch.stack(layer_mats, dim=0)
    if use_delta:
        base = base - _eye_like(layer_mats[0]).unsqueeze(0)
    center = base.mean(dim=0)
    residual_norm = (base - center.unsqueeze(0)).flatten(1).norm(dim=1)
    denom = residual_norm.mean().clamp_min(1e-8) * float(temp)
    weights = torch.softmax(-residual_norm / denom, dim=0)
    fused = torch.einsum("n,nij->ij", weights, base)
    return (shrink * fused).cpu().numpy()


def _median_consensus_delta_layer(layer_mats, gamma=1.0, shrink=1.0):
    identity = _eye_like(layer_mats[0])
    deltas = torch.stack([m - identity for m in layer_mats], dim=0)
    mean_delta = deltas.mean(dim=0)
    consensus = torch.sign(deltas).mean(dim=0).abs().pow(gamma)
    magnitude = torch.median(deltas.abs(), dim=0).values
    fused = torch.sign(mean_delta) * magnitude * consensus
    return (shrink * fused).cpu().numpy()


def _spectral_clip_delta_layer(layer_mats, rank=16, shrink=1.0, clip=""):
    identity = _eye_like(layer_mats[0])
    deltas = torch.stack([m - identity for m in layer_mats], dim=0)
    mean_delta = deltas.mean(dim=0)
    u, singular_values, vh = torch.linalg.svd(mean_delta, full_matrices=False)
    if clip:
        if clip == "median":
            max_sv = singular_values.median()
        elif clip == "mean":
            max_sv = singular_values.mean()
        else:
            max_sv = singular_values.new_tensor(float(clip))
        singular_values = torch.minimum(singular_values, max_sv)
        fused = (u * singular_values.unsqueeze(0)) @ vh
    else:
        rank = max(1, min(int(rank), singular_values.shape[0]))
        fused = (u[:, :rank] * singular_values[:rank].unsqueeze(0)) @ vh[:rank]
    return (shrink * fused).cpu().numpy()


def _loo_weighted_delta_layer(layer_mats, tau=1.0, shrink=1.0):
    identity = _eye_like(layer_mats[0])
    deltas = torch.stack([m - identity for m in layer_mats], dim=0)
    center = deltas.mean(dim=0)
    residual_norm = (deltas - center).flatten(1).norm(dim=1)
    denom = residual_norm.mean().clamp_min(1e-8) * float(tau)
    weights = torch.softmax(-residual_norm / denom, dim=0)
    fused = torch.einsum("n,nij->ij", weights, deltas)
    return (shrink * fused).cpu().numpy()


def _parse_rank(method_name, opts, default_rank):
    if "rank" in opts:
        return int(opts["rank"])
    match = re.search(r"r(\d+)", method_name)
    if match:
        return int(match.group(1))
    return int(default_rank)


def compute_xcopa_fusion(transformation_matrixs_fused_list, method, k=1, logger=None):
    name, opts = _parse_method(method)
    shrink = float(opts.get("shrink", os.environ.get("XCOPA_FUSION_SHRINK", "1.0")))
    gamma = float(opts.get("gamma", os.environ.get("XCOPA_CONSENSUS_GAMMA", "1.0")))
    trim = int(opts.get("trim", os.environ.get("XCOPA_TRIM", "1")))
    drop = int(opts.get("drop", os.environ.get("XCOPA_DROP", "1")))
    temp = float(opts.get("temp", os.environ.get("XCOPA_TEMP", "1.0")))
    rank = _parse_rank(name, opts, k)
    tau = float(opts.get("tau", os.environ.get("XCOPA_LOO_TAU", "1.0")))
    spectral_clip = opts.get("clip", os.environ.get("XCOPA_SPECTRAL_CLIP", ""))

    mats = [torch.as_tensor(m, dtype=torch.float32, device="cuda") for m in transformation_matrixs_fused_list]
    num_layers = mats[0].shape[0]
    fused_layers = []

    if logger is not None:
        logger.info(f"XCOPA_FUSION_METHOD={method}, rank={rank}, gamma={gamma}, shrink={shrink}, trim={trim}")

    for layer in range(num_layers):
        layer_mats = [m[layer] for m in mats]
        if name in {"mean_w", "mean"}:
            fused = torch.stack(layer_mats, dim=0).mean(dim=0).cpu().numpy()
        elif name in {"mean_delta", "delta_mean"}:
            fused = _mean_delta_layer(layer_mats, shrink=shrink)
        elif name.startswith("median_consensus_delta") or name.startswith("median_consensus"):
            fused = _median_consensus_delta_layer(layer_mats, gamma=gamma, shrink=shrink)
        elif name.startswith("spectral_clip_delta") or name.startswith("spectral_delta"):
            fused = _spectral_clip_delta_layer(layer_mats, rank=rank, shrink=shrink, clip=spectral_clip)
        elif name.startswith("loo_weighted_delta") or name.startswith("loo_delta"):
            fused = _loo_weighted_delta_layer(layer_mats, tau=tau, shrink=shrink)
        elif name.startswith("delta_consensus") or name.startswith("consensus"):
            fused = _delta_consensus_layer(layer_mats, gamma=gamma, shrink=shrink)
        elif name.startswith("delta_null_consensus") or name.startswith("null_consensus"):
            fused = _delta_null_layer(layer_mats, rank=rank, gamma=gamma, shrink=shrink)
        elif name.startswith("delta_null") or name.startswith("null"):
            fused = _delta_null_layer(layer_mats, rank=rank, gamma=0.0, shrink=shrink)
        elif name.startswith("trimmed_delta"):
            fused = _trimmed_mean_layer(layer_mats, trim=trim, shrink=shrink)
        elif name.startswith("central_delta"):
            fused = _central_layer(layer_mats, drop=drop, use_delta=True, shrink=shrink)
        elif name.startswith("central_w") or name.startswith("central"):
            fused = _central_layer(layer_mats, drop=drop, use_delta=False, shrink=shrink)
        elif name.startswith("soft_delta"):
            fused = _soft_central_layer(layer_mats, temp=temp, use_delta=True, shrink=shrink)
        elif name.startswith("soft_w") or name.startswith("soft"):
            fused = _soft_central_layer(layer_mats, temp=temp, use_delta=False, shrink=shrink)
        else:
            raise ValueError(f"Unknown XCOPA_FUSION_METHOD: {method}")
        fused_layers.append(fused)

    return np.stack(fused_layers, axis=0)
