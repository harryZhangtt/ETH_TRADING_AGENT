from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as F


Reduction = Literal["none", "mean", "sum"]


def _reduce(x: torch.Tensor, reduction: Reduction) -> torch.Tensor:
    if reduction == "none":
        return x
    if reduction == "mean":
        return x.mean()
    if reduction == "sum":
        return x.sum()
    raise ValueError(f"Unsupported reduction: {reduction}")


def _flatten_values(values: torch.Tensor) -> torch.Tensor:
    if values.ndim == 1:
        return values
    if values.ndim == 2 and values.shape[-1] == 1:
        return values.squeeze(-1)
    raise ValueError(
        f"Value tensor must have shape [B] or [B, 1], got {tuple(values.shape)}"
    )


def categorical_kl_from_logits(
    logits_p: torch.Tensor,
    logits_q: torch.Tensor,
    reduction: Reduction = "mean",
) -> torch.Tensor:
    if logits_p.ndim != 2 or logits_q.ndim != 2:
        raise ValueError(
            "categorical_kl_from_logits expects both inputs to have shape [B, A], "
            f"got {tuple(logits_p.shape)} and {tuple(logits_q.shape)}"
        )
    if logits_p.shape != logits_q.shape:
        raise ValueError(
            f"Shape mismatch: logits_p has shape {tuple(logits_p.shape)}, "
            f"logits_q has shape {tuple(logits_q.shape)}"
        )

    logp = F.log_softmax(logits_p, dim=-1)
    logq = F.log_softmax(logits_q, dim=-1)
    p = logp.exp()

    kl_per_sample = (p * (logp - logq)).sum(dim=-1)
    return _reduce(kl_per_sample, reduction=reduction)


def policy_consistency_loss(
    ref_logits: torch.Tensor,
    aug_logits: torch.Tensor,
    detach_ref: bool = True,
    reduction: Reduction = "mean",
) -> torch.Tensor:
    if detach_ref:
        ref_logits = ref_logits.detach()

    return categorical_kl_from_logits(
        logits_p=ref_logits,
        logits_q=aug_logits,
        reduction=reduction,
    )


def value_consistency_loss(
    ref_values: torch.Tensor,
    aug_values: torch.Tensor,
    detach_ref: bool = True,
    reduction: Reduction = "mean",
) -> torch.Tensor:
    ref_values = _flatten_values(ref_values)
    aug_values = _flatten_values(aug_values)

    if ref_values.shape != aug_values.shape:
        raise ValueError(
            f"Shape mismatch: ref_values has shape {tuple(ref_values.shape)}, "
            f"aug_values has shape {tuple(aug_values.shape)}"
        )

    if detach_ref:
        ref_values = ref_values.detach()

    loss_per_sample = (ref_values - aug_values) ** 2
    return _reduce(loss_per_sample, reduction=reduction)
