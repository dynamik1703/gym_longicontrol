"""Exact finite-support categorical expectations, including singleton support."""

import torch


def distribution(logits, mask):
    if logits.shape != mask.shape or logits.shape[-1] != 21:
        raise ValueError("categorical shape must end in 21")
    if mask.dtype != torch.bool or not mask.any(-1).all():
        raise ValueError("empty or nonboolean categorical support")
    if not torch.isfinite(logits).all():
        raise FloatingPointError("nonfinite actor logits")
    raw = torch.log_softmax(logits.masked_fill(~mask, -torch.inf), dim=-1)
    # The mathematical log probability of an invalid action is -inf. Return
    # zero in that slot solely for safe finite expectation arithmetic (0 log 0).
    logp = torch.where(mask, raw, torch.zeros_like(raw))
    probability = torch.where(mask, raw.exp(), torch.zeros_like(raw))
    entropy = -(probability * logp).sum(-1, keepdim=True)
    target = 0.98 * mask.sum(-1, keepdim=True).to(logits.dtype).log()
    return probability, logp, entropy, target


def sample(logits, mask, generator, deterministic=False):
    p, _, _, _ = distribution(logits, mask)
    return p.argmax(-1) if deterministic else torch.multinomial(
        p, 1, generator=generator
    ).squeeze(-1)
