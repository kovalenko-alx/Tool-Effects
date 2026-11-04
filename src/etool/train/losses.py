"""Span-masked objectives. All functions take logits (B, L, V) and ids (B, L); spans select target positions.
Token t is predicted by position t-1, so masks are shifted internally."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def token_logps(logits: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    """log p(ids[:, 1:]) -> (B, L-1)."""
    lp = F.log_softmax(logits[:, :-1].float(), dim=-1)
    return lp.gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)


def shift_mask(mask: torch.Tensor) -> torch.Tensor:
    """Mask over target tokens (B, L) -> mask over predicting positions (B, L-1)."""
    return mask[:, 1:]


# The *_lp variants take the (B, L-1) token log-probs from ``token_logps`` so one log-softmax over the vocabulary serves every
# term of a pair step (each extra float32 log-softmax of a (2, 4096, 151k) logits tensor is 4.3 GB; job 11835829 OOMed on it).
def nll_lp(lp: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = shift_mask(mask).float()
    return -(lp * m).sum() / m.sum().clamp(min=1)


def nll_loss(logits: torch.Tensor, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return nll_lp(token_logps(logits, ids), mask)


def unlikelihood_lp(lp: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """-mean log(1 - p(y_t)) over the span (Welleck et al. 2019)."""
    p = lp.exp().clamp(max=1 - eps)
    m = shift_mask(mask).float()
    return -(torch.log1p(-p) * m).sum() / m.sum().clamp(min=1)


def unlikelihood_loss(logits: torch.Tensor, ids: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return unlikelihood_lp(token_logps(logits, ids), mask, eps)


def seq_logp_lp(lp: torch.Tensor, mask: torch.Tensor, average: bool) -> torch.Tensor:
    """Per-sequence log-prob of the span (sum or token-mean) -> (B,)."""
    m = shift_mask(mask).float()
    s = (lp * m).sum(-1)
    return s / m.sum(-1).clamp(min=1) if average else s


def seq_logp(logits: torch.Tensor, ids: torch.Tensor, mask: torch.Tensor, average: bool) -> torch.Tensor:
    return seq_logp_lp(token_logps(logits, ids), mask, average)


def npo_loss(logp_theta: torch.Tensor, logp_ref: torch.Tensor, beta: float) -> torch.Tensor:
    """Negative Preference Optimization (Zhang et al. 2024): (2/beta) * log(1 + (pi/pi_ref)^beta), averaged."""
    return (2.0 / beta) * F.softplus(beta * (logp_theta - logp_ref)).mean()


def dpo_loss(c_theta: torch.Tensor, c_ref: torch.Tensor, r_theta: torch.Tensor, r_ref: torch.Tensor, beta: float) -> torch.Tensor:
    margin = beta * ((c_theta - c_ref) - (r_theta - r_ref))
    return -F.logsigmoid(margin).mean()


def drop_tokens(mask: torch.Tensor, ids: torch.Tensor, drop_ids: torch.Tensor) -> torch.Tensor:
    """Remove the positions whose token is in ``drop_ids`` from a span mask (B, L). A row that would become empty is
    returned unchanged. Used to keep closing ``</tool_call>`` tags out of the suppression span: closing a call is never
    the decision that made it wrong, and pushing the tag down teaches the model not to close calls (RESULTS §7)."""
    m = mask.bool()
    kept = m & ~torch.isin(ids, drop_ids)
    empty = kept.sum(-1) == 0
    return torch.where(empty[:, None], m, kept)
