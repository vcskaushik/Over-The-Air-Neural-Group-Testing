"""Privacy losses for the encoder's outer step in Stage B.

Two scientifically distinct forms (selected by config in Stage B):

* `priv_loss_ce`: negated adversary cross-entropy. Encoder is rewarded for making
  the adversary's prediction wrong (any way, including confidently wrong). Has a
  label-shift degenerate optimum but is the standard DANN-style baseline.

* `priv_loss_entropy` (single-label) and `priv_loss_entropy_multilabel`:
  negative entropy of the adversary's (per-class) softmax/sigmoid output. Encoder
  is rewarded for making the adversary's posterior maximally uncertain. No
  degenerate optimum; equivalent up to a constant to KL-to-uniform.
"""
import torch
import torch.nn.functional as F


def priv_loss_ce(adv_logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Negated adversary cross-entropy.

    The encoder MINIMIZES this loss, which equals MAXIMIZING the adversary's CE.
    """
    return -F.cross_entropy(adv_logits, target)


def priv_loss_entropy(adv_logits: torch.Tensor) -> torch.Tensor:
    """Negative entropy of adversary's softmax distribution (single-label, ITIT)."""
    log_p = F.log_softmax(adv_logits, dim=-1)
    p = log_p.exp()
    # H(p) = -sum p log p ; loss = -H = sum p log p, then mean over batch
    return (p * log_p).sum(dim=-1).mean()


def priv_loss_entropy_multilabel(adv_logits: torch.Tensor) -> torch.Tensor:
    """Negative sum of per-class Bernoulli entropies (multilabel, GTGT-FM).

    For each of K class outputs, computes H(Bernoulli(sigmoid(z_k))) and sums them.
    Loss = -H_total. Minimum value = -K * log 2 (when all probs = 0.5).
    """
    log_p = F.logsigmoid(adv_logits)            # log p_k
    log_one_minus_p = F.logsigmoid(-adv_logits)  # log(1 - p_k)
    p = log_p.exp()
    one_minus_p = log_one_minus_p.exp()
    # H_k = -[p log p + (1-p) log(1-p)] ; sum over classes ; loss = -sum_k H_k = sum_k [p log p + (1-p) log(1-p)]
    per_class = p * log_p + one_minus_p * log_one_minus_p
    return per_class.sum(dim=-1).mean()
