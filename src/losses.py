"""Training objectives for the dual-graph model.

  * evidential_loss : the Refusal Learning objective (Sec. 4.3 of the paper) --
      digamma-form expected cross-entropy under the Beta opinion (Sensoy et al. 2018)
      + a KL-to-uniform-prior regularizer that shrinks misleading evidence (so the
      model raises its uncertainty / abstains when no class is supported), with a
      per-domain risk weight omega on the true-deny branch.
  * bpr_loss : ranking on the net signed score (the ablation that does NOT identify
      the two evidence channels).
"""
import torch
import torch.nn.functional as F


def kl_dir2(a, b):
    """KL( Dir(a,b) || Dir(1,1) ) for the binary (Beta) case."""
    s = a + b
    return (torch.lgamma(s) - torch.lgamma(a) - torch.lgamma(b)
            + (a - 1.0) * (torch.digamma(a) - torch.digamma(s))
            + (b - 1.0) * (torch.digamma(b) - torch.digamma(s)))


def evidential_loss(e_a, e_d, y, omega=1.0, lam_kl=0.0):
    """e_a,e_d: non-negative evidence; y in {0,1} (1=allow); omega>=1 risk weight on
    the true-deny branch; lam_kl: annealed refusal-regularizer weight."""
    alpha_a = e_a + 1.0
    alpha_d = e_d + 1.0
    S = alpha_a + alpha_d
    # expected cross-entropy (digamma form); both channels supervised independently
    ce = (y * (torch.digamma(S) - torch.digamma(alpha_a))
          + omega * (1.0 - y) * (torch.digamma(S) - torch.digamma(alpha_d)))
    # refusal regularizer: strip the CORRECT-class evidence, pull the rest to Dir(1)
    at_a = torch.where(y > 0.5, torch.ones_like(alpha_a), alpha_a)
    at_d = torch.where(y > 0.5, alpha_d, torch.ones_like(alpha_d))
    kl = kl_dir2(at_a, at_d)
    return (ce + lam_kl * kl).mean()


def bpr_loss(net_pos, net_neg):
    """net_* = signed score s_allow - s_deny for a positive/negative request."""
    return -F.logsigmoid(net_pos - net_neg).mean()
