"""Dual-graph LightGCN model with structured request embeddings and a
two-sided (Beta opinion) evidential head.

  e_allow = softplus(<u_a, r_a>),  e_deny = softplus(<u_d, r_d>)
  alpha_a = e_a + 1, alpha_d = e_d + 1, S = e_a + e_d + 2
  p_allow = alpha_a / S,  u = 2 / S          (uncertainty mass)

Flags isolate the ablations:
  use_deny=False     -> single allow graph, single score (M1 baseline)
  use_sideinfo=False -> request node = bare request-key embedding (no domain/tool/type)
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def build_norm_adj(n_users, n_reqs, edges, device):
    """Symmetric-normalized sparse adjacency over [users; requests] (size N x N)."""
    N = n_users + n_reqs
    if len(edges) == 0:
        idx = torch.empty((2, 0), dtype=torch.long, device=device)
        val = torch.empty((0,), device=device)
        return torch.sparse_coo_tensor(idx, val, (N, N)).coalesce()
    u = edges[:, 0]
    r = edges[:, 1] + n_users
    src = np.concatenate([u, r])
    dst = np.concatenate([r, u])
    deg = np.bincount(src, minlength=N).astype(np.float32)
    dinv = 1.0 / np.sqrt(np.maximum(deg, 1.0))
    val = dinv[src] * dinv[dst]
    idx = torch.tensor(np.stack([src, dst]), dtype=torch.long, device=device)
    val = torch.tensor(val, dtype=torch.float32, device=device)
    return torch.sparse_coo_tensor(idx, val, (N, N)).coalesce()


def build_signed_adj(n_users, n_reqs, allow_edges, deny_edges, device):
    """Single-space SIGNED adjacency: allow edges +1, deny edges -1, normalized by
    |degree| (Derr et al. 2018 style negative message passing). This is the prior
    approach this model avoids by keeping two independent spaces."""
    N = n_users + n_reqs
    src, dst, val = [], [], []
    for edges, sign in ((allow_edges, 1.0), (deny_edges, -1.0)):
        if len(edges) == 0:
            continue
        u = edges[:, 0]; r = edges[:, 1] + n_users
        src += [u, r]; dst += [r, u]
        val += [np.full(len(u), sign, np.float32)] * 2
    if not src:
        idx = torch.empty((2, 0), dtype=torch.long, device=device)
        return torch.sparse_coo_tensor(idx, torch.empty(0, device=device), (N, N)).coalesce()
    src = np.concatenate(src); dst = np.concatenate(dst); val = np.concatenate(val)
    deg = np.bincount(src, weights=np.abs(val), minlength=N).astype(np.float32)
    dinv = 1.0 / np.sqrt(np.maximum(deg, 1.0))
    w = val * dinv[src] * dinv[dst]
    idx = torch.tensor(np.stack([src, dst]), dtype=torch.long, device=device)
    return torch.sparse_coo_tensor(idx, torch.tensor(w, device=device), (N, N)).coalesce()


class Branch(nn.Module):
    """One LightGCN evidence channel (its own embeddings + structured components).

    comp = (use_domain, use_tool, use_dt) toggles each structured component
    (for the Table-3 ablation A/B/C/D); combine in {'sum','proj'}: 'sum' is the
    additive e_r = e_key + e_dom + e_tool + e_dt (Eq. reqemb); 'proj' concatenates
    all four and linearly projects (the D-proj variant)."""

    def __init__(self, n_users, n_reqs, n_dom, n_tool, n_dt, dim, layers,
                 comp=(True, True, True), combine="sum"):
        super().__init__()
        self.n_users, self.n_reqs, self.layers = n_users, n_reqs, layers
        self.comp, self.combine = comp, combine
        self.user_emb = nn.Embedding(n_users, dim)
        self.req_emb = nn.Embedding(n_reqs, dim)
        self.dom_emb = nn.Embedding(n_dom, dim)
        self.tool_emb = nn.Embedding(n_tool, dim)
        self.dt_emb = nn.Embedding(n_dt, dim)
        for m in (self.user_emb, self.req_emb, self.dom_emb, self.tool_emb, self.dt_emb):
            nn.init.normal_(m.weight, std=0.1)
        self.proj = nn.Linear(4 * dim, dim) if combine == "proj" else None

    def req_e0(self, req_dom, req_tool, req_dt):
        if self.combine == "proj":   # concat all four, project (D-proj)
            return self.proj(torch.cat([self.req_emb.weight, self.dom_emb(req_dom),
                                        self.tool_emb(req_tool), self.dt_emb(req_dt)], -1))
        e = self.req_emb.weight
        if self.comp[0]:
            e = e + self.dom_emb(req_dom)
        if self.comp[1]:
            e = e + self.tool_emb(req_tool)
        if self.comp[2]:
            e = e + self.dt_emb(req_dt)
        return e

    def propagate(self, adj, req_dom, req_tool, req_dt):
        x = torch.cat([self.user_emb.weight, self.req_e0(req_dom, req_tool, req_dt)], 0)
        out = x
        for _ in range(self.layers):
            x = torch.sparse.mm(adj, x)
            out = out + x
        out = out / (self.layers + 1)
        return out[:self.n_users], out[self.n_users:]


class DualGraph(nn.Module):
    """mode: 'dual'  -> two-sided evidence (M3/M4, evidential)
             'allow' -> single allow graph, single score (M1 baseline)
             'deny'  -> single deny graph, single score (M2 deny-only diagnostic)"""

    def __init__(self, graph, dim=32, layers=3, mode="dual",
                 comp=(True, True, True), combine="sum", device="cpu"):
        super().__init__()
        self.mode = mode
        self.device = device
        g = graph
        self.req_dom = torch.tensor(g.req_dom, device=device)
        self.req_tool = torch.tensor(g.req_tool, device=device)
        self.req_dt = torch.tensor(g.req_dt, device=device)
        self.n_users, self.n_reqs = g.n_users, g.n_reqs
        self.adj_allow = build_norm_adj(g.n_users, g.n_reqs, g.allow_edges, device)
        self.adj_deny = build_norm_adj(g.n_users, g.n_reqs, g.deny_edges, device)
        args = (g.n_users, g.n_reqs, len(g.doms), len(g.tools), len(g.dts),
                dim, layers, comp, combine)
        self.allow = Branch(*args) if mode in ("dual", "allow") else None
        self.deny = Branch(*args) if mode in ("dual", "deny") else None
        self.to(device)

    def rebuild_adj(self, allow_edges, deny_edges):
        """Swap the propagation graphs (e.g. add a new user's revealed history for
        cold-start warm-up) without changing any learned parameters."""
        self.adj_allow = build_norm_adj(self.n_users, self.n_reqs, allow_edges, self.device)
        self.adj_deny = build_norm_adj(self.n_users, self.n_reqs, deny_edges, self.device)

    def scores(self, u_idx, q_idx):
        z = None
        if self.allow is not None:
            ua, ra = self.allow.propagate(self.adj_allow, self.req_dom, self.req_tool, self.req_dt)
            s_allow = (ua[u_idx] * ra[q_idx]).sum(-1)
            z = s_allow
        else:
            s_allow = None
        if self.deny is not None:
            ud, rd = self.deny.propagate(self.adj_deny, self.req_dom, self.req_tool, self.req_dt)
            s_deny = (ud[u_idx] * rd[q_idx]).sum(-1)
            z = s_deny if z is None else z
        else:
            s_deny = None
        zero = torch.zeros_like(z)
        return (s_allow if s_allow is not None else zero,
                s_deny if s_deny is not None else zero)

    def opinion(self, u_idx, q_idx):
        """Return (p_allow, u_uncertainty, e_allow, e_deny)."""
        s_allow, s_deny = self.scores(u_idx, q_idx)
        if self.mode == "dual":
            e_a = F.softplus(s_allow)
            e_d = F.softplus(s_deny)
        elif self.mode == "allow":
            p = torch.sigmoid(s_allow)
            e_a = p / (1 - p + 1e-6); e_d = torch.zeros_like(e_a)
        else:  # deny-only: single score = -s_deny
            p = torch.sigmoid(-s_deny)
            e_a = p / (1 - p + 1e-6); e_d = torch.zeros_like(e_a)
        S = e_a + e_d + 2.0
        p_allow = (e_a + 1.0) / S
        u = 2.0 / S
        return p_allow, u, e_a, e_d


class SignedGCN(nn.Module):
    """Single-space signed-graph baseline (Derr et al. 2018 style): one embedding
    set propagated over a SIGNED adjacency (allow +1 / deny -1) with negative
    message passing, scored by a single inner product. Uses the SAME structured
    request embedding as the dual-graph model, so the only difference is single-space-signed
    vs. two independent evidence spaces."""

    def __init__(self, graph, dim=32, layers=3, use_sideinfo=True, device="cpu"):
        super().__init__()
        self.device = device
        g = graph
        self.req_dom = torch.tensor(g.req_dom, device=device)
        self.req_tool = torch.tensor(g.req_tool, device=device)
        self.req_dt = torch.tensor(g.req_dt, device=device)
        self.n_users, self.n_reqs = g.n_users, g.n_reqs
        comp = (use_sideinfo,) * 3
        self.b = Branch(g.n_users, g.n_reqs, len(g.doms), len(g.tools), len(g.dts),
                        dim, layers, comp, "sum")
        self.adj = build_signed_adj(g.n_users, g.n_reqs, g.allow_edges, g.deny_edges, device)
        self.to(device)

    def rebuild_adj(self, allow_edges, deny_edges):
        """Swap the signed graph (e.g. add a new user's revealed history for cold-start)."""
        self.adj = build_signed_adj(self.n_users, self.n_reqs, allow_edges, deny_edges,
                                    self.device)

    def scores(self, u_idx, q_idx):
        uu, rr = self.b.propagate(self.adj, self.req_dom, self.req_tool, self.req_dt)
        return (uu[u_idx] * rr[q_idx]).sum(-1)

    def opinion(self, u_idx, q_idx):
        """Match the dual-graph interface: (p_allow, pseudo-uncertainty, e_a, e_d)."""
        s = self.scores(u_idx, q_idx)
        p = torch.sigmoid(s)
        e_a = p / (1 - p + 1e-6)
        return p, 1 - torch.abs(2 * p - 1), e_a, torch.zeros_like(e_a)
