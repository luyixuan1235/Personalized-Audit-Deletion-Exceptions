"""Data loading and dual-graph construction.

Loads the Wu et al. AI-agent permission corpus (data/processed_dataset.json) into
records, then builds, on a shared user x request bipartite node set:
  * an ALLOW graph (edges where label==1) and a DENY graph (edges where label==0);
  * per-request structured component indices (domain / tool / data_type) for the
    structured request embedding e_r = e_key + e_dom + e_tool + e_dt.

A "request" node is the contextual-integrity key (domain | tool | data_type), so
the prediction unit is the full tuple (never a bare data type).

Self-contained (adapts CI-MoE/src/ci_data.py); numpy only, no torch.
"""
import csv
import json
import os
import re
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get("PERM_DATA_DIR", os.path.join(HERE, "..", "data"))
PROCESSED = os.path.join(DATA_DIR, "processed_dataset.json")
QUERIES = os.path.join(DATA_DIR, "queries.json")
DATA_TYPES = os.path.join(DATA_DIR, "data_types.csv")

ALLOWED = {"Yes, always share": 1, "No, never share": 0}
FIRST_PARTY = {"LLM"}


def _canon(s):
    return re.sub(r"\s+", " ", str(s)).strip().lower()


def load_generic_map(path=DATA_TYPES):
    """raw data-type name -> generic (coarser) data type, from data_types.csv."""
    gmap = {}
    if not os.path.exists(path):
        return gmap
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            raw = row.get("Raw Data Type") or row.get("Generic Data Type")
            gen = row.get("Generic Data Type") or raw
            if raw:
                gmap[_canon(raw)] = gen
    return gmap


def _parse_key(pkey):
    """'receiver,datatype' -> (receiver, datatype)."""
    parts = str(pkey).split(",")
    if len(parts) >= 2:
        return parts[0].strip(), ",".join(parts[1:]).strip()
    return "Unknown", str(pkey).strip()


def build_records():
    """Return (records, bio_by_user). Each record:
    user_id, domain, tool, data_type, party, label, request_key."""
    qmap = {int(q["id"]): q for q in json.load(open(QUERIES, encoding="utf-8"))}
    gmap = load_generic_map()
    dataset = json.load(open(PROCESSED, encoding="utf-8"))
    bio_keys = ("ai_familiarity", "ai_frequency", "ai_trust",
                "privacy_importance", "age", "gender", "education")

    records, bio = [], {}
    for uid, ud in dataset.items():
        bio[uid] = {k: ud.get(k, 3 if k != "gender" else 0) for k in bio_keys}
        for split in ("training", "testing"):
            for ex in ud.get(split, []):
                qid = int(ex["id"])
                domain = qmap.get(qid, {}).get("domain", "Unknown")
                for pkey, label in ex.get("answer", {}).items():
                    y = ALLOWED.get(label)
                    if y is None:
                        continue
                    recv, dtype = _parse_key(pkey)
                    dt = gmap.get(_canon(dtype), dtype)
                    records.append({
                        "user_id": uid, "domain": domain, "tool": recv,
                        "data_type": dt,
                        "party": "first" if recv in FIRST_PARTY else "third",
                        "label": int(y),
                        "request_key": f"{domain}|{recv}|{dt}",
                    })
    return records, bio


# --------------------------------------------------------------------------- #
class Graph:
    """Index tables + dual (allow/deny) edge lists for a set of training records,
    plus the structured component indices for every request node."""

    def __init__(self, train_records, all_records=None):
        all_records = all_records or train_records
        # node indices over ALL records so test nodes exist in the graph
        self.users = _index(r["user_id"] for r in all_records)
        self.reqs = _index(r["request_key"] for r in all_records)
        self.doms = _index(r["domain"] for r in all_records)
        self.tools = _index(r["tool"] for r in all_records)
        self.dts = _index(r["data_type"] for r in all_records)
        self.n_users, self.n_reqs = len(self.users), len(self.reqs)

        # per-request component indices (for structured embedding)
        comp = {}
        for r in all_records:
            ri = self.reqs[r["request_key"]]
            comp[ri] = (self.doms[r["domain"]], self.tools[r["tool"]],
                        self.dts[r["data_type"]])
        self.req_dom = np.array([comp[i][0] for i in range(self.n_reqs)], np.int64)
        self.req_tool = np.array([comp[i][1] for i in range(self.n_reqs)], np.int64)
        self.req_dt = np.array([comp[i][2] for i in range(self.n_reqs)], np.int64)
        # inverse name maps + per-request names (for risk weighting / Sens-FPR)
        inv_dom = {v: k for k, v in self.doms.items()}
        inv_dt = {v: k for k, v in self.dts.items()}
        self.req_dom_name = [inv_dom[i] for i in self.req_dom]
        self.req_dt_name = [inv_dt[i] for i in self.req_dt]

        # dual edge sets from TRAIN labels only
        allow, deny = set(), set()
        for r in train_records:
            ui, ri = self.users[r["user_id"]], self.reqs[r["request_key"]]
            (allow if r["label"] == 1 else deny).add((ui, ri))
        self.allow_edges = np.array(sorted(allow), np.int64).reshape(-1, 2)
        self.deny_edges = np.array(sorted(deny), np.int64).reshape(-1, 2)
        # request degree per graph (for the half-cold-start / evidence diagnostics)
        self.allow_user_deg = np.bincount(self.allow_edges[:, 0],
                                          minlength=self.n_users) if len(self.allow_edges) else np.zeros(self.n_users)
        self.deny_user_deg = np.bincount(self.deny_edges[:, 0],
                                         minlength=self.n_users) if len(self.deny_edges) else np.zeros(self.n_users)

    def encode(self, records):
        """Map records to (user_idx, req_idx, label) arrays (skip unknown nodes)."""
        u, q, y = [], [], []
        for r in records:
            if r["user_id"] in self.users and r["request_key"] in self.reqs:
                u.append(self.users[r["user_id"]])
                q.append(self.reqs[r["request_key"]])
                y.append(r["label"])
        return (np.array(u, np.int64), np.array(q, np.int64),
                np.array(y, np.float32))


def _index(it):
    d = {}
    for v in it:
        if v not in d:
            d[v] = len(d)
    return d


# --------------------------------------------------------------------------- #
def per_user_folds(records, k=5, seed=42):
    """Within-user stratified k-fold: each user's records split across folds."""
    rng = np.random.default_rng(seed)
    by_user = defaultdict(list)
    for i, r in enumerate(records):
        by_user[r["user_id"]].append(i)
    fold = np.zeros(len(records), np.int64)
    for idxs in by_user.values():
        idxs = np.array(idxs)
        perm = rng.permutation(len(idxs))
        for pos, j in enumerate(perm):
            fold[idxs[j]] = pos % k
    return fold


def leave_user_out_folds(records, k=5, seed=42):
    """Group k-fold over users: whole users held out together (cold start)."""
    rng = np.random.default_rng(seed)
    users = sorted({r["user_id"] for r in records})
    perm = rng.permutation(len(users))
    umap = {users[u]: int(pos % k) for pos, u in enumerate(perm)}
    return np.array([umap[r["user_id"]] for r in records], np.int64)
