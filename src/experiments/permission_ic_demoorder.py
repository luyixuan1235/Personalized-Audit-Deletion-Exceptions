"""Does the ORDER and BALANCE of the demonstrations drive the exception erasure?

Zhao et al. (ICML 2021) identify two biases in in-context learning: a MAJORITY LABEL bias
(the model favours whichever label is most frequent among the demonstrations) and a RECENCY
bias (it favours labels appearing near the end of the demonstration sequence).

Wu et al.'s system uses the user's own decision history as the demonstration set. For a
permissive user that set is extremely imbalanced by construction -- 20 of 181 users have no
refusal in their history at all, and 77 have fewer than 20% refusals. If the majority-label
bias is what erases their exceptions, then manipulating the demonstrations WITHOUT changing
which decisions they contain should move the exception override rate.

Four conditions. The demonstration SET is identical in all four; only its presentation
changes, so any difference is attributable to the prompt, not to the information available.

  natural   -- as released (the baseline; already run)
  deny_last -- queries containing a refusal are moved to the END of the history
               (tests the RECENCY bias: refusals should become more salient)
  deny_first-- the same queries moved to the FRONT (the opposite prediction)
  balanced  -- grant-only queries are dropped until grants and refusals are balanced
               (tests the MAJORITY LABEL bias directly; the set SHRINKS, so this condition
               removes information -- if it still helps, the bias is what was hurting)

Usage:  OPENAI_API_KEY in ../.env
        python permission_ic_demoorder.py            # runs all three new conditions
        python permission_ic_demoorder.py deny_last  # just one
"""
import copy
import json
import os
import random
import sys

from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env"))

import permission_ic_only as IC  # noqa: E402  (reuses their prompt, model and parser verbatim)

DATA = "../data/processed_dataset.json"
SEED = 42


def has_deny(ex):
    return any("No" in v for v in ex.get("answer", {}).values())


def reorder(train, mode, rng):
    """Return a re-presented demonstration set. Same decisions, different presentation."""
    tr = list(train)
    if mode == "deny_last":
        return [e for e in tr if not has_deny(e)] + [e for e in tr if has_deny(e)]
    if mode == "deny_first":
        return [e for e in tr if has_deny(e)] + [e for e in tr if not has_deny(e)]
    if mode == "balanced":
        deny = [e for e in tr if has_deny(e)]
        grant = [e for e in tr if not has_deny(e)]
        rng.shuffle(grant)
        keep = grant[:len(deny)] if deny else grant          # drop surplus grant-only queries
        out = deny + keep
        rng.shuffle(out)
        return out
    raise ValueError(mode)


def run(mode):
    with open(DATA) as fh:
        data = json.load(fh)
    rng = random.Random(SEED)
    results = {}
    for i, (pid, pdata) in enumerate(data.items(), 1):
        d = copy.deepcopy(pdata)
        d["training"] = reorder(d.get("training", []), mode, rng)
        r = IC.process_participant(pid, d)
        results[pid] = r
        if i % 25 == 0:
            print(f"  [{mode}] {i}/{len(data)}", flush=True)
    out = f"../results/ic_{mode}_predictions.json"
    with open(out, "w") as fh:
        json.dump(results, fh, indent=1)
    print(f"[{mode}] saved -> {out}")


if __name__ == "__main__":
    modes = sys.argv[1:] or ["deny_last", "deny_first", "balanced"]
    for m in modes:
        print(f"\n######## {m}")
        run(m)
