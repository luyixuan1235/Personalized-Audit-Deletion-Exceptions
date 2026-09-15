"""Exception Override Rate (EOR) — paper Table 1.

Primary analysis uses train-only majority (P0-2). Evaluation-split majority is
kept as sensitivity inside table53_eor_deployed.json.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_p0_p1_hardening as H

if __name__ == "__main__":
    H.main()
