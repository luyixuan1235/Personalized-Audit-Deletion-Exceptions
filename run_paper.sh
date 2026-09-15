#!/usr/bin/env bash
# Reproduce every table in the paper.
#
#   bash run_paper.sh              # our experiments, both corpora
#   bash run_paper.sh --wu         # + reproduce the Wu et al. released system
#   bash run_paper.sh --only wu    # ONLY the Wu et al. reproduction
#   bash run_paper.sh --only spa   # ONLY the SPA corpus (the slow half)
#
# Env:
#   PERM_QUICK=1     15 epochs instead of 200 -- smoke test, numbers are NOT valid
#   OMP_NUM_THREADS=8   (recommended on Apple silicon / many-core machines)
#
# Runtime (8 performance cores): Wu ~15 min, SPA ~90 min, Wu-repo ~10 min.
# The SPA half is dominated by the per-user threshold scans, which are O(users).
set -euo pipefail
cd "$(dirname "$0")"

WU_REPRO=0; ONLY=""
while [ $# -gt 0 ]; do
  case "$1" in
    --wu) WU_REPRO=1 ;;
    --only) shift; ONLY="${1:-}" ;;
    *) echo "unknown flag: $1"; exit 1 ;;
  esac; shift
done

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
[ -d .venv ] || { echo "No .venv -- run: bash setup.sh"; exit 1; }

activate_venv() {
  local root="${1:-.venv}"
  if [ -f "$root/Scripts/activate" ]; then
    # shellcheck disable=SC1091
    source "$root/Scripts/activate"
  elif [ -f "$root/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "$root/bin/activate"
  else
    echo "No activation script found in $root"; return 1
  fi
}

run() { echo; echo "-------- $*"; "$@"; }

# ---------------------------------------------------------------- our experiments
ours() {
  local tag="$1"; shift
  activate_venv .venv
  cd src
  echo; echo "================================================================"
  echo "  [$tag]  $(date +%H:%M:%S)"
  echo "================================================================"

  # --- the paper's core results (new in this version)
  run python experiments/run_peruser.py         # Tab 21: per-user harm concentration
  run python experiments/run_remedy.py          # Tab 22: objective x arch @ deployed thr
  run python experiments/run_peruser_honest.py  # Tab 23: per-user thr, honest vs oracle
  run python experiments/run_offset_spread.py   # Tab 19: offset heterogeneity (eta)
  run python experiments/run_posthoc.py         # Tab 17: post-hoc calibration control

  # --- supporting results
  run python experiments/run_variance.py        # Tab 12: per-fold values (needed next)
  run python experiments/run_significance.py    # Tab 15: paired t-tests (reads Tab 12)
  run python experiments/run_grid3.py           # Tab 16: 3x2 grid
  run python experiments/run_evi_sensitivity.py # Tab 18: evidential lambda sweep
  run python experiments/run_evi_aurc.py        # Tab 20: evidential vs signed on AURC
  run python experiments/run_cost.py            # Tab 13: cost-sensitive objectives
  run python experiments/run_compogen.py        # Tab  7: compositional generalization

  cd ..
  deactivate
}

if [ "$ONLY" != "wu" ]; then
  if [ "$ONLY" != "spa" ]; then
    ours "Wu corpus -> results/"
  fi
  if [ "$ONLY" != "wu-corpus" ]; then
    export PERM_DATA_DIR=../data_spa PERM_RESULTS_DIR=../results_spa
    mkdir -p results_spa
    ours "SPA corpus -> results_spa/"
    unset PERM_DATA_DIR PERM_RESULTS_DIR
  fi
fi

# ------------------------------------------------- Wu et al. released system
if [ "$WU_REPRO" = "1" ] || [ "$ONLY" = "wu" ]; then
  [ -d wu-repo/.venv ] || { echo "No wu-repo/.venv -- run: bash setup.sh --wu"; exit 1; }
  echo; echo "================================================================"
  echo "  [Wu et al. released system]  $(date +%H:%M:%S)"
  echo "================================================================"
  cp src/experiments/analyze_wu_peruser.py wu-repo/src/
  cd wu-repo/src
  activate_venv ../.venv

  run python permission_cf_only.py          # LightGCN + BPR. No API key needed.

  if [ -f ../.env ] && grep -q OPENAI_API_KEY ../.env; then
    run python permission_ic_only.py        # LLM alone            (API key, ~$1)
    run python permission_ic_cf.py          # the deployed hybrid  (API key, ~$1)
  else
    echo
    echo "!! No OPENAI_API_KEY in wu-repo/.env -- skipping the LLM models."
    echo "   Without them we can only characterise the CF COMPONENT, not the"
    echo "   deployed hybrid, and the paper's scope must be limited accordingly."
    echo "   To run them:  echo 'OPENAI_API_KEY=sk-...' > wu-repo/.env"
  fi

  run python analyze_wu_peruser.py --results ../results
  deactivate
  cd ../..
fi

echo; echo "================================================================"
echo "  DONE  $(date +%H:%M:%S)"
echo "  results/            Wu corpus"
echo "  results_spa/        SPA corpus"
echo "  wu-repo/results/    Wu et al. released system"
echo "================================================================"

# --- reviewer round 3: asymmetry diagnostic, cross-model IC, decision layer
echo "== [R3] asymmetry diagnostic (table31)"
( cd src && python3 experiments/run_asymmetry_diag.py )
echo "== [R3] cross-model IC analysis (table32; needs Wu-repo predictions incl. gpt-4o-mini)"
( cd src/experiments && python3 analyze_crossmodel_ic.py )
echo "== [R3] exception-aware decision layer (table33)"
( cd src && python3 experiments/run_decision_layer.py )

# --- reviewer round 4: statistical hardening + seventh repair
echo "== [P0/P1] train-only majority primary EOR + user-cluster CIs"
( cd src && python3 experiments/run_p0_p1_hardening.py )
echo "== [P0/P1] eight-repair audit from committed JSON"
( cd src && python3 experiments/audit_eight_repairs.py )
echo "== [P0-10] 5 split-seed FM EOR under train-fold majority"
( cd src && python3 experiments/run_eor_fm_seeds.py )
echo "== [R4] m_u robustness, cluster bootstrap, permutation null, symmetry (table34)"
( cd src && python3 experiments/run_review_stats.py )
echo "== [R4] empirical-Bayes partial pooling, seventh repair (table35/36)"
( cd src && python3 experiments/run_eb_pooling.py )
echo "== [R5] exception stability (table38)"
( cd src && python3 experiments/run_exception_stability.py )
echo "== [R6] non-neural backbone GBDT (table40)"
( cd src && python3 experiments/run_gbdt_backbone.py )
echo "== [R7] SPA asymmetry flip test (table41)"
( cd src && PERM_DATA_DIR=../data_spa PERM_RESULTS_DIR=../results_spa python3 experiments/run_asym_spa.py )
echo "== [R7] exception-aware prompt variant (table42; needs Wu repo + API key)"
echo "   drop src/experiments/permission_ic_only_v2.py into wu-repo/src and run it there"
echo "== [R8] kappa-stratified reweighting rescue (table43)"
( cd src && python3 experiments/run_asym_kappa.py )
echo "== [R9] asymmetry confirming intervention (table44)"
( cd src && python3 experiments/run_asym_confirm.py )
echo "== [R9] residual decomposition by precedent (table45)"
( cd src && python3 experiments/run_residual_decomp.py )
echo "== [R10] novelty-gated deferral comparison (table46)"
( cd src && python3 experiments/run_novelty_defer.py )

echo "== [R11] synthetic control, three estimator regimes (table47)"
( cd src && python3 experiments/run_synth_regimes.py )
echo "== [R11] weight-decay x epochs sweep (table49)"
( cd src && python3 experiments/run_regsweep.py )
echo "== [R11] support-k bins, Prediction 1 (table50)"
( cd src && python3 experiments/run_support_k.py )
echo "== [R12] deferral on exceptions: margin vs history rule (table51)"
( cd src && python3 experiments/run_deferral_exc.py )
echo "== [R12] signed-model margin deferral (table55)"
( cd src && python3 experiments/run_signed_defer.py )
echo "== [R13] two-priors 2x2 with Wilson CIs (table52; needs Wu prediction files)"
( cd src && python3 experiments/run_two_priors.py )
echo "== [R13] deployed EOR table (table53; needs Wu prediction files)"
( cd src && python3 experiments/run_eor_deployed.py )
echo "== [R14] SPA symmetric 2x2 (table54)"
( cd src && PERM_DATA_DIR=../data_spa PERM_RESULTS_DIR=../results_spa python3 experiments/run_spa_symmetric.py )
echo "== [R14] SPA subsampled to Wu sparsity (table48)"
( cd src && PERM_DATA_DIR=../data_spa PERM_RESULTS_DIR=../results_spa PERM_MODEL=atomic python3 experiments/run_spa_subsample.py )

echo "== [R15] remaining paper-cited experiments"
( cd src && python3 experiments/run_denial_ablation.py )      # table24: denial ladder (Fig 4 / Tab 5)
( cd src && python3 experiments/run_is_it_shrinkage.py )      # table26: cost-sensitive rows
( cd src && python3 experiments/run_why.py )                  # table27: variance / MLP / synthetic det-stoch
( cd src && python3 experiments/run_rare_rule.py )            # table28: rare-conjunction rho
( cd src && python3 experiments/run_peruser_weight.py )       # reweighting three rows (Tab 14)
( cd src && python3 experiments/run_remedy_who.py )           # table25: per-user thresholds by user bin
( cd src && python3 experiments/run_exception_memory.py )     # exception-memory repair row
( cd src && python3 experiments/run_prior_surprise.py )       # prior-offset / surprise-weighting rows
( cd src && python3 experiments/run_dissonance.py )           # dual-channel confidences (Appendix D)
( cd src && python3 experiments/run_tristate.py )             # opinion-triangle deferral tie
( cd src && python3 experiments/run_crowd_approval.py )       # idiosyncrasy rho = -0.306
( cd src && python3 experiments/run_noise_checks.py )
( cd src && python3 experiments/analyze_demoorder.py )

echo "== [R16] two-prior controls, intervention, layer curve (tables 56-58)"
( cd src && python3 experiments/run_two_prior_regression.py )
( cd src && python3 experiments/run_history_flip.py )
( cd src && python3 experiments/run_layer_curve.py )

echo "== [R17] macro-user EOR ratio vs macro-level permutation null (table74)"
( cd src && python3 experiments/run_macro_null.py )

echo "== [figures] manuscript PDFs from reproduced JSON (journal-manuscript/fig-*.pdf)"
( cd src && python3 experiments/generate_paper_figures.py )
