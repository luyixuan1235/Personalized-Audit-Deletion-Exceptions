#!/usr/bin/env bash
# Set up both environments needed to reproduce the paper.
#
#   bash setup.sh          # our experiments only  (torch; fast)
#   bash setup.sh --wu     # also the Wu et al. reproduction (tensorflow; ~2.5 GB)
#
# Two separate virtualenvs are created on purpose: our code needs torch, and the
# authors' released code needs tensorflow + recommenders==1.2.1 pinned to numpy<2.
# Installing both into one env leads to dependency conflicts.
set -euo pipefail
cd "$(dirname "$0")"
if [ -n "${PYTHON:-}" ]; then
  PY="$PYTHON"
elif [ -x /c/miniconda3/python.exe ]; then
  # Git Bash may expose an unrelated MSYS python3; use the Windows interpreter.
  PY=/c/miniconda3/python.exe
else
  PY="${PYTHON:-python3}"
fi

echo "==> [1/2] environment for our experiments  (.venv)"
"$PY" -m venv .venv
# Windows venvs activate from Scripts/; POSIX venvs use bin/.
if [ -f .venv/Scripts/activate ]; then
  # shellcheck disable=SC1091
  source .venv/Scripts/activate
elif [ -f .venv/bin/activate ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
else
  echo "No activation script found in .venv"; exit 1
fi
python -m pip install -q --upgrade pip
# Torch 2.9+ default wheels hit WinError 1114 on some Windows machines.
python -m pip install -q --no-cache-dir "numpy<2.3" \
  "torch==2.8.0" --index-url https://download.pytorch.org/whl/cpu
python -c "import torch, numpy; print('    torch', torch.__version__, '| numpy', numpy.__version__)"
deactivate

if [ "${1:-}" != "--wu" ]; then
  echo
  echo "Done. Run:  bash run_paper.sh"
  echo "(For the Wu et al. reproduction as well, re-run: bash setup.sh --wu)"
  exit 0
fi

echo
echo "==> [2/2] environment for the Wu et al. released code  (wu-repo/.venv)"
echo "    NOTE: needs ~2.5 GB free disk for tensorflow."
if ! command -v git >/dev/null; then echo "git is required"; exit 1; fi
[ -d wu-repo ] || git clone --depth 1 -q \
  https://github.com/llm-platform-security/ai-agent-permissions.git wu-repo

cp src/experiments/analyze_wu_peruser.py wu-repo/src/
cd wu-repo
"$PY" -m venv .venv
# Windows venvs activate from Scripts/; POSIX venvs use bin/.
if [ -f .venv/Scripts/activate ]; then
  # shellcheck disable=SC1091
  source .venv/Scripts/activate
elif [ -f .venv/bin/activate ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
else
  echo "No activation script found in wu-repo/.venv"; exit 1
fi
python -m pip install -q --upgrade pip
# numpy<2 is required by recommenders 1.2.1
python -m pip install -q --no-cache-dir "numpy<2" pandas scikit-learn tensorflow-cpu \
    "recommenders==1.2.1" openai python-dotenv tqdm tenacity
python - <<'EOF'
import tensorflow as tf
from recommenders.models.deeprec.models.graphrec.lightgcn import LightGCN  # noqa: F401
print("    tensorflow", tf.__version__, "| LightGCN import OK")
EOF
deactivate
cd ..

echo
echo "Done. Run:  bash run_paper.sh --wu"
echo "To include the LLM hybrid, put your key in wu-repo/.env first:"
echo "    echo 'OPENAI_API_KEY=sk-...' > wu-repo/.env"
