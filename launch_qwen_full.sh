#!/usr/bin/env bash
# Full Qwen runs: Wu anchor (181) then SPA (1737), 8 workers. Started by scheduler.
set -e
cd "$(dirname "$0")/src/experiments"
echo "=== start $(date -u) ==="
python3 qwen_ic_runner.py wu --workers 8
python3 qwen_ic_runner.py spa --workers 8
echo "=== analysis $(date -u) ==="
python3 analyze_qwen_ic.py
echo "=== done $(date -u) ==="

