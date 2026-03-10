#!/usr/bin/env bash
set -euo pipefail

# 汇总 run.log 里的 success rate 并生成 csv/html 表

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT_DIR"

python scripts/collect_success_rates.py
python scripts/json_to_table.py

echo "Wrote: logs/summary_success_rates.json"
echo "Wrote: logs/summary_success_rates.csv"
echo "Wrote: logs/summary_success_rates.html"
