#!/usr/bin/env bash
# Pythia-70M 基线质量评估:wikitext-2 全量 PPL(fp16)。
# 这一步不涉及 PLD,纯粹为论文 Experiments 表格里的"加速无损"列提供基准 PPL。
set -euo pipefail
cd "$(dirname "$0")/.."

mkdir -p results

echo "[1/2] WikiText-2 raw fp16 PPL ..."
python -m src.eval_ppl \
    --subset wikitext-2-raw-v1 \
    --max-length 1024 \
    --stride 512 \
    | tee results/baseline_ppl_wikitext2.log

echo
echo "[2/2] 单条 prompt 的 baseline 测速(对照组,后续 PLD 与之对比)"
python -m src.benchmark \
    --max-new-tokens 256 \
    --K 5 \
    --n-runs 3 \
    --output-csv results/baseline_smoke.csv \
    | tee results/baseline_smoke.log

echo
echo "Done. 结果文件:"
echo "  results/baseline_ppl_wikitext2.log"
echo "  results/baseline_smoke.csv"
echo "  results/baseline_smoke.log"
