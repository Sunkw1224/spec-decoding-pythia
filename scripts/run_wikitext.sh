#!/usr/bin/env bash
# 在 WikiText-2 上跑 baseline vs PLD 测速。
# 从 test split 切 6 段 × 64 token 当 prompt,扫三档输出长度。
set -euo pipefail
cd "$(dirname "$0")/.."

mkdir -p results

for L in 64 128 256 512; do
    echo
    echo "============================================================"
    echo "WikiText-2  max_new_tokens = $L"
    echo "============================================================"
    python -m src.benchmark \
        --dataset wikitext \
        --subset wikitext-2-raw-v1 \
        --split test \
        --num-prompts 6 \
        --prompt-tokens 64 \
        --prompt-offset 256 \
        --max-new-tokens $L \
        --K 5 --max-ngram-size 3 \
        --n-runs 3 \
        --output-csv "results/wikitext_len${L}.csv" \
        | tee "results/wikitext_len${L}.log"
done

echo
echo "Done. CSV 落在 results/wikitext_len*.csv。"
