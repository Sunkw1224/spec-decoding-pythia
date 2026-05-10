#!/usr/bin/env bash
# PLD 主实验:固定 K=5、max_ngram_size=3,扫不同 max_new_tokens,
# 验证"输出越长,加速比越大"(指南踩坑清单 §5)。
set -euo pipefail   # pipefail:管道里只要有任何命令失败,整个管道就算失败
cd "$(dirname "$0")/.."

mkdir -p results

PROMPT_FILE="prompts/sample.txt"
if [[ ! -f "$PROMPT_FILE" ]]; then
    echo "[warn] 未找到 $PROMPT_FILE,使用 benchmark 内置默认 prompt"
    PROMPT_FLAG=""
else
    PROMPT_FLAG="--prompt-file $PROMPT_FILE"
fi

for L in 64 128 256 512; do
    echo
    echo "============================================================"
    echo "max_new_tokens = $L"
    echo "============================================================"
    python -m src.benchmark \
        $PROMPT_FLAG \
        --max-new-tokens $L \
        --K 5 \
        --max-ngram-size 3 \
        --n-runs 3 \
        --output-csv "results/pld_len${L}.csv" \
        | tee "results/pld_len${L}.log"
done

echo
echo "Done. 4 份 CSV 写到 results/pld_len*.csv,可由 notebooks/analysis.ipynb 读取。"
