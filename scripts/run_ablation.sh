#!/usr/bin/env bash
# 消融实验:扫 K(候选数)与 max_ngram_size(查找窗口),固定 max_new_tokens=256。
# 对应论文 Figure 1(K vs 加速比)和 Table 2(n-gram size 影响)。
set -euo pipefail
cd "$(dirname "$0")/.."

mkdir -p results

PROMPT_FILE="prompts/sample.txt"
if [[ -f "$PROMPT_FILE" ]]; then
    PROMPT_FLAG="--prompt-file $PROMPT_FILE"
else
    PROMPT_FLAG=""
fi

# ----------------------------------------------------------------
# 1) K 值扫描:K ∈ {1, 2, 3, 5, 7, 10}
#    K 太小:草稿短,提速有限;K 太大:多花的算力没被接受,反而拖慢
# ----------------------------------------------------------------
echo "[1/2] K sweep"
for K in 1 2 3 5 7 10; do
    echo "  K = $K"
    python -m src.benchmark \
        $PROMPT_FLAG \
        --max-new-tokens 256 \
        --K $K \
        --max-ngram-size 3 \
        --n-runs 3 \
        --output-csv "results/ablation_K${K}.csv" \
        > "results/ablation_K${K}.log"
done

# ----------------------------------------------------------------
# 2) n-gram 大小扫描:max_ngram_size ∈ {1, 2, 3, 4}
#    n=1:单 token 匹配很多但语义弱;n=4:匹配少但精准
# ----------------------------------------------------------------
echo "[2/2] n-gram sweep"
for N in 1 2 3 4; do
    echo "  max_ngram_size = $N"
    python -m src.benchmark \
        $PROMPT_FLAG \
        --max-new-tokens 256 \
        --K 5 \
        --max-ngram-size $N \
        --n-runs 3 \
        --output-csv "results/ablation_ngram${N}.csv" \
        > "results/ablation_ngram${N}.log"
done

echo
echo "Done. 共 10 份 CSV 写到 results/ablation_*.csv"
echo "可视化建议:K vs speedup / acceptance_rate 折线图,见 notebooks/analysis.ipynb"
