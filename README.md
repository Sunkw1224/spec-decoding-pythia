# spec-decoding-pythia

个人作业:**Pythia-70M + 无训练 Speculative Decoding**(Prompt Lookup Decoding,PLD)。

## 仓库结构

```
spec-decoding-pythia/
├── README.md             # 本文件
├── requirements.txt
├── verify_env.py         # 环境自检
├── src/
│   ├── pld.py            # PLD 核心实现
│   ├── baseline.py       # 标准自回归解码基线
│   ├── benchmark.py      # 性能测量(TTFT / TPOT / Throughput)
│   ├── eval_ppl.py       # WikiText PPL 评测
│   ├── data.py           # WikiText 加载 + prompt 切片
│   └── utils.py          # KV cache 工具
├── scripts/
│   ├── run_baseline.sh
│   ├── run_pld.sh
│   ├── run_ablation.sh
│   └── run_wikitext.sh   # WikiText 上的测速一键脚本
├── results/
└── notebooks/
    └── analysis.ipynb    # 三轴消融 + prompt 分类 + 汇总表
```

## 快速开始

```bash
pip install -r requirements.txt
python verify_env.py

# 主要结果:WikiText 上的测速
bash scripts/run_wikitext.sh

# K / max_ngram_size 消融
bash scripts/run_ablation.sh

# 可选:合成 prompt 上的早期对比(prompts/sample.txt)
bash scripts/run_baseline.sh
bash scripts/run_pld.sh
```

## 用 WikiText 跑测速

`benchmark.py` 直接支持 HF `datasets` 加载的 WikiText(与 `eval_ppl.py` 共用 `src/data.py`):

```bash
# 从 wikitext-2-raw-v1 test 切 6 段 × 64 token,生成 256 token
python -m src.benchmark \
    --dataset wikitext --subset wikitext-2-raw-v1 --split test \
    --num-prompts 6 --prompt-tokens 64 --prompt-offset 256 \
    --max-new-tokens 256 --K 3 --max-ngram-size 2 --n-runs 3

# 或一键四档:
bash scripts/run_wikitext.sh
```

参数说明:`--num-prompts` 切几段、`--prompt-tokens` 每段多少 token、
`--prompt-offset` 跳过文本开头的 N 个 token(用来避开 `= = =` 这种 wikitext 标题)。
我们的消融显示 WikiText 上 `K=3, max-ngram-size=2` 是接受率与单步算力的甜点。

## WikiText 上的 PPL 评测

`eval_ppl.py` 在 WikiText-2 test 上做滑窗 PPL,可用来验证模型加载正确
(对比 fp16 / fp32 漂移):

```bash
python -m src.eval_ppl --max-tokens 8192
```
