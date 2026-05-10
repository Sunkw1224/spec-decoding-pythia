# spec-decoding-pythia

个人作业:**Pythia-70M + 无训练 Speculative Decoding**(Prompt Lookup Decoding,PLD)。

完整技术操作指南见 [GUIDE.md](GUIDE.md)。

## 仓库结构

```
spec-decoding-pythia/
├── GUIDE.md              # 完整技术指南(原理 / 算法 / 实验设计 / 写作框架)
├── README.md             # 本文件
├── requirements.txt
├── verify_env.py         # 环境自检
├── src/
│   ├── pld.py            # PLD 核心实现
│   ├── baseline.py       # 标准自回归解码基线
│   ├── benchmark.py      # 性能测量(TTFT / TPOT / Throughput)
│   ├── eval_ppl.py       # PPL 评测
│   └── utils.py          # KV cache 工具
├── scripts/
│   ├── run_baseline.sh
│   ├── run_pld.sh
│   └── run_ablation.sh
├── results/
└── notebooks/
    └── analysis.ipynb    # 接受率分析、可视化
```

## 快速开始

```bash
pip install -r requirements.txt
python verify_env.py
bash scripts/run_baseline.sh
bash scripts/run_pld.sh
```

详细步骤、关键工程坑、与小组(PyramidKV / Performers)集成接口、实验设计与论文写作框架,请参阅 [GUIDE.md](GUIDE.md)。
