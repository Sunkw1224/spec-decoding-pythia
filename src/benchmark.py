"""加速比基准测试。

对一组 prompt × 方法(baseline / PLD)做 ``warmup + N 次测量``,汇总:
  - TTFT(time to first token,首 token 耗时)
  - TPOT(time per output token,平均每 token 耗时)
  - Throughput(tokens / second)
  - Acceptance rate(仅 PLD)
  - Speedup(PLD.TPOT / baseline.TPOT 的倒数)

CLI 用法
--------
对一个 prompt 跑完整对比:
    python -m src.benchmark --prompt-file prompts/sample.txt \\
        --max-new-tokens 256 --K 5 --n-runs 3

也可以把 ``--prompt`` 直接写死:
    python -m src.benchmark --prompt "The quick brown fox ..." --max-new-tokens 128
"""
from __future__ import annotations

if __name__ == "__main__" and (__package__ is None or __package__ == ""):
    import pathlib
    import sys
    _ROOT = pathlib.Path(__file__).resolve().parents[1]
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))
    __package__ = "src"

import argparse
import csv
import json
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

import torch

from .baseline import GenerationResult, greedy_generate
from .data import load_wikitext_text, slice_prompts
from .pld import pld_generate
from .utils import load_model_and_tokenizer, set_seed


# ---------------------------------------------------------------------------
# 单次跑分
# ---------------------------------------------------------------------------
@dataclass
class RunStats:
    """单个方法在单条 prompt 上、跑 N 次的统计结果。"""

    method: str
    K: int
    prompt_len: int
    n_generated: int
    # 多次测量
    ttft_ms: List[float] = field(default_factory=list)            # 首 token 耗时(ms)
    tpot_ms: List[float] = field(default_factory=list)            # 平均每 token 耗时(ms)
    throughput: List[float] = field(default_factory=list)         # tokens/s
    acceptance_rates: List[float] = field(default_factory=list)   # 仅 PLD

    def add(self, r: GenerationResult) -> None:
        ttft = r.per_step_times[0] * 1000.0 if r.per_step_times else 0.0
        self.ttft_ms.append(ttft)
        self.tpot_ms.append(r.tpot * 1000.0)
        self.throughput.append(r.throughput)
        if self.method.startswith("pld"):
            self.acceptance_rates.append(r.acceptance_rate)

    def summary(self) -> Dict[str, float]:
        def _stat(xs: List[float]) -> Dict[str, float]:
            if not xs:
                return {"mean": 0.0, "std": 0.0, "min": 0.0}
            return {
                "mean": statistics.mean(xs),
                "std": statistics.pstdev(xs) if len(xs) > 1 else 0.0,
                "min": min(xs),
            }

        out = {
            "method": self.method,
            "K": self.K,
            "prompt_len": self.prompt_len,
            "n_generated": self.n_generated,
            "ttft_ms": _stat(self.ttft_ms),
            "tpot_ms": _stat(self.tpot_ms),
            "throughput": _stat(self.throughput),
        }
        if self.acceptance_rates:
            out["acceptance_rate"] = _stat(self.acceptance_rates)
        return out


# ---------------------------------------------------------------------------
# 跑分主流程
# ---------------------------------------------------------------------------
def run_method(
    name: str,
    fn: Callable[[], GenerationResult],
    n_warmup: int = 1,
    n_runs: int = 3,
    K: int = 0,
    prompt_len: int = 0,
) -> RunStats:
    """对一个无参可调用 ``fn`` 做 warmup + n_runs 次测量。"""
    # warmup —— 触发 CUDA 上下文 / autotune,丢弃首跑
    for _ in range(n_warmup):
        _ = fn()

    stats = RunStats(method=name, K=K, prompt_len=prompt_len, n_generated=0)
    last_result: Optional[GenerationResult] = None
    for _ in range(n_runs):
        r = fn()
        stats.add(r)
        last_result = r

    if last_result is not None:
        stats.n_generated = last_result.n_generated
    return stats


def benchmark_prompt(
    model,
    tokenizer,
    prompt_text: str,
    max_new_tokens: int,
    K: int,
    max_ngram_size: int,
    n_warmup: int,
    n_runs: int,
    device: str,
    eos_token_id: Optional[int] = None,
) -> Dict[str, RunStats]:
    """对一条 prompt 跑 baseline 与 PLD,返回 {方法名: RunStats}。"""
    input_ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to(device)
    prompt_len = input_ids.size(1)

    def _baseline() -> GenerationResult:
        return greedy_generate(
            model, input_ids, max_new_tokens=max_new_tokens,
            eos_token_id=eos_token_id, device=device,
        )

    def _pld() -> GenerationResult:
        return pld_generate(
            model, input_ids, max_new_tokens=max_new_tokens,
            K=K, max_ngram_size=max_ngram_size,
            eos_token_id=eos_token_id, device=device,
        )

    print(f"\n[prompt] len={prompt_len} chars={len(prompt_text)}")
    print("  >>", prompt_text[:80].replace("\n", " "), "...")

    base = run_method("baseline", _baseline, n_warmup=n_warmup, n_runs=n_runs,
                       K=0, prompt_len=prompt_len)
    pld = run_method(f"pld_K{K}", _pld, n_warmup=n_warmup, n_runs=n_runs,
                      K=K, prompt_len=prompt_len)
    return {"baseline": base, f"pld_K{K}": pld}


def format_table(results: List[Dict[str, RunStats]]) -> str:
    """把多条 prompt 的结果汇成一张文本表。"""
    rows = []
    header = f"{'method':<10} {'K':>3} {'plen':>5} {'ngen':>5} " \
             f"{'TTFT(ms)':>10} {'TPOT(ms)':>10} {'tok/s':>9} " \
             f"{'accept':>8} {'speedup':>8}"
    rows.append(header)
    rows.append("-" * len(header))

    for entry in results:
        base = entry["baseline"].summary()
        base_tpot = base["tpot_ms"]["mean"]
        for key, st in entry.items():
            s = st.summary()
            tpot = s["tpot_ms"]["mean"]
            speedup = base_tpot / tpot if tpot > 0 else 0.0
            acc = s.get("acceptance_rate", {}).get("mean", 0.0) * 100 if "acceptance_rate" in s else 0.0
            rows.append(
                f"{s['method']:<10} {s['K']:>3} {s['prompt_len']:>5} "
                f"{s['n_generated']:>5} "
                f"{s['ttft_ms']['mean']:>10.2f} {tpot:>10.2f} "
                f"{s['throughput']['mean']:>9.1f} "
                f"{acc:>7.1f}% {speedup:>7.2f}x"
            )
    return "\n".join(rows)


def save_csv(results: List[Dict[str, RunStats]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["prompt_idx", "method", "K", "prompt_len", "n_generated",
                    "ttft_mean_ms", "ttft_std_ms",
                    "tpot_mean_ms", "tpot_std_ms",
                    "throughput_mean", "throughput_std",
                    "acceptance_mean"])
        for idx, entry in enumerate(results):
            for st in entry.values():
                s = st.summary()
                acc = s.get("acceptance_rate", {}).get("mean", 0.0)
                w.writerow([
                    idx, s["method"], s["K"], s["prompt_len"], s["n_generated"],
                    f"{s['ttft_ms']['mean']:.4f}", f"{s['ttft_ms']['std']:.4f}",
                    f"{s['tpot_ms']['mean']:.4f}", f"{s['tpot_ms']['std']:.4f}",
                    f"{s['throughput']['mean']:.4f}", f"{s['throughput']['std']:.4f}",
                    f"{acc:.6f}",
                ])


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _load_prompts(args: argparse.Namespace, tokenizer=None) -> List[str]:
    """三种来源,优先级:--dataset wikitext > --prompt-file > --prompt > 内置默认。"""
    if args.dataset == "wikitext":
        assert tokenizer is not None, "wikitext 模式需要先加载 tokenizer"
        print(f"[data] wikitext/{args.subset}/{args.split} "
              f"→ {args.num_prompts} 段 × {args.prompt_tokens} token")
        text = load_wikitext_text(args.subset, args.split)
        return slice_prompts(
            text, tokenizer,
            n_prompts=args.num_prompts,
            tokens_per_prompt=args.prompt_tokens,
            start_offset_tokens=args.prompt_offset,
        )
    if args.prompt_file:
        text = Path(args.prompt_file).read_text(encoding="utf-8")
        # 以空行分隔多条 prompt
        prompts = [p.strip() for p in text.split("\n\n") if p.strip()]
        return prompts
    if args.prompt:
        return [args.prompt]
    # 默认:一条带重复模式的句子,演示用
    return [
        "The quick brown fox jumps over the lazy dog. "
        "The quick brown fox jumps over the lazy dog. "
        "The quick brown fox jumps over the "
    ]


def main() -> None:
    p = argparse.ArgumentParser(description="PLD speedup benchmark")
    p.add_argument("--model", default="EleutherAI/pythia-70m")
    p.add_argument("--prompt", type=str, default=None, help="单条 prompt 文本")
    p.add_argument("--prompt-file", type=str, default=None, help="prompt 文件,空行分隔多条")
    # WikiText 数据源(与 eval_ppl.py 共用 data.py)
    p.add_argument("--dataset", choices=["none", "wikitext"], default="none")
    p.add_argument("--subset", default="wikitext-2-raw-v1",
                   help="wikitext-2-raw-v1 / wikitext-103-raw-v1")
    p.add_argument("--split", default="test")
    p.add_argument("--num-prompts", type=int, default=6,
                   help="从 WikiText 切多少段做 prompt")
    p.add_argument("--prompt-tokens", type=int, default=64,
                   help="每段 prompt 的 token 数")
    p.add_argument("--prompt-offset", type=int, default=0,
                   help="跳过 text 开头的 N 个 token")
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--K", type=int, default=5, help="每步候选 token 数")
    p.add_argument("--max-ngram-size", type=int, default=3)
    p.add_argument("--n-warmup", type=int, default=1)
    p.add_argument("--n-runs", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-csv", type=str, default="results/benchmark.csv")
    p.add_argument("--output-json", type=str, default=None)
    args = p.parse_args()

    set_seed(args.seed)
    print(f"[load] {args.model}")
    model, tokenizer, device, dtype = load_model_and_tokenizer(args.model)
    print(f"       device={device} dtype={dtype}")

    prompts = _load_prompts(args, tokenizer=tokenizer)
    print(f"[bench] {len(prompts)} prompt(s), K={args.K}, "
          f"max_new={args.max_new_tokens}, n_runs={args.n_runs}")

    results: List[Dict[str, RunStats]] = []
    for text in prompts:
        results.append(benchmark_prompt(
            model, tokenizer, text,
            max_new_tokens=args.max_new_tokens,
            K=args.K,
            max_ngram_size=args.max_ngram_size,
            n_warmup=args.n_warmup,
            n_runs=args.n_runs,
            device=device,
        ))

    print()
    print(format_table(results))

    if args.output_csv:
        out_csv = Path(args.output_csv)
        save_csv(results, out_csv)
        print(f"\n[save] {out_csv}")
    if args.output_json:
        out_json = Path(args.output_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(
            json.dumps([{k: v.summary() for k, v in e.items()} for e in results],
                       ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"[save] {out_json}")


if __name__ == "__main__":
    main()
