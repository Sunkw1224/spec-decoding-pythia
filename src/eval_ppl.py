"""WikiText-103 困惑度(PPL)评测,标准滑窗口实现。

为什么 PLD 不需要单独 PPL?
  greedy 模式下 PLD 输出与 baseline 完全一致(已被 ``pld._self_check`` 断言),
  所以 PLD 的 PPL = baseline PPL。本脚本主要用于:
    1) 验证模型加载正确(对比 fp16 / fp32 的 PPL 漂移);
    2) 论文 Experiments 表格里的 PPL 列(证明加速无损)。

CLI 用法
--------
    python -m src.eval_ppl --max-tokens 8192
    python -m src.eval_ppl --dataset wikitext --subset wikitext-2-raw-v1
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
import math
from typing import Optional

import torch
from tqdm import tqdm

from .data import load_wikitext_text
from .utils import load_model_and_tokenizer, set_seed


@torch.no_grad()
def compute_ppl_sliding(
    model,
    tokenizer,
    text: str,
    max_length: int = 1024,
    stride: int = 512,
    device: str = "cuda",
    max_tokens: Optional[int] = None,
) -> float:
    """滑窗口 PPL,参考 HuggingFace ``transformers`` 文档的标准做法。

    把整段文本一次性 tokenize,然后用长度 ``max_length``、步长 ``stride`` 的窗口
    遍历;每个窗口里只对**新进入**的 ``stride`` 个 token 计 NLL,保证每个 token
    被 NLL 计数恰好一次。

    Args:
        max_length: 窗口长度(应 ≤ 模型最大上下文,Pythia-70m 是 2048)。
        stride: 窗口滑动步长。``stride < max_length`` 给前文留出上下文。
        max_tokens: 上限,None 表示用完整数据。
    """
    model.eval()
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids.to(device)
    seq_len = input_ids.size(1)
    if max_tokens is not None:
        seq_len = min(seq_len, max_tokens)
        input_ids = input_ids[:, :seq_len]

    nlls = []
    prev_end = 0
    pbar = tqdm(range(0, seq_len, stride), desc="ppl", leave=False)
    for begin in pbar:
        end = min(begin + max_length, seq_len)
        trg_len = end - prev_end                            # 这一窗里新增 token 数
        ids = input_ids[:, begin:end]
        labels = ids.clone()
        # 不算"已统计过"的旧 token 的 loss
        labels[:, :-trg_len] = -100

        out = model(ids, labels=labels)
        # HF model 返回的 loss 是按"参与计算的 token 数"做了 mean,这里要乘回来
        nll = out.loss * trg_len
        nlls.append(nll)

        prev_end = end
        if end == seq_len:
            break

    total_nll = torch.stack(nlls).sum()
    ppl = torch.exp(total_nll / seq_len).item()
    return ppl


def main() -> None:
    p = argparse.ArgumentParser(description="WikiText 滑窗 PPL 评测")
    p.add_argument("--model", default="EleutherAI/pythia-70m")
    p.add_argument("--dataset", default="wikitext")
    p.add_argument("--subset", default="wikitext-2-raw-v1",
                   help="wikitext-2-raw-v1 / wikitext-103-raw-v1 等")
    p.add_argument("--split", default="test")
    p.add_argument("--max-length", type=int, default=1024)
    p.add_argument("--stride", type=int, default=512)
    p.add_argument("--max-tokens", type=int, default=None,
                   help="只评测前 N 个 token,None=全量")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    set_seed(args.seed)
    print(f"[load] {args.model}")
    model, tokenizer, device, dtype = load_model_and_tokenizer(args.model)
    print(f"       device={device} dtype={dtype}")

    print(f"[data] {args.dataset}/{args.subset}/{args.split}")
    text = load_wikitext_text(args.subset, args.split)
    print(f"       chars: {len(text):,}")

    ppl = compute_ppl_sliding(
        model, tokenizer, text,
        max_length=args.max_length,
        stride=args.stride,
        device=device,
        max_tokens=args.max_tokens,
    )
    print(f"\n[result] PPL = {ppl:.4f}")
    print(f"         (model={args.model}, dtype={dtype}, max_len={args.max_length}, "
          f"stride={args.stride}, max_tokens={args.max_tokens})")


if __name__ == "__main__":
    main()
