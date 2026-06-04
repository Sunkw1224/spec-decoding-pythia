"""WikiText 数据加载与 prompt 切片(供 benchmark.py 与 eval_ppl.py 共用)。

数据格式
--------
- HF ``datasets.load_dataset("wikitext", subset, split=...)``
- 把所有非空 ``text`` 字段用 "\\n\\n" 串成一整段
- 由调用方决定怎么切(滑窗 PPL / 切 N 段做 prompt)

出口
----
- ``load_wikitext_text(subset, split)``                          -> str
- ``slice_prompts(text, tokenizer, n_prompts, tokens_per_prompt)`` -> List[str]
"""
from __future__ import annotations

from typing import List


def load_wikitext_text(
    subset: str = "wikitext-2-raw-v1",
    split: str = "test",
) -> str:
    """合并 wikitext 的所有非空行成一整段文本。

    Args:
        subset: ``wikitext-2-raw-v1`` / ``wikitext-103-raw-v1``。
        split: ``train`` / ``validation`` / ``test``。
    """
    from datasets import load_dataset

    ds = load_dataset("wikitext", subset, split=split)
    texts = [t for t in ds["text"] if t.strip()]
    return "\n\n".join(texts)


def slice_prompts(
    text: str,
    tokenizer,
    n_prompts: int = 6,
    tokens_per_prompt: int = 64,
    start_offset_tokens: int = 0,
) -> List[str]:
    """从 ``text`` 里 tokenize 后,均匀切出 ``n_prompts`` 段,每段 ``tokens_per_prompt`` 个 token。

    切法:
    - 整段 text 一次 tokenize(不截断)
    - 取 ``[start_offset_tokens, ...]`` 之后的部分
    - 均分 n_prompts 段,每段取前 ``tokens_per_prompt`` 个 token,decode 回字符串

    这样保证 prompt 来自真实自然文本,且每条长度可控。
    """
    enc = tokenizer(text, return_tensors="pt", truncation=False)
    ids = enc.input_ids[0]
    seq_len = ids.size(0)

    if seq_len <= start_offset_tokens + tokens_per_prompt:
        raise ValueError(
            f"text too short: {seq_len} tokens, need at least "
            f"{start_offset_tokens + tokens_per_prompt}"
        )

    usable = seq_len - start_offset_tokens
    step = max(usable // n_prompts, tokens_per_prompt)

    prompts: List[str] = []
    for i in range(n_prompts):
        begin = start_offset_tokens + i * step
        end = begin + tokens_per_prompt
        if end > seq_len:
            break
        chunk_ids = ids[begin:end].tolist()
        prompts.append(tokenizer.decode(chunk_ids, skip_special_tokens=True))
    return prompts
