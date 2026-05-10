"""通用工具:模型加载、KV cache 裁剪、计时器、确定性设置。

PLD 验证失败时必须把 KV cache 回滚到"已接受"长度,否则会污染后续步骤。
本模块的 ``trim_kv_cache`` 是核心工具。
"""
from __future__ import annotations

import contextlib
import os
import random
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# ---------------------------------------------------------------------------
# 模型加载
# ---------------------------------------------------------------------------
def load_model_and_tokenizer(
    model_name: str = "EleutherAI/pythia-70m",
    device: Optional[str] = None,
    dtype: Optional[torch.dtype] = None,
):
    """加载 Pythia 模型与 tokenizer。

    Args:
        model_name: HuggingFace 模型 ID。
        device: ``"cuda"`` / ``"cpu"``。``None`` 时自动选择。
        dtype: 权重精度。GPU 默认 fp16,CPU 默认 fp32。
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if dtype is None:
        dtype = torch.float16 if device == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # transformers 4.45+ 改名:torch_dtype -> dtype
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype).to(device)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype).to(device)
    model.eval()
    return model, tokenizer, device, dtype

# ---------------------------------------------------------------------------
# KV cache 裁剪
# ---------------------------------------------------------------------------
def kv_cache_length(past_key_values) -> int:
    """返回 past_key_values 当前的序列长度(time 维)。"""
    if past_key_values is None:
        return 0
    if hasattr(past_key_values, "get_seq_length"):
        return past_key_values.get_seq_length()
    # legacy tuple format: ((k, v), ...) 每个张量形状 [B, H, T, D]
    return past_key_values[0][0].shape[-2]


def trim_kv_cache(past_key_values, target_length: int):
    """把 KV cache 截短到 ``target_length`` 个 token。

    PLD 验证失败时:候选输入了 K 个 token,但只接受了 n 个,需要把 cache
    回滚到 ``prompt_len + n`` 的长度。
    """
    if past_key_values is None or target_length < 0:
        return past_key_values

    cur = kv_cache_length(past_key_values)
    if cur <= target_length:
        return past_key_values

    # 新版 transformers 使用 DynamicCache
    if hasattr(past_key_values, "crop"):
        past_key_values.crop(target_length)
        return past_key_values

    # legacy tuple-of-tuples
    trimmed = tuple(
        (k[..., :target_length, :], v[..., :target_length, :])
        for (k, v) in past_key_values
    )
    return trimmed


# ---------------------------------------------------------------------------
# 计时器
# ---------------------------------------------------------------------------
@dataclass
class Timer:
    """同步计时器(GPU 上必须 sync 才能拿到准确时间)。"""

    device: str = "cuda"
    _start: float = 0.0
    elapsed: float = 0.0

    def __enter__(self) -> "Timer":
        self._sync()
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._sync()
        self.elapsed = time.perf_counter() - self._start

    def _sync(self) -> None:
        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()


@contextlib.contextmanager
def cuda_sync_timer(device: str = "cuda"):
    t = Timer(device=device)
    with t:
        yield t


# ---------------------------------------------------------------------------
# 确定性
# ---------------------------------------------------------------------------
def set_seed(seed: int = 42) -> None:
    """固定随机种子,保证实验可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


# ---------------------------------------------------------------------------
# 调试辅助
# ---------------------------------------------------------------------------
def describe_tensor(name: str, t: torch.Tensor) -> str:
    return f"{name}: shape={tuple(t.shape)} dtype={t.dtype} device={t.device}"
