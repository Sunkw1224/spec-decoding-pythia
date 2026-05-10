"""标准自回归 greedy decoding 基线。

故意**不用** ``model.generate``,而是手写循环 + KV cache,保证:
1. 与 PLD 走相同的 forward 路径,公平对比;
2. 输出 token 序列可被严格断言为"无损参考";
3. 可观测每步耗时(供 benchmark.py 复用)。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import torch

from .utils import cuda_sync_timer, kv_cache_length


@dataclass
class GenerationResult:
    """统一的生成结果容器,baseline 与 PLD 共用。"""

    output_ids: torch.Tensor          # [1, prompt_len + n_new]
    n_generated: int                   # 新生成的 token 数
    elapsed: float                     # 总耗时(秒)
    n_steps: int                       # 调用 forward 的次数
    per_step_times: List[float] = field(default_factory=list)
    # PLD 专用,baseline 留空
    n_proposed: int = 0
    n_accepted: int = 0

    @property
    def tpot(self) -> float:
        """time per output token(秒/token)。"""
        return self.elapsed / max(self.n_generated, 1)

    @property
    def throughput(self) -> float:
        """tokens / second。"""
        return self.n_generated / max(self.elapsed, 1e-9)

    @property
    def acceptance_rate(self) -> float:
        if self.n_proposed == 0:
            return 0.0
        return self.n_accepted / self.n_proposed


@torch.no_grad()
def greedy_generate(
    model,
    input_ids: torch.Tensor,
    max_new_tokens: int = 128,
    eos_token_id: Optional[int] = None,
    device: str = "cuda",
) -> GenerationResult:
    """手写 greedy 自回归解码,带 KV cache 复用。

    Args:
        model: HuggingFace CausalLM。
        input_ids: ``[1, L]`` 的 prompt token。
        max_new_tokens: 最多生成多少新 token。
        eos_token_id: 命中即停止;``None`` 表示忽略。

    Returns:
        ``GenerationResult``,其中 ``output_ids`` 含原 prompt + 新 token。
    """
    assert input_ids.dim() == 2 and input_ids.size(0) == 1, "仅支持 batch=1"

    model.eval()
    prompt_len = input_ids.size(1)
    generated = input_ids
    past_kv = None
    per_step: List[float] = []

    with cuda_sync_timer(device) as total_timer:
        # ---- 第 1 步:跑完整 prompt,拿到 past_kv ----
        with cuda_sync_timer(device) as step_timer:
            out = model(generated, use_cache=True)
            past_kv = out.past_key_values
            next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=-1)
        per_step.append(step_timer.elapsed)

        if eos_token_id is not None and next_token.item() == eos_token_id:
            return _wrap_result(generated, prompt_len, total_timer.elapsed, per_step)

        # ---- 后续步:每次只 forward 1 个新 token ----
        for _ in range(max_new_tokens - 1):
            with cuda_sync_timer(device) as step_timer:
                out = model(next_token, past_key_values=past_kv, use_cache=True)
                past_kv = out.past_key_values
                next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                generated = torch.cat([generated, next_token], dim=-1)
            per_step.append(step_timer.elapsed)

            if eos_token_id is not None and next_token.item() == eos_token_id:
                break

    return _wrap_result(generated, prompt_len, total_timer.elapsed, per_step)


def _wrap_result(
    generated: torch.Tensor,
    prompt_len: int,
    elapsed: float,
    per_step: List[float],
) -> GenerationResult:
    return GenerationResult(
        output_ids=generated,
        n_generated=generated.size(1) - prompt_len,
        elapsed=elapsed,
        n_steps=len(per_step),
        per_step_times=per_step,
    )
