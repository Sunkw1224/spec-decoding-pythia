"""Prompt Lookup Decoding(PLD)实现。

无 draft 模型版的 speculative decoding:从已有上下文中通过 n-gram 匹配
直接"借用"未来的 token 作为草稿,再让 target 一次性并行验证。

核心不变量
----------
每次进入 while 循环前:
  generated.length == kv_cache_length(past_kv) + 1
即"最后一个 token 的 KV 还没进 cache,它就是下一次 forward 的 anchor"。

每一步流程
----------
1. 在 ``generated`` 中查找最近 n-gram 的历史出现位置,取其后的 K 个 token 作 draft;
2. forward([anchor, c_1, ..., c_K]),拿到长度 1+K 的 logits;
3. greedy 比对 ``argmax(logits[i]) == c_{i+1}``,数出最长匹配前缀长度 n;
4. bonus = ``argmax(logits[n])``(target 自己的预测,替代第 n+1 个候选 / 或是全接受时的额外 token);
5. 把 cache 回滚到 "原长 + 1 + n" 的位置,接受 n 个候选 + 1 个 bonus,合计 n+1 个新 token;
6. anchor 切换为 bonus,进入下一轮。

最坏情况(n=0):一步只前进 1 个 token,且多算了 K 个候选位 —— 慢于 baseline。
最好情况(n=K):一步前进 K+1 个 token,且只跑了 1 次 forward —— 加速 ≈ K+1。
"""
from __future__ import annotations

# 允许直接 `python src/pld.py` 或 `python pld.py` 运行(脚本式调试用)。
# 推荐:从项目根目录 `python -m src.pld`。
if __name__ == "__main__" and (__package__ is None or __package__ == ""):
    import pathlib
    import sys
    _ROOT = pathlib.Path(__file__).resolve().parents[1]
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))
    __package__ = "src"

from typing import List, Optional

import torch

from .baseline import GenerationResult
from .utils import cuda_sync_timer, kv_cache_length, trim_kv_cache


# ---------------------------------------------------------------------------
# 草稿生成:n-gram lookup
# ---------------------------------------------------------------------------
@torch.no_grad()
def find_candidate_pred_tokens(
    input_ids: torch.Tensor,
    max_ngram_size: int = 3,
    num_pred_tokens: int = 10,
) -> torch.Tensor:
    """在 ``input_ids`` 中查找最近的 n-gram 重复,返回其后的 ``num_pred_tokens`` 个 token。

    搜索策略(参考 apoorvumang/prompt-lookup-decoding):
    - 从大到小枚举 ngram_size:max_ngram_size, max_ngram_size-1, ..., 1
    - 取 ``input_ids`` 的末尾 ngram_size 个 token 作为查询
    - 在前缀中**从右向左**搜索匹配位置(优先取最近的匹配,语义更相关)
    - 一旦找到,返回匹配位置后紧跟的 ``num_pred_tokens`` 个 token

    Returns:
        ``[n]`` 一维张量,n ≤ num_pred_tokens;无匹配时返回空张量。
    """
    assert input_ids.dim() == 2 and input_ids.size(0) == 1
    input_length = input_ids.size(1)
    device = input_ids.device

    for ngram_size in range(max_ngram_size, 0, -1):
        if input_length <= ngram_size:
            continue

        ngram = input_ids[0, -ngram_size:].tolist()

        # 从右向左扫描,跳过末尾自身位置(否则会"自己匹配自己")
        for i in range(input_length - ngram_size - 1, -1, -1):
            if input_ids[0, i:i + ngram_size].tolist() == ngram:
                start = i + ngram_size
                end = min(start + num_pred_tokens, input_length)
                if end > start:
                    return input_ids[0, start:end]

    return torch.empty(0, dtype=torch.long, device=device)


# ---------------------------------------------------------------------------
# 主循环
# ---------------------------------------------------------------------------
@torch.no_grad()
def pld_generate(
    model,
    input_ids: torch.Tensor,
    max_new_tokens: int = 128,
    K: int = 5,
    max_ngram_size: int = 3,
    eos_token_id: Optional[int] = None,
    device: str = "cuda",
) -> GenerationResult:
    """Prompt Lookup speculative decoding(greedy 模式)。

    Args:
        model: HuggingFace CausalLM。
        input_ids: ``[1, L]`` 的 prompt token。
        max_new_tokens: 最多新生成多少 token(超出会被裁掉)。
        K: 每步最多提议的候选 token 数(num_pred_tokens)。
        max_ngram_size: n-gram 查找的最大 n。
        eos_token_id: 命中即停止;``None`` 表示忽略。

    Returns:
        ``GenerationResult``,其中 ``n_proposed`` / ``n_accepted`` 反映接受率。
    """
    assert input_ids.dim() == 2 and input_ids.size(0) == 1, "仅支持 batch=1"
    model.eval()

    prompt_len = input_ids.size(1)
    generated = input_ids
    past_kv = None
    per_step: List[float] = []
    n_proposed_total = 0
    n_accepted_total = 0
    n_steps = 0
    eos_hit = False

    with cuda_sync_timer(device) as total_timer:
        # ---- prefill:跑完整 prompt,采第一个新 token 作为 anchor ----
        with cuda_sync_timer(device) as step_timer:
            out = model(generated, use_cache=True)
            past_kv = out.past_key_values                       # cache_len = prompt_len
            anchor = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, anchor], dim=-1)  # len = prompt_len + 1
        per_step.append(step_timer.elapsed)
        n_steps += 1

        if eos_token_id is not None and anchor.item() == eos_token_id:
            return _build_result(generated, prompt_len, total_timer.elapsed,
                                 per_step, n_proposed_total, n_accepted_total)

        # ---- speculative 主循环 ----
        while generated.size(1) - prompt_len < max_new_tokens:
            # 1) 草稿:从已生成内容里查 n-gram
            candidates = find_candidate_pred_tokens(
                generated,
                max_ngram_size=max_ngram_size,
                num_pred_tokens=K,
            )
            num_cand = int(candidates.numel())

            # 2) 构造 forward 输入:anchor 之后再拼 K 个候选
            if num_cand == 0:
                fwd_input = anchor                               # [1, 1]
            else:
                fwd_input = torch.cat(
                    [anchor, candidates.unsqueeze(0)], dim=-1
                )                                                # [1, 1 + num_cand]

            cache_len_before = kv_cache_length(past_kv)

            # 3) 一次 forward 验证全部候选
            with cuda_sync_timer(device) as step_timer:
                out = model(fwd_input, past_key_values=past_kv, use_cache=True)
                past_kv = out.past_key_values                    # +1+num_cand
                logits = out.logits                              # [1, 1+num_cand, V]
            per_step.append(step_timer.elapsed)
            n_steps += 1

            target_preds = logits[0].argmax(dim=-1)              # [1 + num_cand]

            # 4) 验证 + 5) bonus
            if num_cand == 0:
                n_accepted = 0
                bonus = target_preds[0:1].unsqueeze(0)           # [1, 1]
                # cache 长度刚好 +1,无需 trim
            else:
                # target_preds[i] (i<num_cand) 是给定 anchor + 前 i 个候选时,
                # 对"第 i+1 个 token"的预测;与 candidates[i] 比较即可。
                matches = (target_preds[:num_cand] == candidates).int()
                # cumprod:第一个失配位置之后全部清零;sum 即最长匹配前缀长度
                n_accepted = int(matches.cumprod(dim=0).sum().item())

                bonus = target_preds[n_accepted:n_accepted + 1].unsqueeze(0)

                # 把 cache 回滚到 "anchor + 前 n_accepted 个候选"
                target_cache_len = cache_len_before + 1 + n_accepted
                past_kv = trim_kv_cache(past_kv, target_cache_len)

            # 6) 落盘:接受的候选 + bonus
            if n_accepted > 0:
                accepted = candidates[:n_accepted].unsqueeze(0)
                generated = torch.cat([generated, accepted, bonus], dim=-1)
            else:
                generated = torch.cat([generated, bonus], dim=-1)

            # 7) 切换 anchor
            anchor = bonus

            n_proposed_total += num_cand
            n_accepted_total += n_accepted

            # 8) EOS 检测(只看本轮新增的 n_accepted+1 个 token)
            if eos_token_id is not None:
                new_tokens = generated[0, -(n_accepted + 1):].tolist()
                if eos_token_id in new_tokens:
                    eos_hit = True
                    break

    # ---- 截断到精确的 max_new_tokens ----
    final = generated[:, : prompt_len + max_new_tokens]
    if eos_hit:
        # 保留到 EOS 为止(不强制截到 max_new_tokens)
        final = generated

    return _build_result(
        final, prompt_len, total_timer.elapsed,
        per_step, n_proposed_total, n_accepted_total,
    )


# ---------------------------------------------------------------------------
# 结果包装
# ---------------------------------------------------------------------------
def _build_result(
    generated: torch.Tensor,
    prompt_len: int,
    elapsed: float,
    per_step: List[float],
    n_proposed: int,
    n_accepted: int,
) -> GenerationResult:
    return GenerationResult(
        output_ids=generated,
        n_generated=generated.size(1) - prompt_len,
        elapsed=elapsed,
        n_steps=len(per_step),
        per_step_times=per_step,
        n_proposed=n_proposed,
        n_accepted=n_accepted,
    )


# ---------------------------------------------------------------------------
# 自检:greedy 下 PLD 输出必须等于 baseline
# ---------------------------------------------------------------------------
def _self_check() -> None:
    """简单冒烟:加载 Pythia-70M,确认 PLD 与 baseline 输出 token-by-token 一致。"""
    from .baseline import greedy_generate
    from .utils import load_model_and_tokenizer, set_seed

    set_seed(42)
    model, tokenizer, device, _ = load_model_and_tokenizer()

    prompt = (
        "The quick brown fox jumps over the lazy dog. "
        "The quick brown fox jumps over the lazy "
    )
    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    base = greedy_generate(model, ids, max_new_tokens=32, device=device)
    pld = pld_generate(model, ids, max_new_tokens=32, K=5, max_ngram_size=3, device=device)

    same = torch.equal(base.output_ids, pld.output_ids)
    print(f"baseline tokens : {tokenizer.decode(base.output_ids[0])!r}")
    print(f"pld      tokens : {tokenizer.decode(pld.output_ids[0])!r}")
    print(f"identical       : {same}")
    print(f"baseline TPOT   : {base.tpot * 1000:.2f} ms/token")
    print(f"pld      TPOT   : {pld.tpot * 1000:.2f} ms/token")
    print(f"speedup         : {base.tpot / pld.tpot:.2f}x")
    print(f"acceptance rate : {pld.acceptance_rate:.2%}  ({pld.n_accepted}/{pld.n_proposed})")
    assert same, "PLD 输出必须与 baseline greedy 完全一致!"


if __name__ == "__main__":
    _self_check()
