"""环境自检:验证 PyTorch / CUDA / transformers / 模型可加载。

用法:
    python verify_env.py
"""
from __future__ import annotations

import sys


def main() -> int:
    print("=" * 60)
    print("环境自检")
    print("=" * 60)

    try:
        import torch
        print(f"[OK] torch              : {torch.__version__}")
        print(f"     CUDA available     : {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"     CUDA device        : {torch.cuda.get_device_name(0)}")
            print(f"     CUDA capability    : {torch.cuda.get_device_capability(0)}")
    except ImportError as e:
        print(f"[FAIL] torch import error: {e}")
        return 1

    try:
        import transformers
        print(f"[OK] transformers       : {transformers.__version__}")
    except ImportError as e:
        print(f"[FAIL] transformers import error: {e}")
        return 1

    try:
        import datasets
        print(f"[OK] datasets           : {datasets.__version__}")
    except ImportError as e:
        print(f"[WARN] datasets import error (PPL 评测需要): {e}")

    print("-" * 60)
    print("尝试加载 Pythia-70M ...")
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_name = "EleutherAI/pythia-70m"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype).to(device)
        model.eval()

        n_params = sum(p.numel() for p in model.parameters())
        print(f"[OK] 模型加载成功:{model_name}")
        print(f"     参数量             : {n_params/1e6:.2f} M")
        print(f"     设备 / dtype       : {device} / {dtype}")
        print(f"     vocab_size         : {tokenizer.vocab_size}")

        # 跑一次最小 forward,确认能用
        ids = tokenizer("Hello world", return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            out = model(ids)
        print(f"     test forward logits: {tuple(out.logits.shape)}")
    except Exception as e:
        print(f"[FAIL] 模型加载失败:{e}")
        return 1

    print("=" * 60)
    print("[OK] 环境就绪")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
