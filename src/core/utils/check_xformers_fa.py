#!/usr/bin/env python3
"""Quick diagnostic: xformers + FlashAttention with bf16.

Run: python check_xformers_flashattention.py

Checks that:
- xformers is installed and importable
- CUDA is available
- memory_efficient_attention with MemoryEfficientAttentionFlashAttentionOp runs in bf16
"""

import sys

GREEN = "\033[92m"
RED = "\033[91m"
CYAN = "\033[96m"
ENDC = "\033[0m"


def main():
    print(f"{CYAN}{'=' * 60}{ENDC}")
    print(f"{CYAN}xformers + FlashAttention (bf16) diagnostic{ENDC}")
    print(f"{CYAN}{'=' * 60}{ENDC}")

    # 1. xformers
    try:
        import xformers
        import xformers.ops as xops

        print(f"{GREEN}[OK]{ENDC} xformers imported: {xformers.__version__}")
    except ImportError as e:
        print(f"{RED}[FAIL]{ENDC} xformers not available: {e}")
        return 1

    # 2. PyTorch + CUDA
    import torch

    print(f"{GREEN}[OK]{ENDC} PyTorch: {torch.__version__}")
    if not torch.cuda.is_available():
        print(f"{RED}[FAIL]{ENDC} CUDA not available (FlashAttention requires GPU)")
        return 1
    print(f"{GREEN}[OK]{ENDC} CUDA available: {torch.cuda.get_device_name(0)}")

    # 3. FlashAttention op availability
    try:
        _ = xops.MemoryEfficientAttentionFlashAttentionOp
        print(f"{GREEN}[OK]{ENDC} MemoryEfficientAttentionFlashAttentionOp is available")
    except AttributeError as e:
        print(f"{RED}[FAIL]{ENDC} FlashAttention op not found: {e}")
        return 1

    # 4. Run a small attention op in bf16 with explicit FlashAttention
    dtype = torch.bfloat16
    device = "cuda"
    B, N, H, D = 2, 64, 4, 64  # batch, seq_len, heads, head_dim

    print(f"\n{CYAN}Running memory_efficient_attention with FlashAttentionOp in bf16...{ENDC}")
    try:
        q = torch.randn(B, N, H, D, device=device, dtype=dtype)
        k = torch.randn(B, N, H, D, device=device, dtype=dtype)
        v = torch.randn(B, N, H, D, device=device, dtype=dtype)

        out = xops.memory_efficient_attention(
            q,
            k,
            v,
            p=0.0,
            op=xops.MemoryEfficientAttentionFlashAttentionOp,
        )
        assert out.shape == (B, N, H, D), f"Bad shape: {out.shape}"
        assert out.dtype == dtype, f"Bad dtype: {out.dtype}"
        print(
            f"{GREEN}[OK]{ENDC} FlashAttention (bf16) output shape={out.shape}, dtype={out.dtype}"
        )
    except Exception as e:
        print(f"{RED}[FAIL]{ENDC} FlashAttention bf16 run failed: {e}")
        return 1

    # 5. Also run a backward pass
    print(f"{CYAN}Running backward pass (bf16)...{ENDC}")
    try:
        q = torch.randn(B, N, H, D, device=device, dtype=dtype, requires_grad=True)
        k = torch.randn(B, N, H, D, device=device, dtype=dtype, requires_grad=True)
        v = torch.randn(B, N, H, D, device=device, dtype=dtype, requires_grad=True)
        out = xops.memory_efficient_attention(
            q,
            k,
            v,
            p=0.0,
            op=xops.MemoryEfficientAttentionFlashAttentionOp,
        )
        loss = out.sum()
        loss.backward()
        print(f"{GREEN}[OK]{ENDC} Backward pass (bf16) succeeded")
    except Exception as e:
        print(f"{RED}[FAIL]{ENDC} Backward pass failed: {e}")
        return 1

    print(f"\n{CYAN}{'=' * 60}{ENDC}")
    print(f"{GREEN}All checks passed. xformers FlashAttention with bf16 is working.{ENDC}")
    print(f"{CYAN}{'=' * 60}{ENDC}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
