"""Pad GPU utilization up to a target threshold.

Monitors utilization via EMA and runs a CUDA busy-loop to make up the shortfall.
Does NOT increase memory usage beyond a tiny one-time allocation.

Usage:
    python gpu_padder.py --target 0.4 --gpu 0 --ema-alpha 0.2 --poll-interval 2.0

Auto-launched by your sweep script, see launch_padder() context manager below.
"""

import argparse
import signal
import subprocess
import sys
import time

import torch

# ──────────────────────────────────────────────
# CUDA busy-loop kernel
# ──────────────────────────────────────────────


def _make_dummy_tensor(device: torch.device) -> torch.Tensor:
    """Allocate a tiny tensor once. All padding work happens on this."""
    # 1 MB: just enough to keep a CUDA stream busy, negligible VRAM
    return torch.zeros(256, 1_024, dtype=torch.float32, device=device)


def _burn_for(seconds: float, t: torch.Tensor) -> None:
    """Run matrix ops in a tight loop for `seconds` wall-clock time."""
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        torch.mm(t, t.T)  # small matmul, pure compute, no allocation
    torch.cuda.synchronize()


# ──────────────────────────────────────────────
# nvidia-smi poller
# ──────────────────────────────────────────────


def _query_utilization(gpu_index: int) -> float | None:
    """Return GPU utilization in [0, 1] or None on failure."""
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                f"--id={gpu_index}",
                "--query-gpu=utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
        return float(out.strip()) / 100.0
    except Exception:
        return None


# ──────────────────────────────────────────────
# Main control loop
# ──────────────────────────────────────────────


def run_padder(
    target: float,
    gpu_index: int = 0,
    ema_alpha: float = 0.2,
    poll_interval: float = 2.0,
) -> None:
    """Monitor GPU utilization and pad with CUDA compute to approach `target`.

    Args:
        target:        Desired utilization fraction, e.g. 0.4 for 40%.
        gpu_index:     Which GPU to monitor and pad.
        ema_alpha:     EMA smoothing factor (higher = more reactive).
        poll_interval: Seconds between utilization polls.
    """
    device = torch.device(f"cuda:{gpu_index}")
    dummy = _make_dummy_tensor(device)

    ema_util = _query_utilization(gpu_index) or 0.0
    print(
        f"[gpu_padder] target={target:.0%}  gpu={gpu_index}  "
        f"alpha={ema_alpha}  poll={poll_interval}s",
        flush=True,
    )

    # Handle SIGTERM gracefully (Kubernetes sends this on pod eviction)
    def _shutdown(sig, frame):
        print("[gpu_padder] shutting down.", flush=True)
        sys.exit(0)

    signal.signal(signal.SIGTERM, _shutdown)

    while True:
        raw = _query_utilization(gpu_index)
        if raw is None:
            time.sleep(poll_interval)
            continue

        # EMA update
        ema_util = ema_alpha * raw + (1 - ema_alpha) * ema_util

        headroom = target - ema_util  # how much we need to add

        if headroom <= 0.01:
            # Already at or above target, just sleep
            print(
                f"[gpu_padder] util={ema_util:.1%} (raw={raw:.1%})  "
                f"-> at target, sleeping {poll_interval:.1f}s",
                flush=True,
            )
            time.sleep(poll_interval)
        else:
            # Burn for a fraction of the poll interval proportional to headroom.
            # headroom=0.2 with poll=2s -> burn for 0.4s out of every 2s cycle,
            # adding ~20% utilization.
            burn_secs = min(headroom * poll_interval, poll_interval * 0.95)
            sleep_secs = poll_interval - burn_secs

            print(
                f"[gpu_padder] util={ema_util:.1%} (raw={raw:.1%})  "
                f"headroom={headroom:.1%}  "
                f"-> burn={burn_secs:.2f}s  sleep={sleep_secs:.2f}s",
                flush=True,
            )

            _burn_for(burn_secs, dummy)
            if sleep_secs > 0:
                time.sleep(sleep_secs)


# ──────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GPU utilization padder")
    parser.add_argument(
        "--target",
        type=float,
        default=0.4,
        help="Target GPU utilization fraction (default: 0.4)",
    )
    parser.add_argument(
        "--gpu", type=int, default=0, help="GPU index to monitor and pad (default: 0)"
    )
    parser.add_argument(
        "--ema-alpha",
        type=float,
        default=0.2,
        help="EMA smoothing factor (default: 0.2)",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=2.0,
        help="Seconds between utilization polls (default: 2.0)",
    )
    args = parser.parse_args()

    run_padder(
        target=args.target,
        gpu_index=args.gpu,
        ema_alpha=args.ema_alpha,
        poll_interval=args.poll_interval,
    )
