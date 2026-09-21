"""Microbench OPUS vs csrc attention on the dense-pretrain shapes.

Shapes match ``examples/{llama2,llama31,qwen3}/run_pretrain_*.sh`` with
TP=1 / CP=1 / 8 GPUs (the configs used for the 50-step suite):

    llama2-7b   MHA  B=4 S=4096  H_q=H_kv=32  D=128  32 layers  8 μbatches
    llama3-8b   GQA  B=2 S=8192  H_q=32 H_kv=8 D=128  32 layers  8 μbatches
    qwen3-8b    GQA  B=2 S=8192  H_q=32 H_kv=8 D=128  36 layers  8 μbatches

Layout matches :class:`~lumen.modules.attention_megatron.LumenDotProductAttention`:
Q/K/V are allocated ``[S, B, H, D]`` (Megatron) and viewed as ``[B, S, H, D]``
without a copy; ``seq_major_out=True``. Causal, bf16, dropout=0.

OPUS only replaces the forward; ``bwd`` is the csrc (CK/asm) backward on both
backends. ``fwd+bwd`` is timed as one autograd call (the training pairing).

The e2e share table multiplies kernel time by ``layers × microbatches`` and
divides by the 50-step suite mean (bf16 and fp8; attention stays bf16 in both).

    python tests/kernels/bench_opus_attn_vs_csrc.py
    python tests/kernels/bench_opus_attn_vs_csrc.py --iters 50 --warmup 10
    pytest tests/kernels/bench_opus_attn_vs_csrc.py -s
"""

from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch

from lumen.kernels.attention.attention_impl import csrc_available, opus_available
from lumen.ops.attention.attention import attention

WARMUP = 10
ITERS = 30
REPO_ROOT = Path(__file__).resolve().parents[2]
CSRC_SUMMARY = REPO_ROOT / "examples/results/opus_attn_dense_perf/SUMMARY.json"
OPUS_SUMMARY = REPO_ROOT / "examples/results/opus_attn_dense_perf_opus/SUMMARY.json"
NPROC = 8  # torchrun --nproc_per_node=8, TP=PP=CP=1 → DP=8


@dataclass(frozen=True)
class Shape:
    name: str
    batch: int
    seq: int
    nhead_q: int
    nhead_kv: int
    head_dim: int
    layers: int
    gbs: int
    note: str

    @property
    def microbatches(self) -> int:
        return self.gbs // NPROC // self.batch

    @property
    def calls_per_step(self) -> int:
        return self.microbatches * self.layers

    def fwd_flops(self, causal: bool = True) -> int:
        flops = 4 * self.batch * self.nhead_q * self.seq * self.seq * self.head_dim
        return flops // 2 if causal else flops


SHAPES = (
    Shape("llama2-7b", 4, 4096, 32, 32, 128, 32, 256, "MHA"),
    Shape("llama3-8b", 2, 8192, 32, 8, 128, 32, 128, "GQA 32/8"),
    Shape("qwen3-8b", 2, 8192, 32, 8, 128, 36, 128, "GQA 32/8"),
)


def _require_gpu():
    if not torch.cuda.is_available():
        pytest.skip("requires a GPU")
    if not csrc_available("fmha_fwd_bf16_opus_fwd"):
        pytest.skip("OPUS forward not exposed by aiter")
    return torch.device("cuda")


def _sbhd(b: int, s: int, h: int, d: int, device, seed: int) -> torch.Tensor:
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    packed = torch.randn(s, b, h, d, device=device, dtype=torch.bfloat16, generator=g)
    return packed.permute(1, 0, 2, 3)


def _make_qkv(shape: Shape, device, seed: int = 0):
    q = _sbhd(shape.batch, shape.seq, shape.nhead_q, shape.head_dim, device, seed)
    k = _sbhd(shape.batch, shape.seq, shape.nhead_kv, shape.head_dim, device, seed + 1)
    v = _sbhd(shape.batch, shape.seq, shape.nhead_kv, shape.head_dim, device, seed + 2)
    return q, k, v


def _attn(q, k, v, backend: str):
    return attention(
        q,
        k,
        v,
        softmax_scale=q.shape[-1] ** -0.5,
        causal=True,
        backend_type=backend,
        seq_major_out=True,
    )


def _stats(samples: list[float]) -> tuple[float, float, float]:
    return statistics.median(samples), min(samples), max(samples)


def _time_cuda(fn, warmup: int, iters: int) -> tuple[float, float, float]:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    def one() -> float:
        torch.cuda.synchronize()
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end)

    for _ in range(warmup):
        one()
    return _stats([one() for _ in range(iters)])


def _bench_fwd(q, k, v, backend: str, warmup: int, iters: int):
    q, k, v = q.detach(), k.detach(), v.detach()

    def fn():
        with torch.no_grad():
            _attn(q, k, v, backend)

    return _time_cuda(fn, warmup, iters)


def _bench_bwd(q, k, v, backend: str, warmup: int, iters: int):
    """Time only ``backward``. Forward is rerun each iter, outside the event."""
    q = q.detach().requires_grad_(True)
    k = k.detach().requires_grad_(True)
    v = v.detach().requires_grad_(True)
    dout = torch.randn_like(_attn(q, k, v, backend).detach())
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    def one() -> float:
        q.grad = k.grad = v.grad = None
        out = _attn(q, k, v, backend)
        torch.cuda.synchronize()
        start.record()
        out.backward(dout)
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end)

    for _ in range(warmup):
        one()
    return _stats([one() for _ in range(iters)])


def _bench_fwd_bwd(q, k, v, backend: str, warmup: int, iters: int):
    q = q.detach().requires_grad_(True)
    k = k.detach().requires_grad_(True)
    v = v.detach().requires_grad_(True)
    dout = torch.randn_like(_attn(q, k, v, backend).detach())

    def fn():
        q.grad = k.grad = v.grad = None
        out = _attn(q, k, v, backend)
        out.backward(dout)

    return _time_cuda(fn, warmup, iters)


def _tflops(flops: int, ms: float) -> float:
    return flops / (ms * 1e-3) / 1e12


def _load_e2e(path: Path) -> dict[tuple[str, str], float]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    return {(j["model"], j["precision"]): j["iter_ms_mean"] for j in data["jobs"]}


def _print_row(name, op, csrc, opus, csrc_tf, opus_tf, extra=""):
    speedup = csrc / opus
    csrc_tf_s = f"{csrc_tf:9.1f}" if csrc_tf is not None else f"{'—':>9}"
    opus_tf_s = f"{opus_tf:9.1f}" if opus_tf is not None else f"{'—':>9}"
    print(
        f"{name:<12}{op:<8}{csrc:10.3f}{opus:10.3f}"
        f"{speedup:9.3f}{csrc_tf_s}{opus_tf_s}  {extra}"
    )


def run_bench(models: tuple[str, ...] | None, warmup: int, iters: int) -> list[dict]:
    device = _require_gpu()
    gfx = torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
    print(f"arch={gfx}  warmup={warmup}  iters={iters}")
    print(
        f"{'model':<12}{'op':<8}{'csrc_ms':>10}{'opus_ms':>10}"
        f"{'speedup':>9}{'csrc_TF':>9}{'opus_TF':>9}  shape"
    )
    print("-" * 108)

    benches = (("fwd", _bench_fwd), ("bwd", _bench_bwd), ("fwd+bwd", _bench_fwd_bwd))
    rows: list[dict] = []
    by_model: dict[str, dict[str, dict[str, float]]] = {}

    for shape in SHAPES:
        if models and shape.name not in models:
            continue
        q, k, v = _make_qkv(shape, device)
        if not opus_available(q, k, v):
            raise RuntimeError(f"{shape.name}: OPUS ineligible for the training shape")

        extra = (
            f"q[{shape.batch},{shape.seq},{shape.nhead_q},{shape.head_dim}] "
            f"kvH={shape.nhead_kv} L={shape.layers} μb={shape.microbatches} {shape.note}"
        )
        by_model[shape.name] = {}
        for op, bench in benches:
            csrc_med, csrc_min, csrc_max = bench(q, k, v, "aiter_csrc", warmup, iters)
            opus_med, opus_min, opus_max = bench(q, k, v, "aiter_opus", warmup, iters)
            flops = shape.fwd_flops() if op == "fwd" else None
            csrc_tf = _tflops(flops, csrc_med) if flops else None
            opus_tf = _tflops(flops, opus_med) if flops else None
            _print_row(shape.name, op, csrc_med, opus_med, csrc_tf, opus_tf, extra)
            rec = {
                "model": shape.name,
                "op": op,
                "csrc_ms": round(csrc_med, 3),
                "opus_ms": round(opus_med, 3),
                "speedup": round(csrc_med / opus_med, 3),
                "csrc_ms_min": round(csrc_min, 3),
                "csrc_ms_max": round(csrc_max, 3),
                "opus_ms_min": round(opus_min, 3),
                "opus_ms_max": round(opus_max, 3),
                "csrc_tflops": None if csrc_tf is None else round(csrc_tf, 1),
                "opus_tflops": None if opus_tf is None else round(opus_tf, 1),
                "batch": shape.batch,
                "seq": shape.seq,
                "nhead_q": shape.nhead_q,
                "nhead_kv": shape.nhead_kv,
                "head_dim": shape.head_dim,
                "layers": shape.layers,
                "microbatches": shape.microbatches,
            }
            rows.append(rec)
            by_model[shape.name][op] = {"csrc": csrc_med, "opus": opus_med}
        print()

    _print_e2e_share(by_model, {s.name: s for s in SHAPES if s.name in by_model})
    return rows


def _print_e2e_share(by_model: dict, shapes: dict[str, Shape]) -> None:
    csrc_e2e = _load_e2e(CSRC_SUMMARY)
    opus_e2e = _load_e2e(OPUS_SUMMARY)
    if not csrc_e2e and not opus_e2e:
        print("e2e SUMMARY.json not found; skip attention share table")
        return

    print(
        "attention share of one training step  "
        "(kernel × layers × microbatches / suite mean_ms)"
    )
    print(
        f"{'model':<12}{'prec':<6}{'be':<6}{'fwd_ms':>8}{'bwd_ms':>8}"
        f"{'attn_ms':>9}{'e2e_ms':>9}{'fwd%':>7}{'bwd%':>7}{'attn%':>7}"
    )
    print("-" * 90)

    for name, shape in shapes.items():
        times = by_model[name]
        calls = shape.calls_per_step
        for prec in ("bf16", "fp8"):
            for be, e2e_map in (("csrc", csrc_e2e), ("opus", opus_e2e)):
                e2e = e2e_map.get((name, prec))
                if e2e is None:
                    continue
                fwd = times["fwd"][be] * calls
                bwd = times["bwd"][be] * calls
                attn = fwd + bwd
                print(
                    f"{name:<12}{prec:<6}{be:<6}{fwd:8.1f}{bwd:8.1f}"
                    f"{attn:9.1f}{e2e:9.1f}"
                    f"{100.0 * fwd / e2e:6.1f}%{100.0 * bwd / e2e:6.1f}%"
                    f"{100.0 * attn / e2e:6.1f}%"
                )
        print()


def test_opus_vs_csrc_real_shapes():
    """Training-shape smoke + timing. Does not fail on speedup (run-to-run noise)."""
    rows = run_bench(models=None, warmup=5, iters=10)
    assert {r["model"] for r in rows} == {"llama2-7b", "llama3-8b", "qwen3-8b"}
    assert {r["op"] for r in rows} == {"fwd", "bwd", "fwd+bwd"}
    assert all(r["csrc_ms"] > 0 and r["opus_ms"] > 0 for r in rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--warmup", type=int, default=WARMUP)
    ap.add_argument("--iters", type=int, default=ITERS)
    ap.add_argument(
        "--models",
        default="",
        help="comma-separated subset of llama2-7b,llama3-8b,qwen3-8b",
    )
    args = ap.parse_args()
    models = tuple(m.strip() for m in args.models.split(",") if m.strip()) or None
    run_bench(models, args.warmup, args.iters)


if __name__ == "__main__":
    main()
