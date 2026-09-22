#!/usr/bin/env python3
"""Render the performance charts embedded in the Qwen3-30B-A3B optimization write-up.

Confluence Cloud has no built-in Chart macro, so the charts are pre-rendered to PNG
and uploaded as page attachments instead.

    python3 -m venv /tmp/chartenv && /tmp/chartenv/bin/pip install matplotlib
    /tmp/chartenv/bin/python examples/qwen3-30b-a3b/docs/make_charts.py

Labels are English on purpose: the box has no CJK fonts installed.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).parent / "charts"
OUT.mkdir(exist_ok=True)

BEFORE, AFTER, ACCENT, MUTED = "#B8531F", "#1F7A4D", "#2B6CB0", "#9AA5B1"
plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.25,
                     "axes.axisbelow": True, "figure.dpi": 150})

STAGES = ["A0\nbaseline", "A1\n+CK Attn", "A2\n+Triton GEMM",
          "A3\n+EP Overlap", "A4\n+CONN=8", "A5\n+attn CUDA Graph"]
STEP_S = [2.27, 1.94, 1.72, 1.62, 1.57, 1.14]
SPS = [7.04, 8.24, 9.28, 9.90, 10.19, 14.07]
TFLOPS = [82.9, 97.0, 109.3, 116.6, 120.1, 165.7]


def save(fig, name):
    path = OUT / name
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {path}")


def c01_e2e_bigbatch():
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.4))
    names = ["baseline", "4 opts", "+ CUDA Graph"]
    panels = [
        ("Step time (s)\nlower is better", [24.71, 17.47, 12.72], "-48.5%", "{:.2f}"),
        ("Throughput (samples/s)\nhigher is better", [10.36, 14.66, 20.13], "+94.3%", "{:.2f}"),
        ("TFLOP/s/GPU\nhigher is better", [122.0, 172.7, 237.2], "+94.4%", "{:.1f}"),
    ]
    for ax, (title, vals, delta, fmt) in zip(axes, panels):
        bars = ax.bar(names, vals, color=[BEFORE, ACCENT, AFTER], width=0.6)
        ax.bar_label(bars, fmt=fmt, padding=3, fontweight="bold")
        ax.set_title(title, fontsize=11)
        ax.set_ylim(0, max(vals) * 1.30)
        ax.annotate(delta, xy=(0.5, 0.90), xycoords="axes fraction", ha="center",
                    fontsize=13, fontweight="bold", color=AFTER)
        ax.tick_params(axis="x", labelsize=9.5)
    fig.suptitle("End-to-end: MBS=2, GBS=256, seq=4096  (median of steps 11-20)",
                 fontsize=13, fontweight="bold")
    save(fig, "01-e2e-bigbatch.png")


def c02_cumulative_steptime():
    fig, ax = plt.subplots(figsize=(10, 4.6))
    colors = [BEFORE] + [ACCENT] * 4 + [AFTER]
    bars = ax.bar(STAGES, STEP_S, color=colors, width=0.6)
    ax.bar_label(bars, fmt="%.2f s", padding=3, fontweight="bold")
    for i in range(1, len(STEP_S)):
        pct = (STEP_S[i] - STEP_S[i - 1]) / STEP_S[i - 1] * 100
        ax.annotate(f"{pct:+.1f}%", xy=(i, STEP_S[i] * 0.5), ha="center",
                    color="white", fontweight="bold", fontsize=10)
    ax.set_ylabel("Median step time (s)")
    ax.set_ylim(0, 2.7)
    ax.set_title("Cumulative optimization: step time\nMBS=1, GBS=16 (median of steps 4-10)",
                 fontweight="bold")
    ax.annotate("", xy=(5, 2.45), xytext=(0, 2.45),
                arrowprops=dict(arrowstyle="<->", color="#444", lw=1.4))
    ax.annotate("total -49.9%  (2.00x)", xy=(2.5, 2.52), ha="center",
                fontweight="bold", fontsize=11)
    save(fig, "02-cumulative-steptime.png")


def c03_throughput_tflops():
    fig, ax = plt.subplots(figsize=(10, 4.6))
    ax.plot(STAGES, SPS, "o-", color=AFTER, lw=2.4, ms=8, label="Throughput (samples/s)")
    for x, y in zip(range(len(SPS)), SPS):
        ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points",
                    xytext=(0, 10), ha="center", color=AFTER, fontweight="bold")
    ax.set_ylabel("Throughput (samples/s)", color=AFTER)
    ax.set_ylim(6, 16.2)

    ax2 = ax.twinx()
    ax2.plot(STAGES, TFLOPS, "s--", color=ACCENT, lw=2.2, ms=7, label="TFLOP/s/GPU")
    for x, y in zip(range(len(TFLOPS)), TFLOPS):
        ax2.annotate(f"{y:.1f}", (x, y), textcoords="offset points",
                     xytext=(0, -18), ha="center", color=ACCENT, fontweight="bold")
    ax2.set_ylabel("TFLOP/s/GPU", color=ACCENT)
    ax2.set_ylim(70, 190)
    ax2.grid(False)

    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(lines, [l.get_label() for l in lines], loc="upper left", framealpha=0.9)
    ax.set_title("Cumulative optimization: throughput and compute utilization\n"
                 "MBS=1, GBS=16 (median of steps 4-10)", fontweight="bold")
    save(fig, "03-throughput-tflops.png")


def c04_waterfall():
    fig, ax = plt.subplots(figsize=(10.5, 4.6))
    labels = ["A0\nbaseline", "CK\nAttention", "Triton\nGrouped GEMM",
              "EP A2A\nOverlap", "CONN=8", "attn\nCUDA Graph", "A5\nfinal"]
    deltas = [-330, -220, -100, -50, -430]
    start = 2270.0
    running = start
    ax.bar(0, start, color=BEFORE, width=0.6)
    ax.annotate(f"{start:.0f}", (0, start), textcoords="offset points",
                xytext=(0, 5), ha="center", fontweight="bold")
    for i, d in enumerate(deltas, start=1):
        ax.bar(i, -d, bottom=running + d, color=ACCENT, width=0.6, alpha=0.9)
        if -d >= 150:
            ax.annotate(f"{d}", (i, running + d / 2), ha="center", va="center",
                        color="white", fontweight="bold")
        else:
            ax.annotate(f"{d}", (i, running + 12), ha="center", va="bottom",
                        color=ACCENT, fontweight="bold")
        ax.plot([i - 0.3, i + 0.7], [running + d, running + d],
                color="#666", lw=0.9, ls=":")
        running += d
    ax.bar(len(deltas) + 1, running, color=AFTER, width=0.6)
    ax.annotate(f"{running:.0f}", (len(deltas) + 1, running), textcoords="offset points",
                xytext=(0, 5), ha="center", fontweight="bold")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Median step time (ms)")
    ax.set_ylim(0, 2600)
    ax.set_title("Where the 1130 ms came from  (MBS=1, GBS=16)", fontweight="bold")
    save(fig, "04-waterfall.png")


def c05_microbench():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    ax = axes[0]
    x = [0, 1]
    w = 0.35
    tri, ck = [1.03, 3.21], [0.17, 0.68]
    b1 = ax.bar([i - w / 2 for i in x], tri, w, label="Lumen Triton", color=BEFORE)
    b2 = ax.bar([i + w / 2 for i in x], ck, w, label="AITER CK (fmha_v3)", color=AFTER)
    ax.bar_label(b1, fmt="%.2f", padding=2, fontsize=9)
    ax.bar_label(b2, fmt="%.2f", padding=2, fontsize=9)
    for i, (t, c) in enumerate(zip(tri, ck)):
        ax.annotate(f"{t / c:.1f}x", (i, max(t, c) * 1.12), ha="center",
                    fontweight="bold", color=AFTER, fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(["Forward", "Backward"])
    ax.set_ylabel("Latency (ms)")
    ax.set_ylim(0, 4.0)
    ax.legend(fontsize=9)
    ax.set_title("Attention kernel\nQ[1,4096,32,128] KV[1,4096,4,128] BF16 causal", fontsize=10)

    ax = axes[1]
    names = ["Triton\n(Qwen3 tuned)", "Triton\n(autotune)", "hipBLASLt\nmultistream", "hipBLASLt\ngrouped"]
    vals = [1.78, 1.79, 2.30, 0.0]
    bars = ax.bar(names, vals, color=[AFTER, ACCENT, BEFORE, MUTED], width=0.6)
    ax.bar_label(bars, labels=["1.78", "1.79", "2.30", "FAIL"], padding=3, fontweight="bold")
    ax.set_ylabel("Latency (ms)")
    ax.set_ylim(0, 2.85)
    ax.set_title("Grouped GEMM, 6 ops per layer\nE=16, TK=32768", fontsize=10)
    ax.annotate("unsupported:\nempty expert N=1", (3, 0.45), ha="center",
                fontsize=8.5, color="#8A2020", fontweight="bold")

    fig.suptitle("Kernel-level microbenchmarks", fontsize=13, fontweight="bold")
    save(fig, "05-microbench.png")


def c06_profile_ops():
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    labels = ["Grouped GEMM", "Attention\n(fwd+bwd)", "RCCL comm", "Elementwise", "Other"]
    before = [742.0, 347.3, 575.0, 107.7, 701.0]
    after = [197.0, 75.5, 557.2, 131.5, 698.8]
    y = range(len(labels))
    h = 0.36
    b1 = ax.barh([i + h / 2 for i in y], before, h, label="A0 baseline (2.473 s)", color=BEFORE)
    b2 = ax.barh([i - h / 2 for i in y], after, h, label="A2 optimized (1.660 s)", color=AFTER)
    ax.bar_label(b1, fmt="%.0f", padding=3, fontsize=9)
    ax.bar_label(b2, fmt="%.0f", padding=3, fontsize=9)
    for i, (bb, aa) in enumerate(zip(before, after)):
        pct = (aa - bb) / bb * 100
        ax.annotate(f"{pct:+.0f}%", (max(bb, aa) + 95, i), va="center",
                    fontweight="bold", fontsize=10,
                    color=AFTER if pct < -1 else ("#8A2020" if pct > 1 else MUTED))
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Self CUDA time (ms), rank0, one train step")
    ax.set_xlim(0, 940)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, frameon=False)
    ax.set_title("Profiler: per-operator self CUDA time\nMBS=1 GBS=16, step 6", fontweight="bold")
    save(fig, "06-profile-ops.png")


def c07_remaining():
    fig, ax = plt.subplots(figsize=(9, 4.4))
    labels = ["Other\n(permute/sort/norm)", "RCCL comm", "Grouped GEMM",
              "Elementwise", "Attention"]
    vals = [698.8, 557.2, 197.0, 131.5, 75.5]
    targets = ["-", "F1 / F3", "F7 / F9", "F2", "done"]
    colors = [MUTED, BEFORE, ACCENT, ACCENT, AFTER]
    bars = ax.barh(labels, vals, color=colors, height=0.6)
    ax.bar_label(bars, fmt="%.0f ms", padding=4, fontweight="bold", fontsize=10)
    for i, t in enumerate(targets):
        ax.annotate(t, (18, i), va="center", color="white",
                    fontweight="bold", fontsize=9.5)
    ax.invert_yaxis()
    ax.set_xlabel("Self CUDA time (ms)")
    ax.set_xlim(0, 860)
    ax.set_title("Remaining hotspots and which follow-up item targets them\n"
                 "(A2 profile; labels refer to section 8)", fontweight="bold")
    save(fig, "07-remaining-hotspots.png")


if __name__ == "__main__":
    c01_e2e_bigbatch()
    c02_cumulative_steptime()
    c03_throughput_tflops()
    c04_waterfall()
    c05_microbench()
    c06_profile_ops()
    c07_remaining()
