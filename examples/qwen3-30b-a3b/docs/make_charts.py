#!/usr/bin/env python3
"""Render the charts used by the two BF16 optimization write-ups.

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

BEFORE, AFTER, ACCENT = "#B8531F", "#1F7A4D", "#2B6CB0"
plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.25,
                     "axes.axisbelow": True, "figure.dpi": 150})



def save(fig, name):
    path = OUT / name
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {path}")


def c01_e2e_bigbatch():
    fig, axes = plt.subplots(1, 2, figsize=(14.4, 5.0))
    labels = ["baseline", "+4 opts", "+attn graph", "+FlyDSL",
              "+OPUS fwd", "+async D2H", "+Q RMSNorm", "+router bf16",
              "+atomic16"]
    steps = [24.50, 16.31, 12.19, 11.08, 10.66, 10.55, 10.20, 10.17, 9.88]
    sps = [10.45, 15.69, 21.01, 23.10, 24.01, 24.28, 25.09, 25.16, 25.92]
    step_top = ["24.50", "16.31", "12.19", "11.08", "10.66", "10.55", "10.20", "10.17", "9.88"]
    sps_top = ["10.45", "15.69", "21.01", "23.10", "24.01", "24.28", "25.09", "25.16", "25.92"]
    colors = [BEFORE] + [ACCENT] * 7 + [AFTER]
    for ax, vals, top, title in (
        (axes[0], steps, step_top, "Step time (s), lower is better"),
        (axes[1], sps, sps_top, "Throughput (samples/s), higher is better"),
    ):
        bars = ax.bar(range(len(vals)), vals, color=colors, width=0.72)
        ax.bar_label(bars, labels=top, padding=2, fontweight="bold", fontsize=8)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_title(title, fontsize=11)
        ax.set_ylim(0, max(vals) * 1.22)
    fig.suptitle("End-to-end: MBS=2, GBS=256, seq=4096  (median of steps 11-20)",
                 fontsize=13, fontweight="bold")
    save(fig, "01-e2e-bigbatch.png")



def c09_step_vs_ceiling():
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    labels = ["Measured\nFlyDSL + OPUS", "GEMMs at\nsampled kernel rate", "Link floor\nRCCL all-to-all stream"]
    vals = [9.875, 7.63, 5.33]
    colors = [AFTER, ACCENT, "#a65b12"]
    bars = ax.bar(labels, vals, color=colors, width=0.58)
    ax.bar_label(bars, labels=["9.88 s", "7.6 s", "5.3 s"], padding=4, fontweight="bold")
    ax.set_ylabel("Step time (s), 1.05M tokens")
    ax.set_ylim(0, 12)
    ax.set_title("Step-time bounds of the current link on 8x MI350X\nseq=4096, MBS=2, GBS=256",
                 fontweight="bold")
    save(fig, "09-step-vs-ceiling.png")


# FSDP BF16 serial path, MBS=2 GBS=256, median of steps 11-20.
FSDP_LABELS = [
    "SDPA",
    "+OPUS",
    "defer RS",
    "+transpose",
    "+reduce",
    "+gather",
    "+remap",
    "+SwiGLU",
    "+reduce bwd",
    "+dgrad",
    "+skip gather",
]
FSDP_STEP = [19.49, 16.97, 16.22, 15.16, 14.67, 14.09, 11.85, 11.73, 11.56, 11.38, 10.95]
FSDP_TOP = ["19.49", "16.97", "16.22", "15.16", "14.67", "14.09", "11.85", "11.73", "11.56", "11.38", "10.95"]


def c10_fsdp_journey():
    fig, ax = plt.subplots(figsize=(14.4, 5.2))
    colors = [BEFORE] + [ACCENT] * (len(FSDP_STEP) - 2) + [AFTER]
    bars = ax.bar(range(len(FSDP_STEP)), FSDP_STEP, color=colors, width=0.72)
    ax.bar_label(bars, labels=FSDP_TOP, padding=3, fontweight="bold", fontsize=8.5)
    ax.set_xticks(range(len(FSDP_LABELS)))
    ax.set_xticklabels(FSDP_LABELS, fontsize=8)
    ax.set_ylabel("Median step time (s)")
    ax.set_ylim(0, 26)
    ax.set_title("FSDP BF16 serial path\nMBS=2, GBS=256, seq=4096, median of steps 11-20",
                 fontweight="bold")
    ax.axhline(9.875, color="#a65b12", ls="--", lw=1.2, label="Megatron current  9.88 s")
    ax.legend(loc="upper right", framealpha=0.95)
    save(fig, "10-fsdp-bf16-journey.png")


def c11_fsdp_vs_megatron():
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    labels = ["FSDP current\nmeasured step", "Megatron current\nmeasured step", "FSDP all-to-all\nexposed on the step"]
    vals = [10.95, 9.875, 2.87]
    colors = [AFTER, ACCENT, "#a65b12"]
    bars = ax.bar(labels, vals, color=colors, width=0.58)
    ax.bar_label(bars, labels=["10.95 s", "9.88 s", "2.87 s"], padding=4, fontweight="bold")
    ax.set_ylabel("Time (s)")
    ax.set_ylim(0, 14)
    ax.set_title("Current FSDP step against the Megatron step\n2.87 s is exposed all-to-all, not a step time",
                 fontweight="bold")
    save(fig, "11-fsdp-bf16-vs-megatron.png")


if __name__ == "__main__":
    c01_e2e_bigbatch()
    c09_step_vs_ceiling()
    c10_fsdp_journey()
    c11_fsdp_vs_megatron()
