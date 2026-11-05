"""第 0 篇：画出第一张正弦波。

在「傅里叶变换」文件夹里运行：

    python code/ch00_sine.py

图会保存到 assets/figures/。文件顶部「改这里」只影响 ch00_my_sine.png。
课文对照图每次都按固定参数重画，方便和网页正文对得上。
"""

from pathlib import Path

# 先选定不弹窗的后端，再导入绑图库。SHOW_WINDOW 为 True 时才尝试弹出窗口。
SHOW_WINDOW = False

import matplotlib

if not SHOW_WINDOW:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from plot_style import TEAL, WAVE, finish, use_style

# ===== 改这里 =====
# 只影响 assets/figures/ch00_my_sine.png
AMPLITUDE = 1.0  # 振幅 A：波峰离开中线多远
FREQUENCY = 1.0  # 频率 f：1 秒内完成多少次完整来回，单位 Hz
PHASE = 0.0  # 相位 φ：起步角度，单位弧度。可改成 np.pi / 2
DURATION = 1.0  # 你的图画几秒
# ==================

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "assets" / "figures"


def sine(t, amplitude, frequency, phase):
    """y = A * sin(2π f t + φ)。t 可以是一个数，也可以是一整排时间。"""
    return amplitude * np.sin(2 * np.pi * frequency * t + phase)


def time_axis(seconds=1.0, n=2000):
    # 取很多点只为了把曲线画顺。这不是声音文件的采样率，第 1 篇再讲采样。
    return np.linspace(0, seconds, n)


def new_axes(width=8.8, height=4.5):
    fig, ax = plt.subplots(figsize=(width, height), layout="constrained")
    return fig, ax


def style_time_axis(ax, seconds=1.0):
    finish(ax)
    ax.set_xlim(0, seconds)
    ax.set_xlabel("时间 t（秒）")
    ax.set_ylabel("强弱 y")
    if seconds <= 1:
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    path = FIG / name
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"已保存 {path}")


def plot_basic():
    t = time_axis()
    fig, ax = new_axes()
    ax.plot(t, sine(t, 1.0, 1.0, 0.0), color=WAVE, lw=2.4, zorder=3)
    ax.annotate(
        "",
        xy=(0.25, 1.0),
        xytext=(0.25, 0.0),
        arrowprops=dict(arrowstyle="<->", color=WAVE, lw=1.3),
        zorder=4,
    )
    ax.annotate(
        "振幅 A = 1",
        xy=(0.25, 1.0),
        xytext=(0.02, 1.22),
        color=WAVE,
        fontsize=12,
        arrowprops=dict(arrowstyle="->", color=WAVE, lw=1.0),
    )
    ax.annotate(
        "",
        xy=(1.0, -1.38),
        xytext=(0.0, -1.38),
        arrowprops=dict(arrowstyle="<->", color=TEAL, lw=1.3),
    )
    ax.text(
        0.5,
        -1.68,
        "一个周期 T = 1 秒      频率 f = 1 Hz",
        color=TEAL,
        ha="center",
        fontsize=12,
    )
    ax.set_title("1 秒里来回 1 次的正弦波")
    style_time_axis(ax)
    ax.set_ylim(-2.05, 1.6)
    save(fig, "ch00_basic.png")


def plot_frequency():
    t = time_axis()
    fig, ax = new_axes()
    ax.axvline(0.25, color="#8d8376", lw=1.0, ls=(0, (2, 3)), zorder=1)
    ax.plot(t, sine(t, 1, 3, 0), color=WAVE, lw=2.0, zorder=2)
    ax.plot(t, sine(t, 1, 1, 0), color=TEAL, lw=2.5, zorder=3)
    ax.text(0.083, 1.18, "3 Hz", color=WAVE, ha="center", fontsize=11)
    ax.text(0.25, 1.38, "1 Hz 的波峰\n0.25 秒", color=TEAL, ha="center", va="bottom", fontsize=10)
    ax.annotate(
        "",
        xy=(1 / 3, -1.38),
        xytext=(0.0, -1.38),
        arrowprops=dict(arrowstyle="<->", color=WAVE, lw=1.2),
    )
    ax.text(0.36, -1.68, "橙色的一个周期 ≈ 0.33 秒", color=WAVE, fontsize=12)
    ax.set_title("同一秒里：青色来回 1 次，橙色来回 3 次")
    style_time_axis(ax)
    ax.set_ylim(-2.05, 1.85)
    save(fig, "ch00_freq.png")


def plot_amplitude():
    t = time_axis()
    fig, ax = new_axes()
    ax.plot(t, sine(t, 1.0, 2, 0), color=WAVE, lw=2.3, zorder=3)
    ax.plot(t, sine(t, 0.5, 2, 0), color=TEAL, lw=2.3, zorder=3)
    for height, color in ((1.0, WAVE), (-1.0, WAVE), (0.5, TEAL), (-0.5, TEAL)):
        ax.axhline(height, color=color, lw=0.8, ls=(0, (3, 3)), zorder=2)
    ax.text(0.98, 1.08, "A = 1", ha="right", color=WAVE, fontsize=12)
    ax.text(0.98, 0.58, "A = 0.5", ha="right", color=TEAL, fontsize=12)
    ax.set_title("频率都是 2 Hz：橙色更高，青色更矮")
    style_time_axis(ax)
    ax.set_ylim(-1.7, 1.55)
    save(fig, "ch00_amplitude.png")


def plot_phase():
    t = time_axis()
    fig, ax = new_axes()
    ax.plot(t, sine(t, 1, 1, 0), color=WAVE, lw=2.3, zorder=3)
    ax.plot(t, sine(t, 1, 1, np.pi / 2), color=TEAL, lw=2.3, zorder=3)
    ax.scatter([0], [0], s=42, color=WAVE, zorder=4)
    ax.scatter([0], [1], s=42, color=TEAL, zorder=4)
    ax.annotate(
        "φ = 0，从中线向上",
        xy=(0, 0),
        xytext=(0.1, -0.48),
        color=WAVE,
        fontsize=11,
        arrowprops=dict(arrowstyle="->", color=WAVE, lw=1.0),
    )
    ax.annotate(
        "φ = π/2，从最高点出发",
        xy=(0, 1),
        xytext=(0.14, 1.28),
        color=TEAL,
        fontsize=11,
        arrowprops=dict(arrowstyle="->", color=TEAL, lw=1.0),
    )
    ax.set_title("频率都是 1 Hz：起步位置不同")
    style_time_axis(ax)
    ax.set_ylim(-1.85, 1.7)
    save(fig, "ch00_phase.png")


def plot_mine():
    if DURATION <= 0:
        raise SystemExit("DURATION 请用正数，单位是秒。")
    t = time_axis(DURATION)
    y = sine(t, AMPLITUDE, FREQUENCY, PHASE)
    fig, ax = new_axes()
    ax.plot(t, y, color=WAVE, lw=2.4, zorder=3)
    style_time_axis(ax, DURATION)
    peak = max(1.0, abs(float(AMPLITUDE)))
    ax.set_ylim(-peak * 1.45, peak * 1.4)
    if DURATION > 1:
        step = 0.5 if DURATION <= 3 else 1.0
        ax.set_xticks(np.arange(0, DURATION + 1e-9, step))
    ax.set_title(
        f"你的参数：A = {AMPLITUDE:g}，f = {FREQUENCY:g} Hz，φ = {PHASE:g} 弧度"
    )
    save(fig, "ch00_my_sine.png")
    if SHOW_WINDOW:
        fig2, ax2 = new_axes()
        ax2.plot(t, y, color=WAVE, lw=2.4)
        style_time_axis(ax2, DURATION)
        ax2.set_ylim(-peak * 1.45, peak * 1.4)
        ax2.set_title("你的正弦波")
        plt.show()
        plt.close(fig2)


def self_check():
    peak = sine(0.25, 1.0, 1.0, 0.0)
    start = sine(0.0, 1.0, 1.0, np.pi / 2)
    if abs(peak - 1.0) > 1e-9 or abs(start - 1.0) > 1e-9:
        raise SystemExit(f"自检失败：波峰 {peak}，相位 π/2 的起点 {start}")
    print("自检通过：f = 1 Hz 时，t = 0.25 秒的 y = 1；φ = π/2 时，起点 y = 1。")


def main():
    use_style()
    self_check()
    plot_basic()
    plot_frequency()
    plot_amplitude()
    plot_phase()
    plot_mine()
    print("课文对照图：ch00_basic.png、ch00_freq.png、ch00_amplitude.png、ch00_phase.png")
    print("你的图：ch00_my_sine.png（只跟随文件顶部的四个参数）")


if __name__ == "__main__":
    main()
