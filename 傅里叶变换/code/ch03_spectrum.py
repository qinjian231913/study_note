"""第 3 篇：同一段波的时域和频域。

在「傅里叶变换」文件夹里运行：

    python code/ch03_spectrum.py

课文对照图每次都按固定配方重画。
文件顶部的三组频率和振幅只影响 ch03_mine.png 和 ch03_mine.wav。

这一篇的柱高是配方里写好的振幅，不是从波形反推出来的。
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_style import INK, MUTED, TEAL, WAVE, finish, use_style

# ===== 改这里 =====
# 每组是频率（Hz）和振幅。振幅用 0 到 1。
# 振幅写成 0，或频率写成 0，表示关掉这根柱子。
# 只影响 assets/figures/ch03_mine.png 和 assets/audio/ch03_mine.wav
F1, A1 = 440.0, 0.55
F2, A2 = 660.0, 0.35
F3, A3 = 880.0, 0.25
# ==================

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "assets" / "figures"
AUDIO = ROOT / "assets" / "audio"
SAMPLE_RATE = 44100


def synthesize(t, tones, phases=None):
    """按配方把正弦加起来。tones 是 (频率, 振幅)，phases 是弧度，省略时全为 0。

    振幅为 0 或频率不大于 0 的项会跳过，相当于这根柱子不存在。
    """
    t = np.asarray(t, dtype=float)
    y = np.zeros_like(t, dtype=float)
    if phases is None:
        phases = np.zeros(len(tones))
    for (frequency, amplitude), phase in zip(tones, phases):
        if amplitude == 0 or frequency <= 0:
            continue
        y = y + amplitude * np.sin(2 * np.pi * frequency * t + phase)
    return y


def square_tones(n_terms, frequency=1.0):
    """方波前 n_terms 项的配方：奇数倍，振幅 (4/π) / 奇数。"""
    tones = []
    for k in range(int(n_terms)):
        harmonic = 2 * k + 1
        tones.append((harmonic * frequency, (4 / np.pi) / harmonic))
    return tones


def ideal_square(t):
    frac = np.mod(t, 1.0)
    return np.where(frac < 0.5, 1.0, -1.0)


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    path = FIG / name
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"已保存 {path}")


def dress_time(ax):
    finish(ax)
    ax.set_ylabel("强弱 y")


def active(tones):
    return [(float(f), float(a)) for f, a in tones if a != 0 and f > 0]


def bar_width(freqs, freq_max):
    width = freq_max * 0.04
    ordered = sorted(freqs)
    gaps = [b - a for a, b in zip(ordered, ordered[1:]) if b > a]
    if gaps:
        width = min(width, min(gaps) * 0.7)
    return max(width, freq_max * 0.012)


def draw_bars(ax, tones, freq_max, ticks, title):
    chosen = active(tones)
    freqs = [f for f, _a in chosen]
    amps = [a for _f, a in chosen]
    if freq_max is None:
        freq_max = (max(freqs) * 1.35) if freqs else 1.0
    peak = max(amps) if amps else 1.0
    ymax = max(1.15, peak * 1.38)
    if freqs:
        ax.bar(freqs, amps, width=bar_width(freqs, freq_max), color=WAVE, zorder=3)
        for frequency, amplitude in zip(freqs, amps):
            ax.text(
                frequency,
                amplitude + ymax * 0.03,
                f"{amplitude:.2f}",
                ha="center",
                va="bottom",
                fontsize=10,
                color=INK,
            )
    ax.set_xlim(0, freq_max)
    ax.set_ylim(0, ymax)
    ax.set_xticks(ticks)
    ax.set_xlabel("频率（Hz）")
    ax.set_ylabel("振幅")
    ax.set_title(title)
    finish(ax)


def plot_two():
    tones = [(440.0, 1.0), (880.0, 1.0)]
    seconds = 0.01
    t = np.linspace(0, seconds, 4000, endpoint=False)
    y1 = synthesize(t, [tones[0]])
    y2 = synthesize(t, [tones[1]])
    fig, axes = plt.subplots(
        2, 1, figsize=(8.8, 6.5), layout="constrained", height_ratios=(1.25, 1)
    )
    x = t * 1000
    axes[0].plot(x, y1, color=TEAL, lw=1.5, zorder=2, label="440 Hz")
    axes[0].plot(x, y2, color=WAVE, lw=1.5, zorder=2, label="880 Hz")
    axes[0].plot(x, y1 + y2, color=INK, lw=2.2, zorder=3, label="相加")
    axes[0].set_title("时域：440 Hz 与 880 Hz 相加，振幅都是 1")
    axes[0].set_xlabel("时间（毫秒）")
    axes[0].set_xlim(0, 10)
    axes[0].set_ylim(-2.4, 2.4)
    axes[0].legend(frameon=False, ncol=3, loc="upper right")
    dress_time(axes[0])
    draw_bars(
        axes[1],
        tones,
        freq_max=1100,
        ticks=[0, 440, 660, 880],
        title="频域：两根柱。660 Hz 没有写入配方，所以那里是空的",
    )
    save(fig, "ch03_two.png")


def plot_phase():
    tones = [(1.0, 1.0), (2.0, 1.0)]
    seconds = 1.0
    t = np.linspace(0, seconds, 4000, endpoint=False)
    fig, axes = plt.subplots(
        3, 1, figsize=(8.8, 8.4), layout="constrained", height_ratios=(1, 1, 0.85)
    )
    specs = (
        (0.0, "时域：第二条相位 0°。0.25 秒时，慢的在波峰、快的过零，相加为 1"),
        (np.pi / 2, "时域：第二条相位 90°。0.25 秒时两条一正一负，相加为 0"),
    )
    for ax, (phase, title) in zip(axes[:2], specs):
        phases = [0.0, phase]
        y1 = synthesize(t, [tones[0]], [0.0])
        y2 = synthesize(t, [tones[1]], [phase])
        ax.plot(t, y1, color=TEAL, lw=1.4, zorder=2, label="1 Hz")
        ax.plot(t, y2, color=WAVE, lw=1.4, zorder=2, label="2 Hz")
        ax.plot(t, y1 + y2, color=INK, lw=2.2, zorder=3, label="相加")
        ax.axvline(0.25, color="#8d8376", lw=1.0, ls=(0, (2, 3)), zorder=1)
        ax.set_title(title)
        ax.set_xlim(0, 1)
        ax.set_ylim(-2.4, 2.4)
        ax.legend(frameon=False, ncol=3, loc="upper right")
        dress_time(ax)
    axes[1].set_xlabel("时间 t（秒）")
    draw_bars(
        axes[2],
        tones,
        freq_max=4.5,
        ticks=[0, 1, 2, 3, 4],
        title="频域：上面两张时域图共用这一对柱高，相位没有画进来",
    )
    save(fig, "ch03_phase.png")


def plot_square():
    tones = square_tones(5, 1.0)
    t = np.linspace(0, 2, 8000, endpoint=False)
    fig, axes = plt.subplots(
        2, 1, figsize=(8.8, 6.6), layout="constrained", height_ratios=(1.2, 1)
    )
    axes[0].plot(t, ideal_square(t), color=TEAL, lw=1.3, zorder=2, label="目标方波")
    axes[0].plot(t, synthesize(t, tones), color=WAVE, lw=1.8, zorder=3, label="前 5 项")
    axes[0].set_title("时域：橙色是前 5 项相加，青色是要靠近的方波")
    axes[0].set_xlabel("时间 t（秒）")
    axes[0].set_xlim(0, 2)
    axes[0].set_xticks([0, 0.5, 1.0, 1.5, 2.0])
    axes[0].set_ylim(-1.8, 1.9)
    axes[0].legend(frameon=False, loc="upper right")
    dress_time(axes[0])
    draw_bars(
        axes[1],
        tones,
        freq_max=11,
        ticks=[0, 1, 3, 5, 7, 9],
        title="频域：这 5 根柱就是橙色曲线的配方",
    )
    save(fig, "ch03_square.png")


def mine_tones():
    raw = [(F1, A1), (F2, A2), (F3, A3)]
    tones = []
    for frequency, amplitude in raw:
        if not isinstance(frequency, (int, float)) or not isinstance(amplitude, (int, float)):
            raise SystemExit("频率和振幅请写成数字。")
        if isinstance(frequency, bool) or isinstance(amplitude, bool):
            raise SystemExit("频率和振幅请写成数字。")
        if amplitude < 0 or amplitude > 1:
            raise SystemExit("振幅请用 0 到 1。写成 0 表示关掉这根柱子。")
        if frequency < 0:
            raise SystemExit("频率请写成 0 或正数。写成 0 表示关掉这根柱子。")
        tones.append((float(frequency), float(amplitude)))
    return tones


def plot_mine(tones):
    chosen = active(tones)
    if chosen:
        fastest = max(f for f, _a in chosen)
        seconds = min(1.0, 4.0 / fastest)
    else:
        seconds = 1.0
    count = 5000
    t = np.linspace(0, seconds, count, endpoint=False)
    y = synthesize(t, tones)
    use_ms = seconds <= 0.05
    x = t * (1000 if use_ms else 1)
    fig, axes = plt.subplots(
        2, 1, figsize=(8.8, 6.5), layout="constrained", height_ratios=(1.25, 1)
    )
    axes[0].plot(x, y, color=INK, lw=2.0, zorder=3)
    if chosen:
        names = "、".join(f"{f:g} Hz" for f, _a in chosen)
        axes[0].set_title(f"时域：你的配方相加（{names}）")
    else:
        axes[0].set_title("时域：三根柱都关掉了，高度保持为 0")
    axes[0].set_xlabel("时间（毫秒）" if use_ms else "时间 t（秒）")
    axes[0].set_xlim(x[0], x[-1] if len(x) else 1)
    peak = float(np.max(np.abs(y))) if len(y) else 0.0
    limit = max(1.2, peak * 1.25)
    axes[0].set_ylim(-limit, limit)
    dress_time(axes[0])
    if chosen:
        freq_max = max(f for f, _a in chosen) * 1.35
        ticks = [0] + [f for f, _a in chosen]
    else:
        freq_max = 1.0
        ticks = [0, 1]
    draw_bars(axes[1], tones, freq_max=freq_max, ticks=ticks, title="频域：柱高就是你写下的振幅")
    save(fig, "ch03_mine.png")
    return seconds


def to_pcm(samples):
    scaled = np.round(np.clip(samples, -1, 1) * 32767)
    return scaled.astype("<i2")


def write_wav(path, samples, sample_rate=SAMPLE_RATE):
    import wave

    pcm = to_pcm(samples)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "w") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(int(sample_rate))
        handle.writeframes(pcm.tobytes())
    print(f"已保存 {path}")


def audible(tones, phases=None, seconds=1.0, gain=1.0):
    count = int(round(SAMPLE_RATE * seconds))
    t = np.arange(count) / SAMPLE_RATE
    safe = []
    safe_phase = []
    if phases is None:
        phases = [0.0] * len(tones)
    for (frequency, amplitude), phase in zip(tones, phases):
        if amplitude == 0 or frequency <= 0:
            continue
        if frequency >= SAMPLE_RATE / 2:
            print(f"提示：{frequency:g} Hz 不低于采样率的一半，这段声音里去掉了它。图上仍会画出这根柱。")
            continue
        safe.append((frequency, amplitude))
        safe_phase.append(phase)
    y = gain * synthesize(t, safe, safe_phase)
    peak = float(np.max(np.abs(y))) if len(y) else 0.0
    if peak > 0.95:
        y = y * (0.9 / peak)
        print("声音按峰值缩小到 0.9，避免削顶。图上的柱高仍是配方里的振幅。")
    return y


def write_audio(mine):
    a = 0.3
    write_wav(AUDIO / "ch03_two.wav", audible([(440.0, a), (880.0, a)]))
    write_wav(
        AUDIO / "ch03_phase90.wav",
        audible([(440.0, a), (880.0, a)], [0.0, np.pi / 2]),
    )
    write_wav(AUDIO / "ch03_square.wav", audible(square_tones(5, 220.0), gain=0.4))
    write_wav(AUDIO / "ch03_mine.wav", audible(mine))


def self_check():
    two = [(1.0, 1.0), (2.0, 1.0)]
    at_quarter = float(synthesize(np.array([0.25]), two)[0])
    if abs(at_quarter - 1) > 1e-9:
        raise SystemExit(f"自检失败：相位 0 时 0.25 秒的高度是 {at_quarter}")
    at_zero = float(synthesize(np.array([0.0]), two, [0.0, np.pi / 2])[0])
    if abs(at_zero - 1) > 1e-9:
        raise SystemExit(f"自检失败：相位 90° 时 0 秒的高度是 {at_zero}")
    at_quarter_90 = float(synthesize(np.array([0.25]), two, [0.0, np.pi / 2])[0])
    if abs(at_quarter_90) > 1e-9:
        raise SystemExit(f"自检失败：相位 90° 时 0.25 秒的高度是 {at_quarter_90}")
    bars = square_tones(5, 1.0)
    ratio = bars[0][1] / bars[1][1]
    if abs(ratio - 3) > 1e-9:
        raise SystemExit(f"自检失败：基频柱高除以 3 倍柱高是 {ratio}")
    if abs(bars[0][1] - 4 / np.pi) > 1e-9:
        raise SystemExit(f"自检失败：基频柱高是 {bars[0][1]}")
    print("自检通过：相位 0 时 0.25 秒高度为 1；第二条相位 90° 时，0 秒高度为 1，0.25 秒高度为 0。")
    print(f"方波基频柱高约为 {bars[0][1]:.3f}，3 倍柱高是它的三分之一。")


def main():
    tones = mine_tones()
    use_style()
    self_check()
    plot_two()
    plot_phase()
    plot_square()
    plot_mine(tones)
    write_audio(tones)
    print("课文图：ch03_two.png、ch03_phase.png、ch03_square.png")
    print("你的图：ch03_mine.png（只跟随 F1 A1、F2 A2、F3 A3）")


if __name__ == "__main__":
    main()
