"""第 2 篇：把几条正弦加在一起。

在「傅里叶变换」文件夹里运行：

    python code/ch02_sum.py

课文对照图和课文音频每次都按固定参数重画、重写。
文件顶部的 N_TERMS 只影响 ch02_my_square.png 和 ch02_square_mine.wav。
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_style import INK, TEAL, WAVE, finish, use_style

# ===== 改这里 =====
# 只影响 assets/figures/ch02_my_square.png 和 assets/audio/ch02_square_mine.wav
# 1 表示只留基频，2 表示基频 + 3 倍，5 表示加到第 5 个奇数倍（9 倍）。
N_TERMS = 5
# ==================

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "assets" / "figures"
AUDIO = ROOT / "assets" / "audio"
SAMPLE_RATE = 44100
SQUARE_F0 = 220.0  # 方波声音的基频。20 项时最高谐波仍低于采样率的一半。


def sine(t, amplitude, frequency):
    """相位为 0 的正弦。t 可以是一个数，也可以是一整排时间。"""
    return amplitude * np.sin(2 * np.pi * frequency * t)


def square_partial(t, n_terms, frequency=1.0):
    """前 n_terms 个奇数倍正弦相加，用来靠近高度为 1 的方波。

    第 1 项是 1 倍频，第 2 项是 3 倍频，第 k 项是 (2k-1) 倍频，振幅按 1/(2k-1) 缩小。
    前面的 4/π 让很多项加完之后，平台高度靠近 1。
    """
    if n_terms < 1:
        raise ValueError("项数至少为 1")
    total = np.zeros_like(np.asarray(t, dtype=float), dtype=float)
    for k in range(int(n_terms)):
        harmonic = 2 * k + 1
        total = total + np.sin(2 * np.pi * harmonic * frequency * t) / harmonic
    return (4 / np.pi) * total


def ideal_square(t):
    """目标方波：每半个周期高度是 1，另外半个周期高度是 -1。"""
    frac = np.mod(t, 1.0)
    return np.where(frac < 0.5, 1.0, -1.0)


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    path = FIG / name
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"已保存 {path}")


def dress(ax, ylabel=True):
    finish(ax)
    if ylabel:
        ax.set_ylabel("强弱 y")


def plot_pair(name, f1, f2, seconds, x_scale, xlabel, mark=None):
    """上、中两条分别画出，下图把同一时刻的高度相加。x_scale 用来把秒换成图上的单位。"""
    t = np.linspace(0, seconds, 5000)
    y1 = sine(t, 1.0, f1)
    y2 = sine(t, 1.0, f2)
    x = t * x_scale
    fig, axes = plt.subplots(3, 1, figsize=(8.8, 7.4), layout="constrained", sharex=True)
    axes[0].plot(x, y1, color=TEAL, lw=2.0, zorder=3)
    axes[0].set_title(f"第一条：{f1:g} Hz，振幅 1")
    axes[1].plot(x, y2, color=WAVE, lw=2.0, zorder=3)
    axes[1].set_title(f"第二条：{f2:g} Hz，振幅 1")
    axes[2].plot(x, y1, color=TEAL, lw=1.3, zorder=2, label="第一条")
    axes[2].plot(x, y2, color=WAVE, lw=1.3, zorder=2, label="第二条")
    axes[2].plot(x, y1 + y2, color=INK, lw=2.3, zorder=3, label="相加")
    axes[2].set_title("同一时刻，两条的高度相加")
    axes[2].legend(frameon=False, ncol=3, loc="upper right")
    for ax, limit in zip(axes, (1.35, 1.35, 2.45)):
        dress(ax)
        ax.set_ylim(-limit, limit)
    if mark is not None:
        for ax in axes:
            ax.axvline(mark * x_scale, color="#8d8376", lw=1.0, ls=(0, (2, 3)), zorder=1)
    axes[2].set_xlabel(xlabel)
    axes[2].set_xlim(x[0], x[-1])
    save(fig, name)


def plot_square_steps():
    t = np.linspace(0, 2, 8000, endpoint=False)
    ideal = ideal_square(t)
    fig, axes = plt.subplots(2, 2, figsize=(8.8, 6.8), layout="constrained", sharex=True, sharey=True)
    specs = (
        (1, "1 项：只有基频"),
        (3, "3 项：加到 5 倍"),
        (8, "8 项：加到 15 倍"),
        (20, "20 项：加到 39 倍"),
    )
    for ax, (n_terms, title) in zip(axes.ravel(), specs):
        ax.plot(t, ideal, color=TEAL, lw=1.25, zorder=2)
        ax.plot(t, square_partial(t, n_terms), color=WAVE, lw=1.7, zorder=3)
        ax.set_title(title)
        dress(ax, ylabel=ax in (axes[0, 0], axes[1, 0]))
        ax.set_ylim(-1.65, 1.7)
    axes[1, 0].set_xlabel("时间 t（秒）")
    axes[1, 1].set_xlabel("时间 t（秒）")
    for ax in axes[1]:
        ax.set_xticks([0, 0.5, 1.0, 1.5, 2.0])
    save(fig, "ch02_square_steps.png")


def plot_recipe():
    harmonics = np.arange(1, 16, 2)
    amps = (4 / np.pi) / harmonics
    fig, ax = plt.subplots(figsize=(8.8, 3.9), layout="constrained")
    positions = np.arange(len(harmonics))
    ax.bar(positions, amps, color=WAVE, width=0.72, zorder=3)
    ax.set_xticks(positions, [f"{int(h)} 倍" for h in harmonics])
    ax.set_ylabel("这一项的振幅")
    ax.set_xlabel("频率是基频的几倍")
    ax.set_title("方波的配方：只留下奇数倍，越快的项越矮")
    ax.set_ylim(0, 1.55)
    ax.annotate(
        "基频",
        xy=(0, amps[0]),
        xytext=(0.55, 1.38),
        color=TEAL,
        fontsize=12,
        arrowprops=dict(arrowstyle="->", color=TEAL, lw=1.0),
    )
    finish(ax)
    ax.axhline(0, color="#5c564c", lw=0.8)
    save(fig, "ch02_recipe.png")


def plot_mine():
    t = np.linspace(0, 2, 8000, endpoint=False)
    fig, ax = plt.subplots(figsize=(8.8, 4.2), layout="constrained")
    ax.plot(t, ideal_square(t), color=TEAL, lw=1.25, zorder=2, label="目标方波")
    ax.plot(t, square_partial(t, N_TERMS), color=WAVE, lw=1.8, zorder=3, label=f"前 {N_TERMS} 项相加")
    highest = 2 * N_TERMS - 1
    ax.set_title(f"你的项数：{N_TERMS} 项，最高到 {highest} 倍")
    ax.set_xlabel("时间 t（秒）")
    ax.set_xlim(0, 2)
    ax.set_xticks([0, 0.5, 1.0, 1.5, 2.0])
    ax.set_ylim(-1.65, 1.7)
    ax.legend(frameon=False, loc="upper right")
    dress(ax)
    save(fig, "ch02_my_square.png")


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
    return pcm


def tone(frequency, amplitude, seconds=1.0):
    count = int(round(SAMPLE_RATE * seconds))
    t = np.arange(count) / SAMPLE_RATE
    return sine(t, amplitude, frequency)


def write_audio():
    a = 0.3
    write_wav(AUDIO / "ch02_440.wav", tone(440, a))
    write_wav(AUDIO / "ch02_880.wav", tone(880, a))
    write_wav(AUDIO / "ch02_440_plus_880.wav", tone(440, a) + tone(880, a))
    write_wav(AUDIO / "ch02_440_plus_660.wav", tone(440, a) + tone(660, a))

    count = SAMPLE_RATE
    t = np.arange(count) / SAMPLE_RATE
    gain = 0.4
    write_wav(AUDIO / "ch02_square_1.wav", gain * square_partial(t, 1, SQUARE_F0))
    write_wav(AUDIO / "ch02_square_8.wav", gain * square_partial(t, 8, SQUARE_F0))
    write_wav(AUDIO / "ch02_square_mine.wav", gain * square_partial(t, N_TERMS, SQUARE_F0))


def self_check():
    slow = sine(0.25, 1, 1) + sine(0.25, 1, 2)
    if abs(slow - 1) > 1e-9:
        raise SystemExit(f"自检失败：0.25 秒的相加结果是 {slow}")
    both = sine(0.125, 1, 1) + sine(0.125, 1, 2)
    expect = np.sin(np.pi / 4) + 1
    if abs(both - expect) > 1e-9:
        raise SystemExit(f"自检失败：0.125 秒的相加结果是 {both}")
    one = float(square_partial(np.array([0.25]), 1)[0])
    if abs(one - 4 / np.pi) > 1e-9:
        raise SystemExit(f"自检失败：1 项在 0.25 秒是 {one}")
    many = float(square_partial(np.array([0.25]), 80)[0])
    if abs(many - 1) > 0.02:
        raise SystemExit(f"自检失败：80 项在 0.25 秒是 {many}")
    print("自检通过：0.25 秒时 1 Hz 与 2 Hz 相加得 1；方波加到很多项后，平台靠近 1。")
    print(f"0.125 秒时，1 Hz 与 2 Hz 相加约为 {both:.3f}。")


def main():
    if not isinstance(N_TERMS, int) or not (1 <= N_TERMS <= 40):
        raise SystemExit("N_TERMS 请用 1 到 40 的整数。")
    highest = (2 * N_TERMS - 1) * SQUARE_F0
    if highest >= SAMPLE_RATE / 2:
        print("提示：最高谐波已经不低于采样率的一半，方波听起来会走样。请把 N_TERMS 改小。")

    use_style()
    self_check()
    plot_pair("ch02_sum_slow.png", 1, 2, 1.0, 1.0, "时间 t（秒）", mark=0.25)
    plot_pair("ch02_sum_fast.png", 440, 880, 0.01, 1000.0, "时间（毫秒）")
    plot_square_steps()
    plot_recipe()
    plot_mine()
    write_audio()
    print("课文图：ch02_sum_slow.png、ch02_sum_fast.png、ch02_square_steps.png、ch02_recipe.png")
    print("你的图：ch02_my_square.png（只跟随 N_TERMS）")


if __name__ == "__main__":
    main()
