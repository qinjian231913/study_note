"""第 4 篇：周期信号的傅里叶级数。

在「傅里叶变换」文件夹里运行：

    python code/ch04_series.py

课文对照图每次都按固定参数重画。
文件顶部的 N_TERMS 只影响 ch04_my_gibbs.png 和 ch04_square_mine.wav。

系数用「和一个周期里的标准正弦相乘，再平均，再乘 2」算出来，
不从波形反推以外的黑箱里取。方波的结果应和第 2 篇的配方一致。
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_style import INK, MUTED, TEAL, WAVE, finish, use_style

# ===== 改这里 =====
# 只影响 assets/figures/ch04_my_gibbs.png 和 assets/audio/ch04_square_mine.wav
# 1 表示只有基频，5 表示加到 9 倍，20 表示加到 39 倍。
N_TERMS = 20
# ==================

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "assets" / "figures"
AUDIO = ROOT / "assets" / "audio"
SAMPLE_RATE = 44100
PERIOD = 1.0
POINTS = 8000


def square_partial(t, n_terms, frequency=1.0):
    """前 n_terms 个奇数倍正弦相加，系数是 (4/π) / 奇数。"""
    if n_terms < 1:
        raise ValueError("项数至少为 1")
    total = np.zeros_like(np.asarray(t, dtype=float), dtype=float)
    for k in range(int(n_terms)):
        harmonic = 2 * k + 1
        total = total + np.sin(2 * np.pi * harmonic * frequency * t) / harmonic
    return (4 / np.pi) * total


def ideal_square(t):
    frac = np.mod(t, PERIOD)
    return np.where(frac < 0.5, 1.0, -1.0)


def sample_period(kind, points=POINTS):
    """一个周期均匀切成 points 段，时间取每段中点。"""
    t = (np.arange(points) + 0.5) / points * PERIOD
    if kind == "square":
        y = ideal_square(t)
    elif kind == "sine":
        y = 0.8 * np.sin(2 * np.pi * t / PERIOD)
    else:
        raise ValueError("kind 只能是 square 或 sine")
    return t, np.asarray(y, dtype=float)


def coefficients(y, t, n_max):
    """a0、an、bn。an、bn 从 n=1 排到 n_max。

    先把波形和标准余弦或正弦相乘，在一个周期里平均，再乘 2。
    乘 2 是因为正弦自己和自己相乘，一个周期的平均是 1/2。
    """
    dt = float(t[1] - t[0])
    a0 = 2 * np.sum(y) * dt / PERIOD
    an = np.empty(n_max)
    bn = np.empty(n_max)
    for n in range(1, n_max + 1):
        angle = 2 * np.pi * n * t / PERIOD
        an[n - 1] = 2 * np.sum(y * np.cos(angle)) * dt / PERIOD
        bn[n - 1] = 2 * np.sum(y * np.sin(angle)) * dt / PERIOD
    return float(a0), an, bn


def formula_bn(n):
    if n % 2 == 0:
        return 0.0
    return 4 / (np.pi * n)


def first_peak(n_terms):
    """跳变之后、平台中段之前的最高点。"""
    span = min(0.25, 3.0 / max(int(n_terms), 1))
    t = np.linspace(1e-5, span, 100000)
    y = square_partial(t, n_terms)
    index = int(np.argmax(y))
    return float(t[index]), float(y[index])


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    path = FIG / name
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"已保存 {path}")


def dress_time(ax):
    finish(ax)
    ax.set_ylabel("强弱 y")


def plot_period():
    fig, axes = plt.subplots(2, 1, figsize=(8.8, 5.6), layout="constrained", sharex=True)
    specs = (
        (1.0, "1 Hz：第二秒重复第一秒，可以当基频"),
        (1.5, "1.5 Hz：第二秒是第一秒的上下翻转"),
    )
    t = np.linspace(0, 2, 4000)
    for ax, (frequency, title) in zip(axes, specs):
        y = np.sin(2 * np.pi * frequency * t)
        first = t <= 1
        second = t >= 1
        ax.plot(t[first], y[first], color=TEAL, lw=2.0, zorder=3, label="第一秒")
        ax.plot(t[second], y[second], color=WAVE, lw=2.0, zorder=3, label="第二秒")
        ax.axvline(1, color="#8d8376", lw=1.0, ls=(0, (2, 3)), zorder=1)
        ax.set_title(title)
        ax.set_ylim(-1.35, 1.45)
        ax.legend(frameon=False, ncol=2, loc="upper right")
        dress_time(ax)
    axes[1].set_xlabel("时间 t（秒）")
    axes[1].set_xlim(0, 2)
    save(fig, "ch04_period.png")


def plot_calibrate():
    t, y = sample_period("sine")
    _a0, an, bn = coefficients(y, t, 4)
    fig, axes = plt.subplots(
        2, 1, figsize=(8.8, 5.8), layout="constrained", height_ratios=(1.2, 1)
    )
    dense = np.linspace(0, 1, 1000, endpoint=False)
    axes[0].plot(dense, 0.8 * np.sin(2 * np.pi * dense), color=TEAL, lw=2.0, zorder=3)
    axes[0].set_title("时域：振幅 0.8 的 1 Hz 正弦")
    axes[0].set_xlabel("时间 t（秒）")
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(-1.2, 1.2)
    dress_time(axes[0])

    freqs = np.arange(1, 5)
    width = 0.35
    axes[1].bar(freqs - width / 2, an, width=width, color=TEAL, zorder=3, label="余弦系数 an")
    axes[1].bar(freqs + width / 2, bn, width=width, color=WAVE, zorder=3, label="正弦系数 bn")
    axes[1].set_xticks(freqs)
    axes[1].set_xlim(0.4, 4.6)
    axes[1].set_ylim(-0.2, 1.15)
    axes[1].set_xlabel("n（基频的倍数）")
    axes[1].set_ylabel("系数")
    axes[1].set_title(f"算出来：b1 = {bn[0]:.2f}，其余系数接近 0")
    axes[1].legend(frameon=False, ncol=2, loc="upper right")
    finish(axes[1])
    save(fig, "ch04_calibrate.png")


def plot_align():
    t = np.linspace(0, 1, 3000, endpoint=False)
    square = ideal_square(t)
    sines = (
        np.sin(2 * np.pi * t),
        np.sin(2 * np.pi * 2 * t),
    )
    titles = ("和 1 Hz 正弦", "和 2 Hz 正弦")
    fig, axes = plt.subplots(3, 2, figsize=(8.8, 7.4), layout="constrained", sharex=True)
    for col, (sine, title) in enumerate(zip(sines, titles)):
        product = square * sine
        mean = float(np.mean(product))
        axes[0, col].plot(t, square, color=TEAL, lw=1.6, zorder=3)
        axes[0, col].set_title(title)
        axes[0, col].set_ylim(-1.45, 1.55)
        axes[1, col].plot(t, sine, color=WAVE, lw=1.6, zorder=3)
        axes[1, col].set_ylim(-1.45, 1.55)
        axes[2, col].fill_between(
            t, product, 0, where=product >= 0, color=WAVE, alpha=0.45, interpolate=True, zorder=2
        )
        axes[2, col].fill_between(
            t, product, 0, where=product < 0, color=TEAL, alpha=0.45, interpolate=True, zorder=2
        )
        axes[2, col].plot(t, product, color=INK, lw=1.15, zorder=3)
        axes[2, col].axhline(mean, color="#8d8376", lw=1.0, ls=(0, (2, 3)), zorder=4)
        axes[2, col].text(0.02, 1.15, f"平均 {mean:.2f}，乘 2 得 {2 * mean:.2f}", color=INK, fontsize=10)
        axes[2, col].set_ylim(-1.45, 1.55)
        axes[2, col].set_xlabel("时间 t（秒）")
        for row in range(3):
            finish(axes[row, col])
    axes[0, 0].set_ylabel("方波")
    axes[1, 0].set_ylabel("标准波")
    axes[2, 0].set_ylabel("相乘")
    for ax in axes[:, 0]:
        ax.set_xlim(0, 1)
    save(fig, "ch04_align.png")


def plot_coeff():
    t, y = sample_period("square")
    _a0, an, bn = coefficients(y, t, 8)
    orders = np.arange(1, 9)
    expect = np.array([formula_bn(int(n)) for n in orders])
    fig, axes = plt.subplots(2, 1, figsize=(8.8, 6.2), layout="constrained", sharex=True)
    axes[0].bar(orders, an, color=TEAL, width=0.72, zorder=3)
    axes[0].set_ylim(-0.25, 0.25)
    axes[0].set_ylabel("an")
    axes[0].set_title("余弦系数：这个方波算出来都在 0 附近")
    finish(axes[0])

    axes[1].bar(orders, bn, color=WAVE, width=0.72, zorder=3, label="相乘再平均，再乘 2")
    axes[1].plot(orders, expect, color=INK, ls="none", marker="o", ms=6, zorder=4, label="第 2 篇配方 (4/π)/n，偶数倍为 0")
    axes[1].set_xticks(orders)
    axes[1].set_xlim(0.4, 8.6)
    axes[1].set_ylim(-0.08, 1.55)
    axes[1].set_xlabel("n（基频的倍数）")
    axes[1].set_ylabel("bn")
    axes[1].set_title("正弦系数：奇数倍和手写配方重合，偶数倍是 0")
    axes[1].legend(frameon=False, loc="upper right")
    finish(axes[1])
    save(fig, "ch04_coeff.png")
    return an, bn


def plot_gibbs():
    fig, axes = plt.subplots(
        2, 1, figsize=(8.8, 6.8), layout="constrained", height_ratios=(1, 1.15)
    )
    t_full = np.linspace(0, 1, 8000, endpoint=False)
    axes[0].plot(t_full, ideal_square(t_full), color=TEAL, lw=1.2, zorder=2, label="目标方波")
    axes[0].plot(t_full, square_partial(t_full, 80), color=WAVE, lw=1.3, zorder=3, label="前 80 项")
    axes[0].set_title("一整秒：平台已经很平，跳变旁边仍有一小截过冲")
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(-1.6, 1.65)
    axes[0].legend(frameon=False, ncol=2, loc="upper right")
    dress_time(axes[0])

    zoom = np.linspace(0, 0.1, 4000, endpoint=False)
    styles = (
        (5, MUTED, "5 项"),
        (20, WAVE, "20 项"),
        (80, INK, "80 项"),
    )
    for n_terms, color, label in styles:
        axes[1].plot(zoom, square_partial(zoom, n_terms), color=color, lw=1.8, zorder=3, label=label)
    axes[1].axhline(1, color=TEAL, lw=1.0, ls=(0, (2, 3)), zorder=1)
    axes[1].axhline(1.18, color="#8d8376", lw=1.0, ls=(0, (2, 3)), zorder=1)
    axes[1].text(0.072, 1.22, "约 1.18", color="#5c564c", fontsize=11)
    axes[1].text(0.072, 0.78, "平台 1", color=TEAL, fontsize=11)
    axes[1].set_title("放大跳变之后的 0.1 秒：最高点停在大约 1.18，只是越来越靠近 0")
    axes[1].set_xlabel("时间 t（秒）")
    axes[1].set_xlim(0, 0.1)
    axes[1].set_ylim(0, 1.5)
    axes[1].legend(frameon=False, ncol=3, loc="lower right")
    dress_time(axes[1])
    save(fig, "ch04_gibbs.png")


def plot_mine():
    span = 0.5 if N_TERMS <= 2 else min(0.5, max(0.08, 6.0 / (4 * N_TERMS)))
    t = np.linspace(0, span, 4000, endpoint=False)
    y = square_partial(t, N_TERMS)
    peak_t, peak_y = first_peak(N_TERMS)
    fig, ax = plt.subplots(figsize=(8.8, 4.4), layout="constrained")
    ax.axhline(1, color=TEAL, lw=1.1, ls=(0, (2, 3)), zorder=1, label="平台 1")
    ax.axhline(1.18, color="#8d8376", lw=1.0, ls=(0, (2, 3)), zorder=1, label="约 1.18")
    ax.plot(t, y, color=WAVE, lw=2.0, zorder=3, label=f"前 {N_TERMS} 项")
    if peak_t <= span:
        ax.plot(peak_t, peak_y, marker="o", color=INK, ms=5, zorder=4)
    highest = 2 * N_TERMS - 1
    ax.set_title(f"你的项数：{N_TERMS} 项，最高到 {highest} 倍。最高点约 {peak_y:.2f}，在 {peak_t:.3f} 秒")
    ax.set_xlabel("时间 t（秒）")
    ax.set_xlim(0, span)
    ax.set_ylim(-0.2, 1.6)
    ax.legend(frameon=False, ncol=3, loc="lower right")
    dress_time(ax)
    save(fig, "ch04_my_gibbs.png")
    return peak_t, peak_y


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


def square_audio(n_terms, frequency=220.0):
    count = SAMPLE_RATE
    t = np.arange(count) / SAMPLE_RATE
    highest = (2 * n_terms - 1) * frequency
    if highest >= SAMPLE_RATE / 2:
        print(f"提示：{n_terms} 项的最高谐波 {highest:g} Hz 不低于采样率的一半，这段声音不写。")
        return None
    return 0.4 * square_partial(t, n_terms, frequency)


def write_audio():
    write_wav(AUDIO / "ch04_square_5.wav", square_audio(5))
    write_wav(AUDIO / "ch04_square_40.wav", square_audio(40))
    mine = square_audio(N_TERMS)
    if mine is not None:
        write_wav(AUDIO / "ch04_square_mine.wav", mine)


def self_check():
    t, sine = sample_period("sine")
    _a0, an, bn = coefficients(sine, t, 3)
    if abs(bn[0] - 0.8) > 1e-3 or abs(an[0]) > 1e-3 or abs(bn[1]) > 1e-3:
        raise SystemExit(f"自检失败：0.8 正弦的系数是 a1={an[0]} b1={bn[0]} b2={bn[1]}")

    t, square = sample_period("square")
    a0, an, bn = coefficients(square, t, 8)
    if abs(a0) > 1e-3:
        raise SystemExit(f"自检失败：方波 a0 是 {a0}")
    if abs(bn[0] - 4 / np.pi) > 1e-3:
        raise SystemExit(f"自检失败：b1 是 {bn[0]}")
    if abs(bn[1]) > 1e-3:
        raise SystemExit(f"自检失败：b2 是 {bn[1]}")
    if abs(bn[2] - 4 / (3 * np.pi)) > 1e-3:
        raise SystemExit(f"自检失败：b3 是 {bn[2]}")
    if np.max(np.abs(an)) > 1e-3:
        raise SystemExit(f"自检失败：余弦系数最大是 {np.max(np.abs(an))}")

    peaks = {}
    for n_terms in (5, 20, 80):
        _tp, yp = first_peak(n_terms)
        peaks[n_terms] = yp
        if abs(yp - 1.18) > 0.02:
            raise SystemExit(f"自检失败：{n_terms} 项的最高点是 {yp}")
    if abs(peaks[20] - peaks[80]) > 0.01:
        raise SystemExit(f"自检失败：20 项和 80 项的最高点差得太多：{peaks}")
    mid = float(square_partial(np.array([0.25]), 80)[0])
    if abs(mid - 1) > 0.02:
        raise SystemExit(f"自检失败：80 项在 0.25 秒是 {mid}")
    print("自检通过：0.8 正弦的 b1 约为 0.80；方波 b1 约为 1.273，b2 约为 0。")
    print(
        f"5、20、80 项在跳变旁的最高点分别约为 {peaks[5]:.2f}、{peaks[20]:.2f}、{peaks[80]:.2f}。"
    )


def main():
    if not isinstance(N_TERMS, int) or isinstance(N_TERMS, bool) or not (1 <= N_TERMS <= 40):
        raise SystemExit("N_TERMS 请用 1 到 40 的整数。课文里的 80 项图不跟这个数走。")
    use_style()
    self_check()
    plot_period()
    plot_calibrate()
    plot_align()
    an, bn = plot_coeff()
    plot_gibbs()
    plot_mine()
    write_audio()
    print("方波前 8 个系数（相乘再平均，再乘 2）：")
    for n in range(1, 9):
        print(f"  n={n}  an={an[n - 1]: .4f}  bn={bn[n - 1]: .4f}  配方={formula_bn(n): .4f}")
    print("课文图：ch04_period.png、ch04_calibrate.png、ch04_align.png、ch04_coeff.png、ch04_gibbs.png")
    print("你的图：ch04_my_gibbs.png（只跟随 N_TERMS）")


if __name__ == "__main__":
    main()
