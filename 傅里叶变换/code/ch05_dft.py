"""第 5 篇：离散傅里叶变换 DFT。

在「傅里叶变换」文件夹里运行：

    python code/ch05_dft.py

课文对照图每次都按固定参数重画。
文件顶部的 FREQUENCY 和 N_MINE 只影响 ch05_mine_*.png。

慢速 DFT 用双重循环手写，不调用 numpy.fft；再用 fft 对照误差。
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_style import INK, MUTED, TEAL, WAVE, finish, use_style

# ===== 改这里 =====
# 只影响 assets/figures/ch05_mine_mag.png 和 ch05_mine_wave.png
# FREQUENCY：合成余弦的频率（Hz）。请让它落在某个格子上，或故意偏离看泄漏。
# N_MINE：你自己那张图用的采样点数。建议 16、32、64。
FREQUENCY = 2.0
N_MINE = 32
SAMPLE_RATE_MINE = 32.0
# ==================

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "assets" / "figures"

# 课文固定例子：N=8，fs=8，正好一周期的 1 Hz 余弦
TEXTBOOK_N = 8
TEXTBOOK_FS = 8.0
TEXTBOOK_X = np.cos(2 * np.pi * np.arange(TEXTBOOK_N) / TEXTBOOK_N)


def slow_dft(x):
    """双重循环 DFT。X[k] = Σ x[n] (cos θ − j sin θ)，θ = 2π k n / N。

    不用 numpy.fft，方便对照「和旋转余弦/正弦有多齐」。
    """
    x = np.asarray(x, dtype=float)
    n_points = len(x)
    spectrum = np.zeros(n_points, dtype=complex)
    for k in range(n_points):
        total = 0.0 + 0.0j
        for n in range(n_points):
            angle = 2 * np.pi * k * n / n_points
            total = total + x[n] * (np.cos(angle) - 1j * np.sin(angle))
        spectrum[k] = total
    return spectrum


def bin_freqs(n_points, sample_rate):
    return np.arange(n_points) * sample_rate / n_points


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    path = FIG / name
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"已保存 {path}")


def dress_time(ax):
    finish(ax)
    ax.set_ylabel("强弱 y")


def plot_samples():
    n = np.arange(TEXTBOOK_N)
    t = n / TEXTBOOK_FS
    fig, ax = plt.subplots(figsize=(8.8, 4.2), layout="constrained")
    ax.stem(t, TEXTBOOK_X, linefmt=WAVE, markerfmt="o", basefmt=MUTED)
    for ti, yi in zip(t, TEXTBOOK_X):
        ax.text(ti, yi + 0.12 if yi >= 0 else yi - 0.18, f"{yi:.2f}", ha="center", fontsize=10, color=INK)
    ax.set_title("N = 8 个采样点：1 Hz 余弦，fs = 8 Hz，正好一圈")
    ax.set_xlabel("时间 t（秒）")
    ax.set_xlim(-0.05, 1.0)
    ax.set_ylim(-1.45, 1.45)
    dress_time(ax)
    save(fig, "ch05_samples.png")


def plot_bins():
    spectrum = slow_dft(TEXTBOOK_X)
    freqs = bin_freqs(TEXTBOOK_N, TEXTBOOK_FS)
    mag = np.abs(spectrum)
    fig, axes = plt.subplots(2, 1, figsize=(8.8, 6.4), layout="constrained")
    axes[0].stem(np.arange(TEXTBOOK_N), mag, linefmt=WAVE, markerfmt="o", basefmt=MUTED)
    axes[0].set_xticks(np.arange(TEXTBOOK_N))
    axes[0].set_xlabel("格子编号 k")
    axes[0].set_ylabel("|X[k]|")
    axes[0].set_title("幅度：能量在 k = 1 和 k = 7（一对共轭）")
    axes[0].set_ylim(0, 5.2)
    finish(axes[0])

    axes[1].stem(freqs, mag, linefmt=TEAL, markerfmt="o", basefmt=MUTED)
    axes[1].set_xticks(freqs)
    axes[1].set_xlabel("频率（Hz）= k × fs / N")
    axes[1].set_ylabel("|X[k]|")
    axes[1].set_title(f"分辨率 Δf = fs/N = {TEXTBOOK_FS / TEXTBOOK_N:g} Hz")
    axes[1].set_ylim(0, 5.2)
    finish(axes[1])
    save(fig, "ch05_bins.png")
    return spectrum


def plot_real_imag(spectrum):
    real = spectrum.real
    imag = spectrum.imag
    k = np.arange(TEXTBOOK_N)
    fig, axes = plt.subplots(2, 1, figsize=(8.8, 6.0), layout="constrained", sharex=True)
    axes[0].bar(k, real, color=WAVE, width=0.72, zorder=3)
    axes[0].set_ylabel("实部（对齐余弦）")
    axes[0].set_title("实部：和余弦同向为正，反向为负")
    axes[0].set_ylim(-5.2, 5.2)
    finish(axes[0])

    axes[1].bar(k, imag, color=TEAL, width=0.72, zorder=3)
    axes[1].set_xticks(k)
    axes[1].set_xlabel("格子编号 k")
    axes[1].set_ylabel("虚部（对齐正弦）")
    axes[1].set_title("虚部：这个例子几乎贴着 0（纯余弦）")
    axes[1].set_ylim(-1.2, 1.2)
    finish(axes[1])
    save(fig, "ch05_real_imag.png")


def reconstruct_from(spectrum, use_phase=True):
    """用幅度，可选是否保留相位，再 ifft 回时域。"""
    if use_phase:
        return np.fft.ifft(spectrum).real
    mag = np.abs(spectrum)
    return np.fft.ifft(mag).real


def phase_demo_signal():
    """带相位的 1 Hz 正弦。幅度谱与余弦相同，相位不同，零相位重建会对不上。"""
    return np.sin(2 * np.pi * np.arange(TEXTBOOK_N) / TEXTBOOK_N)


def plot_phase_matter():
    original = phase_demo_signal()
    spectrum = slow_dft(original)
    n = np.arange(TEXTBOOK_N)
    t = n / TEXTBOOK_FS
    true_wave = reconstruct_from(spectrum, use_phase=True)
    zero_phase = reconstruct_from(spectrum, use_phase=False)
    fig, axes = plt.subplots(2, 1, figsize=(8.8, 6.2), layout="constrained", sharex=True)
    axes[0].stem(t, true_wave, linefmt=WAVE, markerfmt="o", basefmt=MUTED)
    axes[0].plot(t, original, color=TEAL, lw=1.2, alpha=0.7, zorder=2)
    axes[0].set_title("用真实相位重建：回到原来的正弦")
    axes[0].set_ylim(-1.45, 1.45)
    dress_time(axes[0])

    axes[1].stem(t, zero_phase, linefmt=WAVE, markerfmt="o", basefmt=MUTED)
    axes[1].plot(t, original, color=TEAL, lw=1.2, alpha=0.55, zorder=2, label="原来的正弦")
    axes[1].set_title("只留幅度、相位全清零：变成更像余弦的形状")
    axes[1].set_xlabel("时间 t（秒）")
    axes[1].set_ylim(-1.45, 1.45)
    axes[1].legend(frameon=False, loc="upper right")
    dress_time(axes[1])
    save(fig, "ch05_phase.png")


def plot_mag_phase(spectrum):
    mag = np.abs(spectrum)
    phase = np.angle(spectrum)
    # 幅度接近 0 的格子相位无意义，画成 0 以免干扰
    phase = np.where(mag < 1e-9, 0.0, phase)
    k = np.arange(TEXTBOOK_N)
    fig, axes = plt.subplots(2, 1, figsize=(8.8, 6.0), layout="constrained", sharex=True)
    axes[0].stem(k, mag, linefmt=WAVE, markerfmt="o", basefmt=MUTED)
    axes[0].set_ylabel("|X[k]|")
    axes[0].set_title("幅度谱")
    axes[0].set_ylim(0, 5.2)
    finish(axes[0])

    axes[1].stem(k, phase, linefmt=TEAL, markerfmt="o", basefmt=MUTED)
    axes[1].set_xticks(k)
    axes[1].set_xlabel("格子编号 k")
    axes[1].set_ylabel("相位（弧度）")
    axes[1].set_title("相位谱（幅度近 0 的格子画成 0）")
    axes[1].set_ylim(-np.pi - 0.2, np.pi + 0.2)
    finish(axes[1])
    save(fig, "ch05_mag_phase.png")


def plot_mine():
    n_points = int(N_MINE)
    sample_rate = float(SAMPLE_RATE_MINE)
    t = np.arange(n_points) / sample_rate
    y = np.cos(2 * np.pi * float(FREQUENCY) * t)
    spectrum = np.fft.fft(y)
    freqs = bin_freqs(n_points, sample_rate)
    mag = np.abs(spectrum)

    fig, ax = plt.subplots(figsize=(8.8, 4.4), layout="constrained")
    ax.stem(t, y, linefmt=WAVE, markerfmt="o", basefmt=MUTED)
    ax.set_title(f"你的时域：f = {FREQUENCY:g} Hz，N = {n_points}，fs = {sample_rate:g}")
    ax.set_xlabel("时间 t（秒）")
    dress_time(ax)
    save(fig, "ch05_mine_wave.png")

    fig, ax = plt.subplots(figsize=(8.8, 4.4), layout="constrained")
    # 只画到 Nyquist，避免负频率镜像抢画面
    half = n_points // 2 + 1
    ax.stem(freqs[:half], mag[:half], linefmt=TEAL, markerfmt="o", basefmt=MUTED)
    ax.set_title(f"你的幅度谱：Δf = {sample_rate / n_points:g} Hz")
    ax.set_xlabel("频率（Hz）")
    ax.set_ylabel("|X[k]|")
    finish(ax)
    save(fig, "ch05_mine_mag.png")


def self_check():
    slow = slow_dft(TEXTBOOK_X)
    fast = np.fft.fft(TEXTBOOK_X)
    err = float(np.max(np.abs(slow - fast)))
    if err > 1e-9:
        raise SystemExit(f"自检失败：慢速 DFT 与 fft 最大误差 {err}")

    # N=4 手算表：x = [1, 0, -1, 0]
    x4 = np.array([1.0, 0.0, -1.0, 0.0])
    s4 = slow_dft(x4)
    expect = np.array([0.0, 2.0, 0.0, 2.0]) + 0j
    if np.max(np.abs(s4 - expect)) > 1e-9:
        raise SystemExit(f"自检失败：N=4 手算例是 {s4}")

    # 1 Hz 余弦在 N=8、fs=8 时应在 k=1 与 k=7
    mag = np.abs(slow)
    if mag[1] < 3.5 or mag[7] < 3.5:
        raise SystemExit(f"自检失败：余弦峰值不在 k=1,7：{mag}")
    if mag[0] > 1e-8 or mag[2] > 1e-8:
        raise SystemExit(f"自检失败：其它格子应接近 0：{mag}")

    sine = phase_demo_signal()
    sine_spec = slow_dft(sine)
    true_wave = reconstruct_from(sine_spec, True)
    zero_wave = reconstruct_from(sine_spec, False)
    if np.max(np.abs(true_wave - sine)) > 1e-9:
        raise SystemExit("自检失败：带相位重建对不上原信号")
    if np.max(np.abs(zero_wave - sine)) < 0.2:
        raise SystemExit("自检失败：零相位重建不应几乎等于原正弦")
    if np.max(np.abs(zero_wave - TEXTBOOK_X)) > 0.2:
        raise SystemExit("自检失败：零相位重建应接近同频余弦")

    print("自检通过：慢速 DFT 与 numpy.fft.fft 一致。")
    print(f"最大绝对误差 = {err:.3e}（应接近 0）。")
    print("N=4 手算例 [1,0,-1,0] → X = [0, 2, 0, 2]。")
    return err


def main():
    if not isinstance(N_MINE, int) or isinstance(N_MINE, bool) or N_MINE < 4:
        raise SystemExit("N_MINE 请用不小于 4 的整数。")
    if SAMPLE_RATE_MINE <= 0 or FREQUENCY < 0:
        raise SystemExit("SAMPLE_RATE_MINE 须为正，FREQUENCY 须非负。")
    use_style()
    err = self_check()
    plot_samples()
    spectrum = plot_bins()
    plot_real_imag(spectrum)
    plot_mag_phase(spectrum)
    plot_phase_matter()
    plot_mine()

    print("课文 N=8 的 X[k]（慢速 DFT）：")
    for k, value in enumerate(spectrum):
        print(f"  k={k}  Re={value.real: .4f}  Im={value.imag: .4f}  |X|={abs(value): .4f}")
    print(f"与 numpy.fft.fft 最大绝对误差：{err:.3e}")
    print("课文图：ch05_samples.png、ch05_bins.png、ch05_real_imag.png、ch05_mag_phase.png、ch05_phase.png")
    print("你的图：ch05_mine_wave.png、ch05_mine_mag.png（只跟随 FREQUENCY / N_MINE）")


if __name__ == "__main__":
    main()
