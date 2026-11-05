"""第 6 篇：FFT、逆变换与窗。

在「傅里叶变换」文件夹里运行：

    python code/ch06_fft.py

课文对照图与示范 wav 每次都按固定参数重画。
文件顶部的 CUTOFF_BIN 只影响 ch06_mine_*.png / ch06_filtered_mine.wav。

慢速 DFT 只在较小的 N 上计时，避免 N=4096 双重循环拖太久。
"""

from pathlib import Path
import time
import wave

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_style import INK, MUTED, TEAL, WAVE, finish, use_style

# ===== 改这里 =====
# 只影响你的低通结果：ch06_mine_spectrum.png、ch06_mine_wave.png、ch06_filtered_mine.wav
# CUTOFF_BIN：保留的正频率格子上限（含）。更大 → 更亮；更小 → 更闷。
# 课文示范用的截断格子不读这个数。
CUTOFF_BIN = 400
# ==================

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "assets" / "figures"
AUDIO = ROOT / "assets" / "audio"

FS = 8000
N_LEAK = 256
TONE = 440.0
DURATION = 1.0
SAMPLE_RATE_AUDIO = 8000


def slow_dft(x):
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


def hann(n_points):
    """汉宁窗：一点公式。两端压到 0，中间为 1。"""
    n = np.arange(n_points)
    return 0.5 - 0.5 * np.cos(2 * np.pi * n / (n_points - 1))


def to_pcm(samples):
    scaled = np.round(np.clip(samples, -1, 1) * 32767)
    return scaled.astype("<i2")


def write_wav(path, samples, sample_rate=SAMPLE_RATE_AUDIO):
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "w") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(int(sample_rate))
        handle.writeframes(to_pcm(samples).tobytes())
    print(f"已保存 {path}")


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    path = FIG / name
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"已保存 {path}")


def tone(frequency, n_points, sample_rate=FS, phase=0.0):
    t = np.arange(n_points) / sample_rate
    return np.cos(2 * np.pi * frequency * t + phase)


def positive_mag(spectrum, sample_rate):
    n_points = len(spectrum)
    half = n_points // 2 + 1
    freqs = np.arange(half) * sample_rate / n_points
    mag = np.abs(spectrum[:half]) * 2 / n_points
    mag[0] = np.abs(spectrum[0]) / n_points
    if n_points % 2 == 0:
        mag[-1] = np.abs(spectrum[half - 1]) / n_points
    return freqs, mag


def plot_timing():
    """慢速 DFT 只比到 N=1024；再画一张概念条：FFT 在更大 N 上仍很快。"""
    sizes = [64, 256, 1024]
    slow_ms = []
    fft_ms = []
    rng = np.random.default_rng(0)
    for n_points in sizes:
        x = rng.standard_normal(n_points)
        t0 = time.perf_counter()
        _ = slow_dft(x)
        slow_ms.append((time.perf_counter() - t0) * 1000)
        t0 = time.perf_counter()
        for _ in range(20):
            _ = np.fft.fft(x)
        fft_ms.append((time.perf_counter() - t0) * 1000 / 20)

    # 额外：N=4096 只测 fft，并在图注里说明慢速 DFT 会很久
    x_big = rng.standard_normal(4096)
    t0 = time.perf_counter()
    for _ in range(30):
        _ = np.fft.fft(x_big)
    fft_4096_ms = (time.perf_counter() - t0) * 1000 / 30

    fig, ax = plt.subplots(figsize=(8.8, 4.6), layout="constrained")
    xpos = np.arange(len(sizes))
    width = 0.36
    ax.bar(xpos - width / 2, slow_ms, width=width, color=WAVE, zorder=3, label="慢速双重循环 DFT")
    ax.bar(xpos + width / 2, fft_ms, width=width, color=TEAL, zorder=3, label="numpy.fft.fft")
    ax.set_xticks(xpos)
    ax.set_xticklabels([f"N={n}" for n in sizes])
    ax.set_ylabel("耗时（毫秒）")
    ax.set_title(f"同一 DFT：慢速随 N² 涨。N=4096 时 fft 约 {fft_4096_ms:.2f} ms（未跑慢速）")
    ax.legend(frameon=False, loc="upper left")
    finish(ax)
    save(fig, "ch06_timing.png")
    return sizes, slow_ms, fft_ms, fft_4096_ms


def plot_roundtrip():
    n_points = 1024
    x = tone(440.0, n_points, sample_rate=8000, phase=0.3)
    y = np.fft.ifft(np.fft.fft(x)).real
    err = float(np.max(np.abs(x - y)))
    t = np.arange(120) / 8000
    fig, axes = plt.subplots(2, 1, figsize=(8.8, 5.8), layout="constrained")
    axes[0].plot(t, x[:120], color=TEAL, lw=2.0, zorder=3, label="原信号")
    axes[0].plot(t, y[:120], color=WAVE, lw=1.2, ls=(0, (2, 2)), zorder=4, label="fft → ifft")
    axes[0].set_title(f"来回一趟：最大绝对误差 {err:.2e}")
    axes[0].set_xlim(0, t[-1])
    axes[0].legend(frameon=False, loc="upper right")
    finish(axes[0])
    axes[0].set_ylabel("强弱 y")

    diff = (y - x)[:120]
    axes[1].plot(t, diff, color=INK, lw=1.4, zorder=3)
    axes[1].set_title("差：原信号减去重建（应贴着 0）")
    axes[1].set_xlabel("时间 t（秒）")
    axes[1].set_xlim(0, t[-1])
    span = max(float(np.max(np.abs(diff))), 1e-15) * 1.5
    axes[1].set_ylim(-span, span)
    finish(axes[1])
    axes[1].set_ylabel("差")
    save(fig, "ch06_roundtrip.png")
    return err


def plot_leakage():
    # 不对齐：440 Hz，fs=8000，N=256 → 分辨率 31.25 Hz，440/31.25 = 14.08
    leak = tone(TONE, N_LEAK, FS)
    # 对齐：选最接近 440 的格子频率
    bin_k = int(round(TONE * N_LEAK / FS))
    aligned_f = bin_k * FS / N_LEAK
    aligned = tone(aligned_f, N_LEAK, FS)

    freqs_l, mag_l = positive_mag(np.fft.fft(leak), FS)
    freqs_a, mag_a = positive_mag(np.fft.fft(aligned), FS)

    fig, axes = plt.subplots(2, 1, figsize=(8.8, 6.4), layout="constrained", sharex=True)
    axes[0].plot(freqs_l, mag_l, color=WAVE, lw=1.6, zorder=3)
    axes[0].axvline(TONE, color=MUTED, lw=1.0, ls=(0, (2, 3)), zorder=2)
    axes[0].set_xlim(0, 1000)
    axes[0].set_ylabel("幅度")
    axes[0].set_title(f"440 Hz 未落在格子上：能量摊到旁边（Δf = {FS / N_LEAK:g} Hz）")
    finish(axes[0])

    axes[1].plot(freqs_a, mag_a, color=TEAL, lw=1.6, zorder=3)
    axes[1].axvline(aligned_f, color=MUTED, lw=1.0, ls=(0, (2, 3)), zorder=2)
    axes[1].set_xlim(0, 1000)
    axes[1].set_xlabel("频率（Hz）")
    axes[1].set_ylabel("幅度")
    axes[1].set_title(f"改成对齐频率 {aligned_f:.2f} Hz（k = {bin_k}）：主峰更尖")
    finish(axes[1])
    save(fig, "ch06_leakage.png")
    return aligned_f, bin_k


def plot_window():
    leak = tone(TONE, N_LEAK, FS)
    windowed = leak * hann(N_LEAK)
    freqs_r, mag_r = positive_mag(np.fft.fft(leak), FS)
    freqs_w, mag_w = positive_mag(np.fft.fft(windowed), FS)

    fig, axes = plt.subplots(2, 1, figsize=(8.8, 6.6), layout="constrained")
    n = np.arange(N_LEAK)
    axes[0].plot(n, hann(N_LEAK), color=TEAL, lw=2.0, zorder=3)
    axes[0].set_title("汉宁窗：w[n] = 0.5 − 0.5 cos(2π n /(N−1))")
    axes[0].set_xlabel("采样点 n")
    axes[0].set_ylabel("窗值")
    axes[0].set_xlim(0, N_LEAK - 1)
    finish(axes[0])

    axes[1].plot(freqs_r, mag_r, color=MUTED, lw=1.3, zorder=2, label="矩形截断（未加窗）")
    axes[1].plot(freqs_w, mag_w, color=WAVE, lw=1.8, zorder=3, label="乘汉宁窗")
    axes[1].set_xlim(200, 700)
    axes[1].set_xlabel("频率（Hz）")
    axes[1].set_ylabel("幅度")
    axes[1].set_title("泄漏变矮，主瓣变宽")
    axes[1].legend(frameon=False, loc="upper right")
    finish(axes[1])
    save(fig, "ch06_window.png")


def lowpass_ifft(samples, cutoff_bin):
    """fft → 清掉高于 cutoff 的正频率（及对应负频率）→ ifft。"""
    spectrum = np.fft.fft(samples)
    n_points = len(spectrum)
    filtered = spectrum.copy()
    # 正频率：k = cutoff_bin+1 … N/2
    # 负频率镜像：k = N-cutoff_bin … N-1 中高于 cutoff 的部分
    if cutoff_bin < n_points // 2:
        filtered[cutoff_bin + 1 : n_points - cutoff_bin] = 0
    return np.fft.ifft(filtered).real, spectrum, filtered


def plot_filter_pipeline():
    count = int(SAMPLE_RATE_AUDIO * DURATION)
    # 亮一点的音色：基频 + 若干谐波，方便听出「切掉高频变闷」
    t = np.arange(count) / SAMPLE_RATE_AUDIO
    bright = (
        0.55 * np.sin(2 * np.pi * 220 * t)
        + 0.28 * np.sin(2 * np.pi * 440 * t)
        + 0.18 * np.sin(2 * np.pi * 880 * t)
        + 0.12 * np.sin(2 * np.pi * 1320 * t)
        + 0.08 * np.sin(2 * np.pi * 1760 * t)
    )
    bright = 0.9 * bright / np.max(np.abs(bright))

    textbook_cutoff = 40  # 正频率保留到 k=40 → 40*8000/8000 = 40 Hz? Wait N = count = 8000
    # Δf = fs/N = 8000/8000 = 1 Hz, so cutoff_bin=40 means 40 Hz — too low!
    # For 1 second at 8000 Hz, N=8000, bins are 1 Hz each.
    # To keep up to ~600 Hz: cutoff_bin = 600
    textbook_cutoff = 600

    dull, full_spec, filt_spec = lowpass_ifft(bright, textbook_cutoff)
    dull = 0.9 * dull / np.max(np.abs(dull))

    freqs_f, mag_f = positive_mag(full_spec, SAMPLE_RATE_AUDIO)
    freqs_d, mag_d = positive_mag(filt_spec, SAMPLE_RATE_AUDIO)

    fig, axes = plt.subplots(2, 1, figsize=(8.8, 6.4), layout="constrained")
    axes[0].plot(freqs_f, mag_f, color=MUTED, lw=1.2, zorder=2, label="原频谱")
    axes[0].plot(freqs_d, mag_d, color=WAVE, lw=1.6, zorder=3, label=f"保留 k≤{textbook_cutoff}")
    axes[0].axvline(textbook_cutoff, color=TEAL, lw=1.0, ls=(0, (2, 3)), zorder=1)
    axes[0].set_xlim(0, 2200)
    axes[0].set_ylabel("幅度")
    axes[0].set_title("流水线：fft → 清掉高频格子 → ifft")
    axes[0].legend(frameon=False, loc="upper right")
    finish(axes[0])

    show = slice(0, 400)
    axes[1].plot(t[show], bright[show], color=TEAL, lw=1.3, zorder=2, label="原来")
    axes[1].plot(t[show], dull[show], color=WAVE, lw=1.5, zorder=3, label="切掉高频后")
    axes[1].set_xlabel("时间 t（秒）")
    axes[1].set_ylabel("强弱 y")
    axes[1].set_title("时域：细节变少，听起来更闷")
    axes[1].legend(frameon=False, loc="upper right")
    finish(axes[1])
    save(fig, "ch06_filter.png")

    write_wav(AUDIO / "ch06_bright.wav", bright)
    write_wav(AUDIO / "ch06_dull.wav", dull)
    return textbook_cutoff, bright


def plot_mine(bright):
    dull, full_spec, filt_spec = lowpass_ifft(bright, int(CUTOFF_BIN))
    peak = np.max(np.abs(dull))
    if peak > 0:
        dull = 0.9 * dull / peak
    freqs_f, mag_f = positive_mag(full_spec, SAMPLE_RATE_AUDIO)
    freqs_d, mag_d = positive_mag(filt_spec, SAMPLE_RATE_AUDIO)
    t = np.arange(len(bright)) / SAMPLE_RATE_AUDIO

    fig, ax = plt.subplots(figsize=(8.8, 4.4), layout="constrained")
    ax.plot(freqs_f, mag_f, color=MUTED, lw=1.2, zorder=2, label="原频谱")
    ax.plot(freqs_d, mag_d, color=WAVE, lw=1.6, zorder=3, label=f"你的截断 k≤{CUTOFF_BIN}")
    ax.axvline(CUTOFF_BIN, color=TEAL, lw=1.0, ls=(0, (2, 3)), zorder=1)
    ax.set_xlim(0, 2200)
    ax.set_xlabel("频率（Hz）")
    ax.set_ylabel("幅度")
    ax.set_title(f"你的截断：CUTOFF_BIN = {CUTOFF_BIN}（约 {CUTOFF_BIN} Hz）")
    ax.legend(frameon=False, loc="upper right")
    finish(ax)
    save(fig, "ch06_mine_spectrum.png")

    fig, ax = plt.subplots(figsize=(8.8, 4.2), layout="constrained")
    show = slice(0, 400)
    ax.plot(t[show], bright[show], color=TEAL, lw=1.2, zorder=2, label="原来")
    ax.plot(t[show], dull[show], color=WAVE, lw=1.5, zorder=3, label="你的结果")
    ax.set_xlabel("时间 t（秒）")
    ax.set_ylabel("强弱 y")
    ax.set_title("你的时域对照")
    ax.legend(frameon=False, loc="upper right")
    finish(ax)
    save(fig, "ch06_mine_wave.png")
    write_wav(AUDIO / "ch06_filtered_mine.wav", dull)


def self_check():
    x = np.array([1.0, 0.0, -1.0, 0.0])
    err = float(np.max(np.abs(slow_dft(x) - np.fft.fft(x))))
    if err > 1e-9:
        raise SystemExit(f"自检失败：慢速 DFT 误差 {err}")

    n_points = 512
    signal = tone(125.0, n_points, sample_rate=8000)
    back = np.fft.ifft(np.fft.fft(signal)).real
    rt = float(np.max(np.abs(signal - back)))
    if rt > 1e-10:
        raise SystemExit(f"自检失败：往返误差 {rt}")

    w = hann(9)  # 奇数点时中点正好是 1
    if abs(w[0]) > 1e-12 or abs(w[-1]) > 1e-12 or abs(float(w[4]) - 1.0) > 1e-9:
        raise SystemExit(f"自检失败：汉宁窗端点应为 0、中点应为 1：{w}")

    # 对齐音应比未对齐音在旁瓣上更干净：主峰旁平均幅度更小
    leak = tone(440.0, 256, 8000)
    bin_k = int(round(440.0 * 256 / 8000))
    aligned = tone(bin_k * 8000 / 256, 256, 8000)
    mag_l = np.abs(np.fft.fft(leak))
    mag_a = np.abs(np.fft.fft(aligned))
    # 旁瓣：去掉主峰附近 ±2 格
    def side(mag, peak_k):
        mask = np.ones(len(mag), dtype=bool)
        for kk in range(peak_k - 2, peak_k + 3):
            if 0 <= kk < len(mag):
                mask[kk] = False
            mirror = len(mag) - kk
            if 0 <= mirror < len(mag):
                mask[mirror] = False
        return float(np.mean(mag[mask]))

    if side(mag_a, bin_k) >= side(mag_l, bin_k):
        raise SystemExit("自检失败：对齐音的旁瓣应更小")

    print("自检通过：fft 再 ifft 往返误差接近 0；汉宁窗端点为 0。")
    print(f"慢速 DFT 与 fft 对照误差 {err:.3e}；往返误差 {rt:.3e}。")
    return rt


def main():
    if not isinstance(CUTOFF_BIN, int) or isinstance(CUTOFF_BIN, bool) or CUTOFF_BIN < 1:
        raise SystemExit("CUTOFF_BIN 请用 ≥ 1 的整数。一秒 8000 点时，它约等于截止频率（Hz）。")
    if CUTOFF_BIN >= SAMPLE_RATE_AUDIO // 2:
        raise SystemExit("CUTOFF_BIN 应小于 Nyquist 对应的格子（默认 < 4000）。")
    use_style()
    rt = self_check()
    sizes, slow_ms, fft_ms, fft_4096_ms = plot_timing()
    plot_roundtrip()
    aligned_f, bin_k = plot_leakage()
    plot_window()
    textbook_cutoff, bright = plot_filter_pipeline()
    plot_mine(bright)

    print("计时（毫秒）：")
    for n_points, s, f in zip(sizes, slow_ms, fft_ms):
        print(f"  N={n_points}: 慢速 {s:.2f}，fft {f:.3f}")
    print(f"  N=4096: 只测 fft ≈ {fft_4096_ms:.3f} ms（慢速双重循环未跑）")
    print(f"往返最大误差：{rt:.3e}")
    print(f"对齐频率约 {aligned_f:.2f} Hz（k={bin_k}）")
    print(f"课文低通截断 k≤{textbook_cutoff}；你的 CUTOFF_BIN={CUTOFF_BIN}")
    print("课文图：ch06_timing.png、ch06_roundtrip.png、ch06_leakage.png、ch06_window.png、ch06_filter.png")
    print("你的图/声：ch06_mine_spectrum.png、ch06_mine_wave.png、ch06_filtered_mine.wav")


if __name__ == "__main__":
    main()
