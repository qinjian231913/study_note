"""第 9 篇：均衡与简单降噪。

在「傅里叶变换」文件夹里运行：

    python code/ch09_effects.py

课文对照图每次都按固定参数重画。
文件顶部的 GAIN_LOW / GAIN_MID / GAIN_HIGH 只影响 ch09_eq_mine.png 和 ch09_eq_mine.wav。

三频段均衡：把整段声音做 FFT，按低频 / 中频 / 高频乘不同增益，再 ifft 回去。
谱减法：用开头 0.5 秒估计噪声谱，从带噪频谱里减掉——会伤到语音细节。
变调只作概念提示，本篇不实现完整声码器。
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_style import INK, MUTED, TEAL, WAVE, finish, use_style

# ===== 改这里 =====
# 只影响 assets/figures/ch09_eq_mine.png 和 assets/audio/ch09_eq_mine.wav
# 增益是线性倍数：1.0 不变，0.3 变弱，2.0 变强。建议 0.05 到 4。
GAIN_LOW = 1.6
GAIN_MID = 0.7
GAIN_HIGH = 1.4
# ==================

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "assets" / "figures"
AUDIO = ROOT / "assets" / "audio"
SAMPLE_RATE = 44100
DURATION = 2.0

# 三频段分界（Hz）
LOW_MAX = 300.0
MID_MAX = 2000.0


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    path = FIG / name
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"已保存 {path}")


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


def make_eq_source(duration=DURATION, sample_rate=SAMPLE_RATE):
    """低频鼓感 + 中频元音感 + 高频嘶声，方便听出三段均衡。"""
    n = int(duration * sample_rate)
    t = np.arange(n) / sample_rate
    kick = 0.55 * np.sin(2 * np.pi * 110 * t) * np.exp(-3.5 * t)
    # 周期性轻敲，让低频更明显
    pulses = np.zeros(n)
    for start in np.arange(0, duration, 0.45):
        i0 = int(start * sample_rate)
        length = int(0.12 * sample_rate)
        i1 = min(n, i0 + length)
        local = np.arange(i1 - i0) / sample_rate
        pulses[i0:i1] += 0.7 * np.sin(2 * np.pi * 90 * local) * np.exp(-28 * local)
    voice = (
        0.28 * np.sin(2 * np.pi * 440 * t)
        + 0.18 * np.sin(2 * np.pi * 880 * t)
        + 0.10 * np.sin(2 * np.pi * 1320 * t)
    )
    voice *= 0.6 + 0.4 * np.sin(2 * np.pi * 2.0 * t)
    bright = 0.08 * np.sin(2 * np.pi * 4500 * t) + 0.05 * np.sin(2 * np.pi * 6200 * t)
    y = kick + pulses + voice + bright
    peak = np.max(np.abs(y))
    if peak > 0:
        y = 0.7 * y / peak
    fade = min(int(0.02 * sample_rate), n // 20)
    if fade > 0:
        y[:fade] *= np.linspace(0, 1, fade)
        y[-fade:] *= np.linspace(1, 0, fade)
    return t, y


def make_noisy_tone(duration=DURATION, sample_rate=SAMPLE_RATE, noise_amp=0.18):
    """稳定正弦 + 全程白噪声；开头 0.5 秒几乎只有噪声，便于估计。"""
    n = int(duration * sample_rate)
    t = np.arange(n) / sample_rate
    rng = np.random.default_rng(7)
    noise = noise_amp * rng.standard_normal(n)
    tone = np.zeros(n)
    # 0.5 秒之后再放进 660 Hz 音，让「噪声估计」尽量干净
    start = int(0.5 * sample_rate)
    tone[start:] = 0.42 * np.sin(2 * np.pi * 660 * t[start:])
    y = tone + noise
    peak = np.max(np.abs(y))
    if peak > 0:
        y = 0.85 * y / peak
    return t, y, tone, noise


def band_gains(freqs, gain_low, gain_mid, gain_high, low_max=LOW_MAX, mid_max=MID_MAX):
    gains = np.empty_like(freqs, dtype=float)
    gains[freqs < low_max] = gain_low
    mid = (freqs >= low_max) & (freqs < mid_max)
    gains[mid] = gain_mid
    gains[freqs >= mid_max] = gain_high
    return gains


def apply_eq(y, gain_low, gain_mid, gain_high, sample_rate=SAMPLE_RATE):
    spectrum = np.fft.rfft(y)
    freqs = np.fft.rfftfreq(len(y), d=1.0 / sample_rate)
    gains = band_gains(freqs, gain_low, gain_mid, gain_high)
    out = np.fft.irfft(spectrum * gains, n=len(y))
    return out, freqs, np.abs(spectrum), np.abs(spectrum * gains)


def spectral_subtraction(y, noise_seconds=0.5, sample_rate=SAMPLE_RATE, over_sub=1.0):
    """用开头一段估计噪声幅度谱，从整段幅度谱里减掉，相位保持原样。"""
    n_noise = min(len(y), int(noise_seconds * sample_rate))
    if n_noise < 16:
        raise ValueError("噪声估计段太短")
    spectrum = np.fft.rfft(y)
    mag = np.abs(spectrum)
    phase = np.angle(spectrum)
    noise_mag = np.abs(np.fft.rfft(y[:n_noise], n=len(y)))
    cleaned = np.maximum(mag - over_sub * noise_mag, 0.05 * mag)
    out = np.fft.irfft(cleaned * np.exp(1j * phase), n=len(y))
    return out, mag, cleaned, noise_mag


def positive_spectrum(freqs, mag, fmax=8000.0):
    mask = (freqs >= 0) & (freqs <= fmax)
    return freqs[mask], mag[mask]


def plot_eq_bands():
    """三频段分界示意。"""
    freqs = np.linspace(0, 8000, 4000)
    gains = band_gains(freqs, 1.8, 0.6, 1.5)
    fig, ax = plt.subplots(figsize=(8.8, 4.2), layout="constrained")
    ax.plot(freqs, gains, color=WAVE, lw=2.2, zorder=3)
    ax.axvline(LOW_MAX, color=TEAL, lw=1.2, ls=(0, (2, 3)), zorder=2)
    ax.axvline(MID_MAX, color=TEAL, lw=1.2, ls=(0, (2, 3)), zorder=2)
    ax.text(120, 1.95, "低频", color=TEAL, fontsize=12)
    ax.text(900, 0.75, "中频", color=TEAL, fontsize=12)
    ax.text(4200, 1.65, "高频", color=TEAL, fontsize=12)
    ax.set_xlim(0, 8000)
    ax.set_ylim(0, 2.3)
    ax.set_xlabel("频率（Hz）")
    ax.set_ylabel("增益（倍数）")
    ax.set_title(f"三频段：<{LOW_MAX:.0f}、{LOW_MAX:.0f}–{MID_MAX:.0f}、>{MID_MAX:.0f} Hz")
    finish(ax)
    save(fig, "ch09_eq_bands.png")


def plot_eq_demo():
    _t, y = make_eq_source()
    # 课文固定：抬低频、压中频、略抬高频
    out, freqs, before, after = apply_eq(y, 1.8, 0.55, 1.5)
    fig, axes = plt.subplots(2, 1, figsize=(8.8, 6.2), layout="constrained", sharex=True)
    fx, by = positive_spectrum(freqs, before)
    _fx, ay = positive_spectrum(freqs, after)
    axes[0].plot(fx, by, color=TEAL, lw=1.2, zorder=3)
    axes[0].set_ylabel("幅度")
    axes[0].set_title("均衡前（整段 FFT 的幅度）")
    axes[0].set_ylim(0, np.max(by) * 1.15)
    finish(axes[0])

    axes[1].plot(fx, ay, color=WAVE, lw=1.2, zorder=3)
    axes[1].axvline(LOW_MAX, color=MUTED, lw=1.0, ls=(0, (2, 3)))
    axes[1].axvline(MID_MAX, color=MUTED, lw=1.0, ls=(0, (2, 3)))
    axes[1].set_xlabel("频率（Hz）")
    axes[1].set_ylabel("幅度")
    axes[1].set_title("均衡后：低频抬高、中频压低、高频略抬")
    axes[1].set_xlim(0, 8000)
    axes[1].set_ylim(0, max(np.max(ay), np.max(by)) * 1.15)
    finish(axes[1])
    save(fig, "ch09_eq_spectrum.png")
    return y, out


def plot_eq_wave(y, out):
    t = np.arange(len(y)) / SAMPLE_RATE
    fig, axes = plt.subplots(2, 1, figsize=(8.8, 5.4), layout="constrained", sharex=True)
    axes[0].plot(t, y, color=TEAL, lw=0.8, zorder=3)
    axes[0].set_title("时域：均衡前")
    axes[0].set_ylim(-1.05, 1.05)
    finish(axes[0])
    axes[0].set_ylabel("强弱 y")

    axes[1].plot(t, out, color=WAVE, lw=0.8, zorder=3)
    axes[1].set_title("时域：均衡后（ifft 回去）")
    axes[1].set_xlabel("时间 t（秒）")
    axes[1].set_xlim(0, DURATION)
    axes[1].set_ylim(-1.05, 1.05)
    finish(axes[1])
    axes[1].set_ylabel("强弱 y")
    save(fig, "ch09_eq_wave.png")


def plot_denoise():
    _t, noisy, clean, _noise = make_noisy_tone()
    out, mag, cleaned, noise_mag = spectral_subtraction(noisy, noise_seconds=0.5)
    freqs = np.fft.rfftfreq(len(noisy), d=1.0 / SAMPLE_RATE)
    fig, axes = plt.subplots(2, 1, figsize=(8.8, 6.2), layout="constrained", sharex=True)
    fx, my = positive_spectrum(freqs, mag, fmax=4000)
    _fx, cy = positive_spectrum(freqs, cleaned, fmax=4000)
    _fx, ny = positive_spectrum(freqs, noise_mag, fmax=4000)
    axes[0].plot(fx, my, color=TEAL, lw=1.1, zorder=3, label="带噪")
    axes[0].plot(fx, ny, color=MUTED, lw=1.1, zorder=3, label="噪声估计（开头 0.5 秒）")
    axes[0].set_ylabel("幅度")
    axes[0].set_title("谱减法：先用开头估计噪声谱")
    axes[0].legend(frameon=False, loc="upper right")
    axes[0].set_ylim(0, np.max(my) * 1.15)
    finish(axes[0])

    axes[1].plot(fx, cy, color=WAVE, lw=1.2, zorder=3)
    axes[1].set_xlabel("频率（Hz）")
    axes[1].set_ylabel("幅度")
    axes[1].set_title("减去噪声谱之后：主峰还在，细小起伏被削掉")
    axes[1].set_xlim(0, 4000)
    axes[1].set_ylim(0, np.max(my) * 1.15)
    finish(axes[1])
    save(fig, "ch09_denoise_spectrum.png")

    t = np.arange(len(noisy)) / SAMPLE_RATE
    fig, axes = plt.subplots(3, 1, figsize=(8.8, 6.8), layout="constrained", sharex=True)
    axes[0].plot(t, noisy, color=TEAL, lw=0.7, zorder=3)
    axes[0].set_title("带噪波形")
    axes[0].axvline(0.5, color=MUTED, lw=1.0, ls=(0, (2, 3)))
    axes[0].text(0.52, 0.75, "噪声估计段", color=MUTED, fontsize=11)
    finish(axes[0])
    axes[0].set_ylabel("y")

    axes[1].plot(t, out, color=WAVE, lw=0.7, zorder=3)
    axes[1].set_title("谱减法之后（也伤到了音调周围的细节）")
    finish(axes[1])
    axes[1].set_ylabel("y")

    axes[2].plot(t, clean, color=INK, lw=0.8, zorder=3)
    axes[2].set_title("对照：只有干净音调（合成时单独留下的）")
    axes[2].set_xlabel("时间 t（秒）")
    axes[2].set_xlim(0, DURATION)
    finish(axes[2])
    axes[2].set_ylabel("y")
    for ax in axes:
        ax.set_ylim(-1.05, 1.05)
    save(fig, "ch09_denoise_wave.png")
    return noisy, out


def plot_pitch_teaser():
    """变调概念：频谱整体平移的示意，不是可听的完整算法。"""
    freqs = np.arange(0, 20)
    amps = np.zeros_like(freqs, dtype=float)
    amps[[2, 4, 6, 8]] = [1.0, 0.7, 0.45, 0.25]
    shifted = np.zeros_like(amps)
    shift = 2
    for i, a in enumerate(amps):
        j = i + shift
        if j < len(shifted):
            shifted[j] = a
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.8), layout="constrained", sharey=True)
    axes[0].bar(freqs, amps, color=TEAL, width=0.7, zorder=3)
    axes[0].set_title("原来的谐波位置")
    axes[0].set_xlabel("频率格子（示意）")
    axes[0].set_ylabel("振幅")
    finish(axes[0])
    axes[1].bar(freqs, shifted, color=WAVE, width=0.7, zorder=3)
    axes[1].set_title("整排往右挪：概念上的升高音调")
    axes[1].set_xlabel("频率格子（示意）")
    finish(axes[1])
    for ax in axes:
        ax.set_xlim(-0.5, 19.5)
        ax.set_ylim(0, 1.25)
    save(fig, "ch09_pitch_teaser.png")


def plot_eq_mine():
    _t, y = make_eq_source()
    out, freqs, before, after = apply_eq(y, GAIN_LOW, GAIN_MID, GAIN_HIGH)
    fx, by = positive_spectrum(freqs, before)
    _fx, ay = positive_spectrum(freqs, after)
    fig, ax = plt.subplots(figsize=(8.8, 4.4), layout="constrained")
    ax.plot(fx, by, color=TEAL, lw=1.1, zorder=2, label="均衡前")
    ax.plot(fx, ay, color=WAVE, lw=1.3, zorder=3, label="你的均衡后")
    ax.axvline(LOW_MAX, color=MUTED, lw=1.0, ls=(0, (2, 3)))
    ax.axvline(MID_MAX, color=MUTED, lw=1.0, ls=(0, (2, 3)))
    ax.set_xlim(0, 8000)
    ax.set_ylim(0, max(np.max(by), np.max(ay)) * 1.15)
    ax.set_xlabel("频率（Hz）")
    ax.set_ylabel("幅度")
    ax.set_title(
        f"你的增益：低 {GAIN_LOW:g} · 中 {GAIN_MID:g} · 高 {GAIN_HIGH:g}"
    )
    ax.legend(frameon=False, loc="upper right")
    finish(ax)
    save(fig, "ch09_eq_mine.png")
    return out


def write_audio(eq_before, eq_after, noisy, denoised, eq_mine):
    write_wav(AUDIO / "ch09_eq_before.wav", eq_before)
    write_wav(AUDIO / "ch09_eq_after.wav", normalize(eq_after))
    write_wav(AUDIO / "ch09_noisy.wav", noisy)
    write_wav(AUDIO / "ch09_denoised.wav", normalize(denoised))
    write_wav(AUDIO / "ch09_eq_mine.wav", normalize(eq_mine))


def normalize(y, peak=0.85):
    m = np.max(np.abs(y))
    if m <= 0:
        return y
    return peak * y / m


def self_check():
    _t, y = make_eq_source()
    out, freqs, before, after = apply_eq(y, 2.0, 1.0, 1.0)
    low = freqs < LOW_MAX
    high = freqs > MID_MAX
    if np.mean(after[low]) < np.mean(before[low]) * 1.5:
        raise SystemExit("自检失败：抬低频后低频幅度应明显变大")
    if abs(np.mean(after[high]) - np.mean(before[high])) / (np.mean(before[high]) + 1e-12) > 0.15:
        raise SystemExit("自检失败：只改低频时高频应大致不动")

    _t, noisy, _clean, _noise = make_noisy_tone()
    denoised, mag, cleaned, noise_mag = spectral_subtraction(noisy)
    # 噪声段能量应下降
    n0 = int(0.5 * SAMPLE_RATE)
    if np.std(denoised[:n0]) >= np.std(noisy[:n0]) * 0.95:
        raise SystemExit("自检失败：谱减法后开头噪声段应更安静")
    if np.max(cleaned) >= np.max(mag) * 0.999 and np.mean(cleaned) >= np.mean(mag) * 0.98:
        raise SystemExit("自检失败：谱减法几乎没有削掉幅度")
    print("自检通过：抬低频只动低频；谱减法会压低开头噪声段。")
    print("提醒：谱减法也会削掉音调周围的细节，听感常发干、有金属感。")


def main():
    for name, value in (("GAIN_LOW", GAIN_LOW), ("GAIN_MID", GAIN_MID), ("GAIN_HIGH", GAIN_HIGH)):
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not (0.05 <= float(value) <= 4.0):
            raise SystemExit(f"{name} 请用 0.05 到 4 之间的数。课文对照图不跟这些数走。")
    use_style()
    self_check()
    plot_eq_bands()
    eq_before, eq_after = plot_eq_demo()
    plot_eq_wave(eq_before, normalize(eq_after))
    noisy, denoised = plot_denoise()
    plot_pitch_teaser()
    eq_mine = plot_eq_mine()
    write_audio(eq_before, eq_after, noisy, denoised, eq_mine)
    print(
        "课文图：ch09_eq_bands.png、ch09_eq_spectrum.png、ch09_eq_wave.png、"
        "ch09_denoise_spectrum.png、ch09_denoise_wave.png、ch09_pitch_teaser.png"
    )
    print("你的图：ch09_eq_mine.png（只跟随 GAIN_LOW / GAIN_MID / GAIN_HIGH）")
    print(f"当前增益：低={GAIN_LOW:g} 中={GAIN_MID:g} 高={GAIN_HIGH:g}")


if __name__ == "__main__":
    main()
