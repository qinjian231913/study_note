"""第 7 篇：真实声音的频谱。

在「傅里叶变换」文件夹里运行：

    python code/ch07_audio.py

课文用合成 wav（不依赖外部采样）。对照图按固定配方重画。
文件顶部的 HARMONICS_MINE / WAV_PATH_MINE 只影响 ch07_mine_*.png 与可选试听。

依赖：numpy、matplotlib；读写 wav 用标准库 wave。
"""

from pathlib import Path
import wave

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_style import INK, MUTED, TEAL, WAVE, finish, use_style

# ===== 改这里 =====
# 只影响 ch07_mine_spectrum.png、ch07_mine_wave.png、ch07_timbre_mine.wav
# HARMONICS_MINE：相对基频的倍数 -> 振幅。改比例就能换「亮 / 闷」。
# WAV_PATH_MINE：若写成已有 wav 的路径（相对项目根或绝对路径），则 mine 图改读这个文件；
#               留空字符串 "" 表示仍用上面的泛音配方合成。
F0_MINE = 220.0
HARMONICS_MINE = {
    1: 0.70,
    2: 0.45,
    3: 0.30,
    4: 0.18,
    5: 0.12,
}
WAV_PATH_MINE = ""
# ==================

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "assets" / "figures"
AUDIO = ROOT / "assets" / "audio"
SAMPLE_RATE = 16000
DURATION = 1.0


def to_pcm(samples):
    scaled = np.round(np.clip(samples, -1, 1) * 32767)
    return scaled.astype("<i2")


def write_wav(path, samples, sample_rate=SAMPLE_RATE):
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "w") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(int(sample_rate))
        handle.writeframes(to_pcm(samples).tobytes())
    print(f"已保存 {path}")


def read_wav_mono(path):
    """读 wav，转单声道浮点，并归一化到峰值约 0.9。"""
    path = Path(path)
    with wave.open(str(path), "r") as handle:
        channels = handle.getnchannels()
        width = handle.getsampwidth()
        rate = handle.getframerate()
        n_frames = handle.getnframes()
        raw = handle.readframes(n_frames)
    if width != 2:
        raise SystemExit(f"本课只示范 16 位 PCM，读到的是 {8 * width} 位：{path}")
    data = np.frombuffer(raw, dtype="<i2").astype(np.float64)
    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1)
    peak = np.max(np.abs(data))
    if peak < 1:
        samples = data / 32767.0
    else:
        samples = data / peak
    peak = np.max(np.abs(samples))
    if peak > 0:
        samples = 0.9 * samples / peak
    return samples, int(rate)


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    path = FIG / name
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"已保存 {path}")


def synthesize(harmonics, f0, duration=DURATION, sample_rate=SAMPLE_RATE):
    """harmonics: {倍数: 振幅}。"""
    count = int(round(sample_rate * duration))
    t = np.arange(count) / sample_rate
    y = np.zeros(count, dtype=float)
    for multiple, amp in harmonics.items():
        if amp == 0 or multiple <= 0:
            continue
        y = y + float(amp) * np.sin(2 * np.pi * float(multiple) * float(f0) * t)
    peak = np.max(np.abs(y))
    if peak > 0:
        y = 0.9 * y / peak
    return y, sample_rate


def magnitude_db(samples, sample_rate):
    spectrum = np.fft.rfft(samples)
    mag = np.abs(spectrum) * 2 / len(samples)
    mag[0] = np.abs(spectrum[0]) / len(samples)
    freqs = np.fft.rfftfreq(len(samples), d=1.0 / sample_rate)
    # 避免 log(0)
    db = 20 * np.log10(np.maximum(mag, 1e-12))
    return freqs, db, mag


def top_peaks(freqs, mag, count=5, min_hz=40.0, min_distance_hz=30.0):
    """在正频率里找若干局部峰。忽略相对主峰过矮的数值噪声。"""
    chosen = []
    floor = float(np.max(mag)) * 0.03
    # 简单非极大抑制
    for index in np.argsort(mag)[::-1]:
        height = float(mag[index])
        if height < floor:
            break
        f = float(freqs[index])
        if f < min_hz:
            continue
        if any(abs(f - prev) < min_distance_hz for prev, _h in chosen):
            continue
        chosen.append((f, height))
        if len(chosen) >= count:
            break
    chosen.sort(key=lambda item: item[0])
    return chosen


def plot_wave_spectrum(samples, sample_rate, stem, title_wave, title_spec, mark_peaks=True):
    t = np.arange(len(samples)) / sample_rate
    freqs, db, mag = magnitude_db(samples, sample_rate)
    peaks = top_peaks(freqs, mag, count=5) if mark_peaks else []

    fig, axes = plt.subplots(2, 1, figsize=(8.8, 6.4), layout="constrained")
    show_n = min(len(samples), int(sample_rate * 0.05))
    axes[0].plot(t[:show_n], samples[:show_n], color=WAVE, lw=1.5, zorder=3)
    axes[0].set_title(title_wave)
    axes[0].set_xlabel("时间 t（秒）")
    axes[0].set_ylabel("强弱 y")
    finish(axes[0])

    f_max = min(4000.0, sample_rate / 2)
    mask = freqs <= f_max
    axes[1].plot(freqs[mask], db[mask], color=TEAL, lw=1.4, zorder=3)
    for frequency, height in peaks:
        if frequency > f_max:
            continue
        # 在 dB 曲线上标峰
        idx = int(np.argmin(np.abs(freqs - frequency)))
        axes[1].plot(frequency, db[idx], marker="o", color=WAVE, ms=5, zorder=4)
        axes[1].annotate(
            f"{frequency:.0f}",
            (frequency, db[idx]),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=10,
            color=INK,
        )
    axes[1].set_xlim(0, f_max)
    axes[1].set_ylim(max(-80, float(np.min(db[mask])) - 5), float(np.max(db[mask])) + 8)
    axes[1].set_xlabel("频率（Hz）")
    axes[1].set_ylabel("幅度（dB）")
    axes[1].set_title(title_spec)
    finish(axes[1])
    save(fig, f"{stem}.png")
    return peaks


def plot_sequence_limitation():
    """先 220 后 330：整段 FFT 看不出先后。"""
    rate = SAMPLE_RATE
    half = int(rate * 0.5)
    t1 = np.arange(half) / rate
    t2 = np.arange(half) / rate
    part1 = 0.7 * np.sin(2 * np.pi * 220 * t1)
    part2 = 0.7 * np.sin(2 * np.pi * 330 * t2)
    sequential = np.concatenate([part1, part2])
    simultaneous = 0.5 * np.sin(2 * np.pi * 220 * np.arange(rate) / rate) + 0.5 * np.sin(
        2 * np.pi * 330 * np.arange(rate) / rate
    )
    sequential = 0.9 * sequential / np.max(np.abs(sequential))
    simultaneous = 0.9 * simultaneous / np.max(np.abs(simultaneous))

    write_wav(AUDIO / "ch07_seq_notes.wav", sequential, rate)
    write_wav(AUDIO / "ch07_chord_ref.wav", simultaneous, rate)

    freqs_s, db_s, _m = magnitude_db(sequential, rate)
    freqs_c, db_c, _m2 = magnitude_db(simultaneous, rate)

    fig, axes = plt.subplots(2, 1, figsize=(8.8, 6.2), layout="constrained", sharex=True)
    mask = freqs_s <= 1000
    axes[0].plot(freqs_s[mask], db_s[mask], color=WAVE, lw=1.5, zorder=3)
    axes[0].set_title("先 220 Hz 再 330 Hz：整段一次 FFT")
    axes[0].set_ylabel("幅度（dB）")
    finish(axes[0])

    axes[1].plot(freqs_c[mask], db_c[mask], color=TEAL, lw=1.5, zorder=3)
    axes[1].set_title("两音同时响：整段一次 FFT（峰的位置很像）")
    axes[1].set_xlabel("频率（Hz）")
    axes[1].set_ylabel("幅度（dB）")
    finish(axes[1])
    for ax in axes:
        ax.set_xlim(0, 1000)
        ax.axvline(220, color=MUTED, lw=0.9, ls=(0, (2, 3)), zorder=1)
        ax.axvline(330, color=MUTED, lw=0.9, ls=(0, (2, 3)), zorder=1)
    save(fig, "ch07_order_blind.png")


def make_textbook_audio():
    pure, rate = synthesize({1: 1.0}, 220.0)
    t = np.arange(int(rate * DURATION)) / rate
    chord = (
        0.55 * np.sin(2 * np.pi * 220 * t)
        + 0.45 * np.sin(2 * np.pi * 275 * t)
        + 0.40 * np.sin(2 * np.pi * 330 * t)
    )
    chord = 0.9 * chord / np.max(np.abs(chord))

    bright, _r = synthesize({1: 0.55, 2: 0.40, 3: 0.32, 4: 0.22, 5: 0.16, 6: 0.10}, 220.0)
    dull, _r = synthesize({1: 0.85, 2: 0.25, 3: 0.08}, 220.0)

    write_wav(AUDIO / "ch07_pure_220.wav", pure, rate)
    write_wav(AUDIO / "ch07_chord.wav", chord, rate)
    write_wav(AUDIO / "ch07_bright.wav", bright, rate)
    write_wav(AUDIO / "ch07_dull.wav", dull, rate)
    return pure, chord, bright, dull, rate


def plot_timbre_compare(bright, dull, rate):
    freqs_b, db_b, mag_b = magnitude_db(bright, rate)
    freqs_d, db_d, mag_d = magnitude_db(dull, rate)
    fig, ax = plt.subplots(figsize=(8.8, 4.8), layout="constrained")
    mask = freqs_b <= 3000
    ax.plot(freqs_b[mask], db_b[mask], color=WAVE, lw=1.6, zorder=3, label="泛音多（亮）")
    ax.plot(freqs_d[mask], db_d[mask], color=TEAL, lw=1.6, zorder=3, label="泛音少（闷）")
    ax.set_xlim(0, 3000)
    ax.set_xlabel("频率（Hz）")
    ax.set_ylabel("幅度（dB）")
    ax.set_title("同一基频 220 Hz，泛音比例不同 → 音色不同")
    ax.legend(frameon=False, loc="upper right")
    finish(ax)
    save(fig, "ch07_timbre.png")
    return top_peaks(freqs_b, mag_b, 6), top_peaks(freqs_d, mag_d, 4)


def plot_mine():
    if str(WAV_PATH_MINE).strip():
        path = Path(WAV_PATH_MINE)
        if not path.is_absolute():
            path = ROOT / path
        samples, rate = read_wav_mono(path)
        title = f"你的 wav：{path.name}"
        write_wav(AUDIO / "ch07_timbre_mine.wav", samples, rate)
    else:
        samples, rate = synthesize(HARMONICS_MINE, F0_MINE)
        title = f"你的配方：f0 = {F0_MINE:g} Hz"
        write_wav(AUDIO / "ch07_timbre_mine.wav", samples, rate)

    peaks = plot_wave_spectrum(
        samples,
        rate,
        "ch07_mine_spectrum",
        title_wave=f"{title}（前 50 ms）",
        title_spec="你的幅度谱（dB），圆点是较高的峰",
        mark_peaks=True,
    )
    # 再单独存一张时域放大，文件名按需求
    t = np.arange(len(samples)) / rate
    fig, ax = plt.subplots(figsize=(8.8, 3.8), layout="constrained")
    n = min(len(samples), int(rate * 0.05))
    ax.plot(t[:n], samples[:n], color=WAVE, lw=1.6, zorder=3)
    ax.set_title(f"{title} · 时域")
    ax.set_xlabel("时间 t（秒）")
    ax.set_ylabel("强弱 y")
    finish(ax)
    save(fig, "ch07_mine_wave.png")
    return peaks


def self_check():
    pure, rate = synthesize({1: 1.0}, 220.0)
    freqs, _db, mag = magnitude_db(pure, rate)
    peak_f = float(freqs[np.argmax(mag[1:]) + 1])
    if abs(peak_f - 220.0) > 2.0:
        raise SystemExit(f"自检失败：纯音峰在 {peak_f} Hz")

    # 读写往返
    path = AUDIO / "_ch07_selfcheck.tmp.wav"
    write_wav(path, pure, rate)
    back, rate2 = read_wav_mono(path)
    path.unlink(missing_ok=True)
    if rate2 != rate:
        raise SystemExit("自检失败：采样率读回不一致")
    if abs(np.max(np.abs(back)) - 0.9) > 0.05:
        raise SystemExit(f"自检失败：归一化峰值 {np.max(np.abs(back))}")

    bright, _r = synthesize({1: 0.5, 2: 0.4, 3: 0.3, 4: 0.2}, 220.0)
    dull, _r = synthesize({1: 0.9, 2: 0.15}, 220.0)
    _fb, _dbb, mag_b = magnitude_db(bright, rate)
    _fd, _dbd, mag_d = magnitude_db(dull, rate)
    # 亮音在 880 Hz 附近应强于闷音
    idx = int(np.argmin(np.abs(freqs - 880)))
    if mag_b[idx] <= mag_d[idx]:
        raise SystemExit("自检失败：亮音色在 880 Hz 应更强")

    print("自检通过：220 Hz 纯音峰位置正确；wav 读回并归一化。")
    print(f"纯音主峰约 {peak_f:.1f} Hz。")
    return peak_f


def main():
    if F0_MINE <= 0:
        raise SystemExit("F0_MINE 须为正。")
    use_style()
    peak_f = self_check()
    pure, chord, bright, dull, rate = make_textbook_audio()

    peaks_pure = plot_wave_spectrum(
        pure,
        rate,
        "ch07_pure",
        title_wave="纯音 220 Hz（前 50 ms）",
        title_spec="幅度谱（dB）：一根主峰",
    )
    peaks_chord = plot_wave_spectrum(
        chord,
        rate,
        "ch07_chord",
        title_wave="和弦：220 + 275 + 330 Hz（前 50 ms）",
        title_spec="幅度谱（dB）：多根峰",
    )
    plot_timbre_compare(bright, dull, rate)
    plot_wave_spectrum(
        bright,
        rate,
        "ch07_bright",
        title_wave="亮音色：同一 220 Hz，泛音多（前 50 ms）",
        title_spec="亮音色的幅度谱（dB）",
    )
    plot_wave_spectrum(
        dull,
        rate,
        "ch07_dull",
        title_wave="闷音色：同一 220 Hz，泛音少（前 50 ms）",
        title_spec="闷音色的幅度谱（dB）",
    )
    plot_sequence_limitation()
    peaks_mine = plot_mine()

    print(f"纯音峰：{peaks_pure}")
    print(f"和弦峰：{peaks_chord}")
    print(f"你的峰：{peaks_mine}")
    print(f"自检主峰约 {peak_f:.1f} Hz")
    print("课文图：ch07_pure.png、ch07_chord.png、ch07_timbre.png、ch07_bright.png、ch07_dull.png、ch07_order_blind.png")
    print("你的图/声：ch07_mine_wave.png、ch07_mine_spectrum.png、ch07_timbre_mine.wav")


if __name__ == "__main__":
    main()
