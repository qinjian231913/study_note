"""第 8 篇：短时傅里叶与语谱图。

在「傅里叶变换」文件夹里运行：

    python code/ch08_stft.py

课文对照图每次都按固定参数重画。
文件顶部的 WINDOW_SAMPLES 只影响 ch08_mine.png 和 ch08_mine.wav。

短时傅里叶：把声音切成一小段一小段，每一段做一次频谱。
语谱图：横轴时间，纵轴频率，颜色表示这一小段里有多响。
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

from plot_style import INK, MUTED, TEAL, WAVE, finish, use_style

# ===== 改这里 =====
# 只影响 assets/figures/ch08_mine.png 和 assets/audio/ch08_mine.wav
# 窗长（采样点数）。44100 Hz 时：512 ≈ 12 ms，1024 ≈ 23 ms，4096 ≈ 93 ms。
WINDOW_SAMPLES = 1024
# ==================

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "assets" / "figures"
AUDIO = ROOT / "assets" / "audio"
SAMPLE_RATE = 44100
DURATION = 2.0

# 课文固定参数：短窗看清时间，长窗看清频率
SHORT_WIN = 256
LONG_WIN = 2048
DEFAULT_HOP_MS = 20.0

SPEC_CMAP = LinearSegmentedColormap.from_list(
    "textbook_spec",
    ["#fffdf8", "#f3e4c8", "#e8b87a", "#c24d28", "#0e6b66", "#1c1915"],
)


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


def make_chirp(duration=DURATION, sample_rate=SAMPLE_RATE, f0=200.0, f1=2000.0):
    """频率从 f0 扫到 f1 的线性调频。"""
    n = int(duration * sample_rate)
    t = np.arange(n) / sample_rate
    k = (f1 - f0) / duration
    phase = 2 * np.pi * (f0 * t + 0.5 * k * t * t)
    y = 0.45 * np.sin(phase)
    fade = min(int(0.02 * sample_rate), n // 10)
    if fade > 0:
        y[:fade] *= np.linspace(0, 1, fade)
        y[-fade:] *= np.linspace(1, 0, fade)
    return t, y


def make_vowel(duration=DURATION, sample_rate=SAMPLE_RATE):
    """基频缓慢变化、带几根谐波的「元音感」合成音。"""
    n = int(duration * sample_rate)
    t = np.arange(n) / sample_rate
    f0 = 140.0 + 40.0 * np.sin(2 * np.pi * 0.7 * t)
    phase = 2 * np.pi * np.cumsum(f0) / sample_rate
    # 振幅包络模拟共振峰：2、3、4 倍略强，更高次变弱
    weights = {1: 0.55, 2: 0.42, 3: 0.32, 4: 0.18, 5: 0.10, 6: 0.06}
    y = np.zeros(n, dtype=float)
    for harmonic, amp in weights.items():
        y = y + amp * np.sin(harmonic * phase)
    envelope = 0.55 + 0.45 * (0.5 + 0.5 * np.sin(2 * np.pi * 1.1 * t))
    y = 0.55 * y * envelope / max(np.max(np.abs(y)), 1e-9)
    fade = min(int(0.03 * sample_rate), n // 10)
    if fade > 0:
        y[:fade] *= np.linspace(0, 1, fade)
        y[-fade:] *= np.linspace(1, 0, fade)
    return t, y, f0


def hop_from_ms(ms, sample_rate=SAMPLE_RATE):
    return max(1, int(round(ms * 0.001 * sample_rate)))


def stft_magnitude(y, nperseg, hop, sample_rate=SAMPLE_RATE):
    """汉宁窗短时傅里叶：返回 (freqs, times, mag)，mag 形状为 频率×帧。"""
    y = np.asarray(y, dtype=float)
    nperseg = int(nperseg)
    hop = int(hop)
    if nperseg < 32:
        raise ValueError("窗长至少 32")
    if hop < 1:
        raise ValueError("步长至少 1")
    window = np.hanning(nperseg)
    if len(y) < nperseg:
        y = np.pad(y, (0, nperseg - len(y)))
    starts = np.arange(0, len(y) - nperseg + 1, hop)
    if len(starts) == 0:
        starts = np.array([0])
    spectra = np.empty((nperseg // 2 + 1, len(starts)), dtype=float)
    for i, start in enumerate(starts):
        frame = y[start : start + nperseg] * window
        spectra[:, i] = np.abs(np.fft.rfft(frame))
    freqs = np.fft.rfftfreq(nperseg, d=1.0 / sample_rate)
    times = (starts + nperseg / 2) / sample_rate
    return freqs, times, spectra


def db_scale(mag, floor_db=-60.0):
    ref = np.max(mag)
    if ref <= 0:
        return np.full_like(mag, floor_db)
    db = 20.0 * np.log10(np.maximum(mag / ref, 1e-12))
    return np.clip(db, floor_db, 0.0)


def draw_spectrogram(ax, freqs, times, mag, fmax=3000.0, title=""):
    db = db_scale(mag)
    fmask = freqs <= fmax
    extent = [float(times[0]), float(times[-1]), 0.0, fmax]
    image = db[fmask][::-1]
    ax.imshow(
        image,
        aspect="auto",
        extent=extent,
        cmap=SPEC_CMAP,
        vmin=-60,
        vmax=0,
        interpolation="nearest",
        zorder=2,
    )
    ax.set_xlabel("时间 t（秒）")
    ax.set_ylabel("频率（Hz）")
    ax.set_title(title)
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(0, fmax)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)


def plot_frames():
    """说明帧长、步长、重叠。"""
    sr = 1000.0
    t = np.arange(0, 0.12, 1.0 / sr)
    y = 0.8 * np.sin(2 * np.pi * 40 * t)
    nperseg = 40
    hop = 20
    fig, axes = plt.subplots(2, 1, figsize=(8.8, 5.8), layout="constrained", height_ratios=(1.2, 1))
    axes[0].plot(t * 1000, y, color=TEAL, lw=1.6, zorder=2)
    colors = (WAVE, TEAL, MUTED)
    for i, start in enumerate((0, hop, 2 * hop)):
        left = start / sr * 1000
        right = (start + nperseg) / sr * 1000
        color = colors[i % len(colors)]
        axes[0].axvspan(left, right, color=color, alpha=0.18, zorder=1)
        axes[0].axvline(left, color=color, lw=1.0, ls=(0, (2, 3)), zorder=3)
        axes[0].text((left + right) / 2, 1.05, f"第 {i + 1} 帧", ha="center", color=color, fontsize=11)
    axes[0].set_xlim(0, 120)
    axes[0].set_ylim(-1.25, 1.35)
    axes[0].set_xlabel("时间（毫秒）")
    axes[0].set_title("帧长 40 ms，步长 20 ms：相邻两帧重叠一半")
    finish(axes[0])
    axes[0].set_ylabel("强弱 y")

    labels = ("帧长", "步长", "重叠")
    values = (40, 20, 20)
    bars = axes[1].bar(labels, values, color=(WAVE, TEAL, MUTED), width=0.55, zorder=3)
    for bar, value in zip(bars, values):
        axes[1].text(bar.get_x() + bar.get_width() / 2, value + 1.5, f"{value} ms", ha="center", color=INK)
    axes[1].set_ylim(0, 55)
    axes[1].set_ylabel("毫秒")
    axes[1].set_title("大约每 20 ms 算一次频谱（步长）")
    finish(axes[1])
    save(fig, "ch08_frames.png")


def plot_chirp_spectrogram():
    _t, y = make_chirp()
    hop = hop_from_ms(DEFAULT_HOP_MS)
    freqs, times, mag = stft_magnitude(y, 1024, hop)
    fig, axes = plt.subplots(2, 1, figsize=(8.8, 6.4), layout="constrained", height_ratios=(1, 1.35))
    t_show = np.arange(len(y)) / SAMPLE_RATE
    axes[0].plot(t_show, y, color=TEAL, lw=0.8, zorder=3)
    axes[0].set_xlim(0, DURATION)
    axes[0].set_ylim(-0.7, 0.7)
    axes[0].set_xlabel("时间 t（秒）")
    axes[0].set_title("时域：频率从 200 Hz 扫到 2000 Hz 的 chirp")
    finish(axes[0])
    axes[0].set_ylabel("强弱 y")
    draw_spectrogram(axes[1], freqs, times, mag, fmax=2500, title="语谱图：亮线从低往高爬")
    save(fig, "ch08_chirp.png")


def plot_vowel_spectrogram():
    _t, y, f0 = make_vowel()
    hop = hop_from_ms(DEFAULT_HOP_MS)
    freqs, times, mag = stft_magnitude(y, 1024, hop)
    fig, axes = plt.subplots(2, 1, figsize=(8.8, 6.4), layout="constrained", height_ratios=(1, 1.35))
    t_show = np.arange(len(y)) / SAMPLE_RATE
    axes[0].plot(t_show, y, color=TEAL, lw=0.7, zorder=2)
    axes[0].plot(t_show, 0.0025 * f0, color=WAVE, lw=1.6, zorder=3, label="基频走势（放大显示）")
    axes[0].set_xlim(0, DURATION)
    axes[0].set_ylim(-0.85, 0.95)
    axes[0].set_xlabel("时间 t（秒）")
    axes[0].set_title("时域：基频在变的「元音感」谐波音")
    axes[0].legend(frameon=False, loc="upper right")
    finish(axes[0])
    axes[0].set_ylabel("强弱 y")
    draw_spectrogram(axes[1], freqs, times, mag, fmax=1200, title="语谱图：几条几乎平行的谐波带一起起伏")
    save(fig, "ch08_vowel.png")


def plot_compare():
    _t, y = make_chirp()
    hop_short = max(1, SHORT_WIN // 4)
    hop_long = max(1, LONG_WIN // 4)
    f_s, t_s, m_s = stft_magnitude(y, SHORT_WIN, hop_short)
    f_l, t_l, m_l = stft_magnitude(y, LONG_WIN, hop_long)
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.6), layout="constrained")
    ms_short = 1000 * SHORT_WIN / SAMPLE_RATE
    ms_long = 1000 * LONG_WIN / SAMPLE_RATE
    df_short = SAMPLE_RATE / SHORT_WIN
    df_long = SAMPLE_RATE / LONG_WIN
    draw_spectrogram(
        axes[0],
        f_s,
        t_s,
        m_s,
        fmax=2500,
        title=f"短窗 {SHORT_WIN} 点（≈{ms_short:.0f} ms）\n时间更清楚，频率格子约 {df_short:.0f} Hz",
    )
    draw_spectrogram(
        axes[1],
        f_l,
        t_l,
        m_l,
        fmax=2500,
        title=f"长窗 {LONG_WIN} 点（≈{ms_long:.0f} ms）\n频率更清楚，时间格子更粗",
    )
    save(fig, "ch08_compare.png")


def plot_mine():
    _t, y = make_chirp()
    nperseg = int(WINDOW_SAMPLES)
    hop = max(1, nperseg // 4)
    freqs, times, mag = stft_magnitude(y, nperseg, hop)
    fig, ax = plt.subplots(figsize=(8.8, 4.6), layout="constrained")
    ms = 1000 * nperseg / SAMPLE_RATE
    df = SAMPLE_RATE / nperseg
    draw_spectrogram(
        ax,
        freqs,
        times,
        mag,
        fmax=2500,
        title=f"你的窗长：{nperseg} 点（≈{ms:.1f} ms），频率格子约 {df:.1f} Hz",
    )
    save(fig, "ch08_mine.png")


def write_audio():
    _t, chirp = make_chirp()
    _t, vowel, _f0 = make_vowel()
    write_wav(AUDIO / "ch08_chirp.wav", chirp)
    write_wav(AUDIO / "ch08_vowel.wav", vowel)
    write_wav(AUDIO / "ch08_mine.wav", chirp)


def self_check():
    _t, y = make_chirp(duration=1.0)
    hop = hop_from_ms(20)
    freqs, times, mag = stft_magnitude(y, 1024, hop)
    if mag.shape[0] != len(freqs) or mag.shape[1] != len(times):
        raise SystemExit("自检失败：语谱图形状和频率/时间轴对不上")
    # chirp 末段能量应高于低频起始附近的中高频带
    early = mag[:, : max(1, mag.shape[1] // 8)]
    late = mag[:, -max(1, mag.shape[1] // 8) :]
    low = freqs < 400
    high = (freqs > 1400) & (freqs < 2200)
    if np.mean(early[low]) <= 0 or np.mean(late[high]) <= 0:
        raise SystemExit("自检失败：chirp 频谱能量异常")
    if np.mean(late[high]) < np.mean(early[high]) * 1.2:
        raise SystemExit("自检失败：chirp 末段高频没有明显变响")

    f_s, _t_s, m_s = stft_magnitude(y, SHORT_WIN, max(1, SHORT_WIN // 4))
    f_l, _t_l, m_l = stft_magnitude(y, LONG_WIN, max(1, LONG_WIN // 4))
    if (f_s[1] - f_s[0]) <= (f_l[1] - f_l[0]):
        raise SystemExit("自检失败：短窗的频率格子应更粗")
    if m_s.shape[1] <= m_l.shape[1]:
        raise SystemExit("自检失败：短窗在同一段声音上应有更多帧")
    print(
        f"自检通过：20 ms 步长约 {hop} 点；短窗频率格子约 {f_s[1] - f_s[0]:.1f} Hz，"
        f"长窗约 {f_l[1] - f_l[0]:.1f} Hz。"
    )


def main():
    if (
        not isinstance(WINDOW_SAMPLES, int)
        or isinstance(WINDOW_SAMPLES, bool)
        or not (64 <= WINDOW_SAMPLES <= 8192)
    ):
        raise SystemExit("WINDOW_SAMPLES 请用 64 到 8192 的整数。课文对照图不跟这个数走。")
    use_style()
    self_check()
    plot_frames()
    plot_chirp_spectrogram()
    plot_vowel_spectrogram()
    plot_compare()
    plot_mine()
    write_audio()
    print("课文图：ch08_frames.png、ch08_chirp.png、ch08_vowel.png、ch08_compare.png")
    print("你的图：ch08_mine.png（只跟随 WINDOW_SAMPLES）")
    print(f"当前 WINDOW_SAMPLES = {WINDOW_SAMPLES}（≈ {1000 * WINDOW_SAMPLES / SAMPLE_RATE:.1f} ms）")


if __name__ == "__main__":
    main()
