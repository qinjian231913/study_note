"""第 1 篇：把声音和图像写成一串数字。

在「傅里叶变换」文件夹里运行：

    python code/ch01_signal.py

会得到 wav、渐变图和若干张课文图。文件顶部「改这里」只影响
note_mine.wav 和 ch01_my_zoom.png。440 Hz 的课文文件每次都按固定参数重写。
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_style import TEAL, WAVE, finish, use_style

# ===== 改这里 =====
# 只影响 assets/audio/note_mine.wav 和 assets/figures/ch01_my_zoom.png
FREQUENCY = 440.0  # 音高，单位 Hz。可改成 880
SAMPLE_RATE = 44100  # 采样率：每秒保存多少个整数。可改成 8000
DURATION = 1.0  # 秒。请保持在 10 秒以内
AMPLITUDE = 0.4  # 0 到 1。1 会顶满音量，0.4 更适合试听
# ==================

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "assets" / "figures"
AUDIO = ROOT / "assets" / "audio"
IMAGES = ROOT / "assets" / "images"

TEXTBOOK_PCM = [0, 821, 1639, 2450, 3252, 4041, 4815, 5569]


def make_wave(sample_rate, frequency, duration, amplitude):
    """生成 duration 秒的正弦。返回时间 t 和强弱 y，y 已经乘过振幅。"""
    count = int(round(sample_rate * duration))
    if count < 2:
        raise SystemExit("采样点数太少。请加大采样率或时长。")
    t = np.arange(count) / sample_rate
    y = amplitude * np.sin(2 * np.pi * frequency * t)
    return t, y


def to_pcm(samples):
    """把 -1 到 1 的小数写成 16 位整数。±1 对应 ±32767。"""
    scaled = np.round(np.clip(samples, -1, 1) * 32767)
    return scaled.astype("<i2")


def write_wav(path, samples, sample_rate):
    import wave

    pcm = to_pcm(samples)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "w") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(int(sample_rate))
        handle.writeframes(pcm.tobytes())
    return pcm


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    path = FIG / name
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"已保存 {path}")


def print_samples(y, pcm, sample_rate, count=8):
    print(f"{'n':>4} {'t(秒)':>12} {'小数 y':>10} {'文件中的整数':>12}")
    for i in range(count):
        print(f"{i:4d} {i / sample_rate:12.8f} {y[i]:10.6f} {int(pcm[i]):12d}")


def quantize(y, levels):
    """把 -1 到 1 四舍五入到 levels 个均匀等级。"""
    x = (y + 1) / 2
    q = np.round(x * (levels - 1)) / (levels - 1)
    return q * 2 - 1


def plot_sampling():
    t_line = np.linspace(0, 1, 2000)
    y_line = np.sin(2 * np.pi * 3 * t_line)
    sample_rate = 20
    t_s = np.arange(sample_rate) / sample_rate
    y_s = np.sin(2 * np.pi * 3 * t_s)

    fig, ax = plt.subplots(figsize=(8.8, 4.3), layout="constrained")
    ax.plot(t_line, y_line, color=TEAL, lw=2.0, label="光滑曲线，只作对照", zorder=2)
    stems = ax.stem(t_s, y_s, basefmt=" ")
    stems.stemlines.set_color(WAVE)
    stems.stemlines.set_linewidth(1.15)
    stems.markerline.set_color(WAVE)
    stems.markerline.set_markersize(7)
    stems.markerline.set_markeredgecolor("#fffdf8")
    stems.markerline.set_markeredgewidth(0.7)
    stems.markerline.set_zorder(4)
    stems.baseline.set_visible(False)
    finish(ax)
    ax.set_xlim(0, 1)
    ax.set_ylim(-1.45, 1.55)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("时间 t（秒）")
    ax.set_ylabel("强弱 y")
    ax.set_title("3 Hz 的波，每秒只留下 20 个橙点")
    ax.legend(frameon=False, loc="upper right")
    save(fig, "ch01_sampling.png")


def plot_quantization():
    t = np.linspace(0, 1, 4000)
    y = np.sin(2 * np.pi * 2 * t)
    fig, axes = plt.subplots(2, 1, figsize=(8.8, 6.6), layout="constrained", sharex=True)
    for ax, levels, title in (
        (axes[0], 5, "四舍五入到 5 个等级：光滑的波变成台阶"),
        (axes[1], 17, "17 个等级：台阶变密，形状靠近原来的波"),
    ):
        ax.plot(t, y, color=TEAL, lw=1.5, zorder=2)
        ax.plot(t, quantize(y, levels), color=WAVE, lw=2.0, zorder=3)
        finish(ax)
        ax.set_ylabel("强弱 y")
        ax.set_title(title)
        ax.set_ylim(-1.45, 1.45)
    axes[1].set_xlabel("时间 t（秒）")
    axes[1].set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    save(fig, "ch01_quant.png")


def plot_zoom(path_name, sample_rate, frequency, amplitude, title_prefix):
    """上面看大约 8 毫秒的整体，下面把开头十几个点放大。"""
    count = int(round(sample_rate * 1.0))
    n = np.arange(count)
    y = amplitude * np.sin(2 * np.pi * frequency * n / sample_rate)
    window = min(count, max(8, int(round(0.008 * sample_rate))))
    t_ms = n[:window] / sample_rate * 1000

    close = min(count, 12)
    t_close = n[:close] / sample_rate * 1000
    y_close = y[:close]

    fig, axes = plt.subplots(2, 1, figsize=(8.8, 6.8), layout="constrained")
    ax = axes[0]
    ax.plot(t_ms, y[:window], color=WAVE, lw=1.8, zorder=3)
    ax.axhline(amplitude, color=TEAL, lw=0.8, ls=(0, (3, 3)))
    ax.axhline(-amplitude, color=TEAL, lw=0.8, ls=(0, (3, 3)))
    finish(ax)
    ax.set_xlim(t_ms[0], t_ms[-1])
    ax.set_ylim(-max(0.55, amplitude * 1.4), max(0.55, amplitude * 1.45))
    ax.set_ylabel("强弱 y")
    ax.set_xlabel("时间（毫秒）")
    # 相位为 0 时，第一个波谷在 3/4 个周期处。把文字放在波谷正上方，避免挡住波峰。
    trough_ms = 750.0 / frequency
    if t_ms[0] < trough_ms < t_ms[-1]:
        label_x = trough_ms
    else:
        label_x = t_ms[len(t_ms) // 2]
    ax.text(
        label_x,
        amplitude,
        f"振幅 {amplitude:g}",
        color=TEAL,
        va="bottom",
        ha="center",
        fontsize=11,
    )
    ax.set_title(f"{title_prefix}：大约 8 毫秒里的 {window} 个采样挤成一条线")

    ax = axes[1]
    stems = ax.stem(t_close, y_close, basefmt=" ")
    stems.stemlines.set_color(WAVE)
    stems.markerline.set_color(WAVE)
    stems.markerline.set_markersize(6)
    stems.baseline.set_visible(False)
    finish(ax)
    ceiling = max(0.5, abs(amplitude))
    ax.set_ylim(-0.08, ceiling * 0.85)
    ax.set_xlabel("时间（毫秒）")
    ax.set_ylabel("强弱 y")
    ax.set_title(f"再放大开头 {close} 个点：点与点之间是空的")
    save(fig, path_name)


def plot_gradient():
    width, height = 480, 160
    ramp = np.linspace(0, 255, width)
    image = np.tile(ramp, (height, 1))

    IMAGES.mkdir(parents=True, exist_ok=True)
    raw_path = IMAGES / "gradient.png"
    plt.imsave(raw_path, image, cmap="gray", vmin=0, vmax=255)
    print(f"已保存 {raw_path}")

    fig, ax = plt.subplots(figsize=(8.8, 2.8), layout="constrained")
    shown = ax.imshow(image, cmap="gray", vmin=0, vmax=255, aspect="auto")
    ax.set_yticks([])
    ax.set_xticks([0, width - 1], ["左，亮度 0", "右，亮度 255"])
    ax.set_xlabel("左右位置")
    ax.set_title(f"一张 {height} 行 × {width} 列的灰度图，共 {height * width} 个亮度")
    bar = fig.colorbar(shown, ax=ax)
    bar.set_label("亮度")
    save(fig, "ch01_gradient.png")
    return image


def draw_cells(ax, data, title):
    ax.imshow(data, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    for row in range(data.shape[0]):
        for col in range(data.shape[1]):
            value = int(data[row, col])
            color = "#1c1915" if value >= 125 else "white"
            ax.text(col, row, str(value), ha="center", va="center", color=color, fontsize=12)
    ax.set_xticks(range(data.shape[1]), [f"列 {i}" for i in range(data.shape[1])])
    ax.set_yticks(range(data.shape[0]), [f"行 {i}" for i in range(data.shape[0])])
    ax.tick_params(length=0)
    ax.set_title(title)


def plot_grids():
    columns = np.arange(6) * 51
    grid_row = np.tile(columns, (6, 1))
    row_index, col_index = np.indices((6, 6))
    grid_2d = (row_index + col_index) * 25

    fig, axes = plt.subplots(2, 1, figsize=(6.4, 10.4), layout="constrained")
    draw_cells(axes[0], grid_row, "只随列变化：每一行都是 0, 51, 102, 153, 204, 255")
    draw_cells(axes[1], grid_2d, "每个格子 =（行号 + 列号）× 25")
    save(fig, "ch01_grids.png")

    csv_path = FIG / "ch01_grid.csv"
    np.savetxt(csv_path, grid_2d, fmt="%d", delimiter=",")
    print(f"已保存 {csv_path}")
    print("6×6 亮度（行号加列号，再乘 25）：")
    print(grid_2d)
    return grid_row, grid_2d


def plot_frames():
    fig, axes = plt.subplots(1, 8, figsize=(10.4, 2.15), layout="constrained")
    for index, ax in enumerate(axes):
        frame = np.full((72, 96), 22, dtype=float)
        left = 6 + index * 8
        frame[14:46, left : left + 18] = 235
        ax.imshow(frame, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"第 {index} 帧", fontsize=10)
        ax.text(
            48,
            66,
            f"{index * 125} ms",
            ha="center",
            va="center",
            color="white",
            fontsize=8,
        )
        for spine in ax.spines.values():
            spine.set_color("#1c1915")
            spine.set_linewidth(0.6)
    save(fig, "ch01_frames.png")


def check_textbook(y, pcm, sample_rate):
    if len(y) != sample_rate:
        raise SystemExit(f"课文音频应有 {sample_rate} 个采样，实际 {len(y)} 个。")
    if abs(y[0]) > 1e-12:
        raise SystemExit("t = 0 时正弦应为 0。")
    got = [int(v) for v in pcm[:8]]
    if got != TEXTBOOK_PCM:
        print("提示：前 8 个整数和课文表格不一致，请以终端打印为准。")
        print("实际：", got)
    else:
        print("自检通过：440 Hz 课文音频的前 8 个整数与网页表格一致。")

    import wave

    path = AUDIO / "note_440hz.wav"
    with wave.open(str(path), "r") as handle:
        if handle.getnframes() != sample_rate or handle.getframerate() != sample_rate:
            raise SystemExit("读回 wav 时，采样点数或采样率不对。")
        if handle.getnchannels() != 1 or handle.getsampwidth() != 2:
            raise SystemExit("课文 wav 应为单声道 16 位。")
    size = path.stat().st_size
    print(f"课文 wav 大小：{size} 字节（约 {size / 1024:.1f} KB）")


def main():
    if FREQUENCY <= 0 or SAMPLE_RATE < 100 or not (0 < DURATION <= 10):
        raise SystemExit("频率请用正数，采样率请 ≥ 100，时长请在 0 到 10 秒之间。")
    if not (0 < AMPLITUDE <= 1):
        raise SystemExit("振幅请用 0 到 1 之间的正数。")
    if FREQUENCY >= SAMPLE_RATE / 2:
        print("提示：频率已经不低于采样率的一半，听感会变成别的音高。")
        print("请把采样率加大，或把频率降低。文件仍会写出来。")

    use_style()
    plot_sampling()
    plot_quantization()

    _, textbook = make_wave(44100, 440.0, 1.0, 0.4)
    textbook_pcm = write_wav(AUDIO / "note_440hz.wav", textbook, 44100)
    print(f"已保存 {AUDIO / 'note_440hz.wav'}")
    print("课文音频的前 8 个采样（440 Hz，每秒 44100 个点，振幅 0.4）：")
    print_samples(textbook, textbook_pcm, 44100)
    check_textbook(textbook, textbook_pcm, 44100)
    plot_zoom("ch01_wav_zoom.png", 44100, 440.0, 0.4, "440 Hz，每秒 44100 点")

    _, mine = make_wave(SAMPLE_RATE, FREQUENCY, DURATION, AMPLITUDE)
    mine_pcm = write_wav(AUDIO / "note_mine.wav", mine, SAMPLE_RATE)
    print(f"已保存 {AUDIO / 'note_mine.wav'}")
    print("你的音频前 8 个采样：")
    print_samples(mine, mine_pcm, SAMPLE_RATE, count=min(8, len(mine)))
    plot_zoom("ch01_my_zoom.png", SAMPLE_RATE, FREQUENCY, AMPLITUDE, "你的参数")

    plot_gradient()
    plot_grids()
    plot_frames()
    print("第 1 篇的文件已写入 assets/audio、assets/images 和 assets/figures。")


if __name__ == "__main__":
    main()
