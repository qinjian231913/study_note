"""第 12 篇：视频多出来的时间轴。

在「傅里叶变换」文件夹里运行：

    python code/ch12_video.py

课文对照图与帧序列按固定参数重画。
文件顶部的 FLASH_HZ 只影响 ch12_my_*.png 和你的帧目录。

视频 = 一串按时间排好的图。盯住中心像素，亮度随时间就是一维信号。
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_style import INK, MUTED, TEAL, WAVE, finish, use_style

# ===== 改这里 =====
# 只影响 assets/figures/ch12_my_waveform.png、ch12_my_spectrum.png
# 以及 assets/figures/ch12_frames_mine/ 里的帧
# 闪烁频率（Hz）。课文固定用 5 Hz。
FLASH_HZ = 5.0
# ==================

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "assets" / "figures"
FPS = 20
DURATION = 2.0
WIDTH = 96
HEIGHT = 72
TEXTBOOK_FLASH_HZ = 5.0
BLOB_PERIOD = 2.0  # 光斑左右走一个来回的秒数


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    path = FIG / name
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"已保存 {path}")


def make_frame(t, flash_hz, width=WIDTH, height=HEIGHT):
    """某一时刻的画面：底色 + 全屏正弦闪 + 底部缓慢移动的亮斑。

    光斑走在靠下的位置，避免盖住中心像素的闪烁主峰。
    """
    frame = np.full((height, width), 48.0)
    # 全屏闪：正弦，频谱干净地落在 flash_hz
    frame = frame + 70.0 * (0.5 + 0.5 * np.sin(2 * np.pi * flash_hz * t))
    # 移动光斑（靠下，中心像素几乎只看见全屏闪）
    cx = width * (0.15 + 0.7 * (0.5 + 0.5 * np.sin(2 * np.pi * t / BLOB_PERIOD)))
    cy = height * 0.78
    yy, xx = np.mgrid[0:height, 0:width]
    blob = 140.0 * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / 28.0)
    frame = frame + blob
    return np.clip(frame, 0, 255)


def synthesize(flash_hz, fps=FPS, duration=DURATION):
    n_frames = int(round(fps * duration))
    times = np.arange(n_frames) / fps
    frames = [make_frame(t, flash_hz) for t in times]
    center = np.array([f[HEIGHT // 2, WIDTH // 2] for f in frames], dtype=float)
    return times, frames, center


def save_frames(frames, folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    for old in folder.glob("frame_*.png"):
        old.unlink()
    for index, frame in enumerate(frames):
        path = folder / f"frame_{index:03d}.png"
        plt.imsave(path, frame, cmap="gray", vmin=0, vmax=255)
    print(f"已保存 {len(frames)} 帧到 {folder}")


def try_gif(frames, path, fps=FPS):
    """若 pillow 可用则写 gif；否则跳过。"""
    try:
        from PIL import Image
    except ImportError:
        print("未安装 pillow，跳过 gif（帧 PNG 已足够）。")
        return False
    images = [Image.fromarray(np.clip(f, 0, 255).astype(np.uint8), mode="L") for f in frames]
    duration_ms = int(round(1000 / fps))
    images[0].save(
        path,
        save_all=True,
        append_images=images[1:],
        duration=duration_ms,
        loop=0,
    )
    print(f"已保存 {path}")
    return True


def plot_montage(frames, name, step=4):
    """抽若干帧横排成胶片条。"""
    picked = frames[::step]
    fig, axes = plt.subplots(1, len(picked), figsize=(11.0, 1.8), layout="constrained")
    for index, (ax, frame) in enumerate(zip(axes, picked)):
        ax.imshow(frame, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        t = (index * step) / FPS
        ax.set_title(f"{t:.1f}s", fontsize=9)
        # 标中心像素
        ax.plot([WIDTH // 2], [HEIGHT // 2], "+", color=WAVE, ms=8, mew=1.2)
    fig.suptitle("胶片条（橙色十字 = 中心像素）", fontsize=12, color=INK)
    save(fig, name)


def plot_waveform(times, center, name, title):
    fig, ax = plt.subplots(figsize=(8.4, 3.2), layout="constrained")
    ax.plot(times, center, color=WAVE, lw=1.8)
    ax.plot(times, center, "o", color=TEAL, ms=3.5, zorder=3)
    finish(ax)
    ax.set_xlabel("时间 t（秒）")
    ax.set_ylabel("中心像素亮度")
    ax.set_title(title)
    ax.set_ylim(0, 255)
    save(fig, name)


def plot_spectrum(times, center, name, title, mark_hz=None):
    n = len(center)
    # 去掉平均值，只看波动
    y = center - center.mean()
    spec = np.fft.rfft(y)
    freqs = np.fft.rfftfreq(n, d=times[1] - times[0])
    amp = np.abs(spec) * 2 / n
    amp[0] = np.abs(spec[0]) / n

    fig, ax = plt.subplots(figsize=(8.4, 3.2), layout="constrained")
    ax.stem(freqs, amp, linefmt=TEAL, markerfmt="o", basefmt=MUTED)
    if mark_hz is not None:
        ax.axvline(mark_hz, color=WAVE, ls="--", lw=1.2, label=f"{mark_hz:g} Hz")
        ax.legend(frameon=False)
    finish(ax)
    ax.set_xlabel("频率（Hz）")
    ax.set_ylabel("振幅")
    ax.set_title(title)
    ax.set_xlim(0, min(FPS / 2, 12))
    save(fig, name)
    peak_hz = float(freqs[int(np.argmax(amp[1:])) + 1]) if n > 1 else 0.0
    return peak_hz, amp


def plot_fps_idea():
    """说明帧率：同样 1 秒，10 帧和 20 帧采样点不同。"""
    t_fine = np.linspace(0, 1, 400)
    y_fine = 120 + 80 * np.sin(2 * np.pi * 5 * t_fine)
    fig, axes = plt.subplots(2, 1, figsize=(8.4, 5.0), layout="constrained", sharex=True)
    for ax, fps, title in (
        (axes[0], 10, "每秒 10 帧：1 秒里只有 10 个采样点"),
        (axes[1], 20, "每秒 20 帧：同样 1 秒有 20 个点，波形更密"),
    ):
        t = np.arange(fps) / fps
        y = 120 + 80 * np.sin(2 * np.pi * 5 * t)
        ax.plot(t_fine, y_fine, color=MUTED, lw=1.0, alpha=0.55)
        ax.plot(t, y, color=WAVE, lw=1.6)
        ax.plot(t, y, "o", color=TEAL, ms=5)
        finish(ax)
        ax.set_ylabel("亮度")
        ax.set_title(title)
        ax.set_ylim(20, 220)
    axes[1].set_xlabel("时间 t（秒）")
    save(fig, "ch12_framerate.png")


def self_check(times, center, peak_hz):
    if len(times) != int(FPS * DURATION):
        raise SystemExit(f"自检失败：帧数应是 {int(FPS * DURATION)}，实际 {len(times)}")
    if abs(peak_hz - TEXTBOOK_FLASH_HZ) > 0.6:
        raise SystemExit(f"自检失败：中心像素频谱峰值在 {peak_hz:.2f} Hz，期望约 5 Hz")
    # 闪烁应让亮度有明显起伏
    if float(center.max() - center.min()) < 40:
        raise SystemExit("自检失败：中心像素起伏太小。")
    print(f"自检通过：{len(times)} 帧，中心像素频谱主峰约 {peak_hz:.2f} Hz。")


def main():
    if not (0.5 <= FLASH_HZ <= 9.0):
        raise SystemExit("FLASH_HZ 请用 0.5 到 9 之间的数（低于帧率一半）。")

    use_style()
    plot_fps_idea()

    times, frames, center = synthesize(TEXTBOOK_FLASH_HZ)
    frame_dir = FIG / "ch12_frames"
    save_frames(frames, frame_dir)
    try_gif(frames, FIG / "ch12_flash.gif")
    plot_montage(frames, "ch12_montage.png", step=4)
    plot_waveform(
        times,
        center,
        "ch12_waveform.png",
        f"中心像素随时间（全屏闪 {TEXTBOOK_FLASH_HZ:g} Hz + 移动光斑）",
    )
    peak_hz, _ = plot_spectrum(
        times,
        center,
        "ch12_spectrum.png",
        "中心像素的频谱（主峰应对准闪烁频率）",
        mark_hz=TEXTBOOK_FLASH_HZ,
    )
    self_check(times, center, peak_hz)

    # 你的参数
    times_m, frames_m, center_m = synthesize(FLASH_HZ)
    save_frames(frames_m, FIG / "ch12_frames_mine")
    try_gif(frames_m, FIG / "ch12_my_flash.gif")
    plot_montage(frames_m, "ch12_my_montage.png", step=4)
    plot_waveform(
        times_m,
        center_m,
        "ch12_my_waveform.png",
        f"你的中心像素波形（FLASH_HZ={FLASH_HZ:g}）",
    )
    plot_spectrum(
        times_m,
        center_m,
        "ch12_my_spectrum.png",
        f"你的中心像素频谱（应对准 {FLASH_HZ:g} Hz）",
        mark_hz=FLASH_HZ,
    )
    print(
        "课文图：ch12_framerate.png、ch12_montage.png、ch12_waveform.png、ch12_spectrum.png；"
        "帧在 assets/figures/ch12_frames/"
    )
    print("你的图：ch12_my_*.png 与 ch12_frames_mine/（跟随 FLASH_HZ）")


if __name__ == "__main__":
    main()
