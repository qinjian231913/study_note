"""第 13 篇：压缩为什么喜欢频域。

在「傅里叶变换」文件夹里运行：

    python code/ch13_dct.py

课文对照图按固定参数重画。
文件顶部的 QUANT_SCALE 只影响 ch13_my_*.png。

JPEG 核心一步：8×8 块 → DCT → 量化（大量变 0）→ 反量化 → IDCT。
DCT 相对 DFT：实数、更适合「一块图像能量挤到左上角」。
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_style import INK, MUTED, TEAL, WAVE, finish, use_style

# ===== 改这里 =====
# 只影响 assets/figures/ch13_my_block.png
# QUANT_SCALE：量化表倍率。越大越狠（零更多、块更糊）；1.0 约等于「中等质量」
QUANT_SCALE = 1.0
# ==================

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "assets" / "figures"
AUDIO = ROOT / "assets" / "audio"
TEXTBOOK_SCALE = 1.0

# JPEG 标准亮度量化表（质量约 50 时的常见表）
JPEG_Q50 = np.array(
    [
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99],
    ],
    dtype=float,
)


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    path = FIG / name
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"已保存 {path}")


def dct_matrix(n=8):
    """正交 DCT-II 矩阵。"""
    k = np.arange(n)[:, None]
    i = np.arange(n)[None, :]
    c = np.cos(np.pi * (i + 0.5) * k / n)
    c[0, :] *= np.sqrt(1.0 / n)
    c[1:, :] *= np.sqrt(2.0 / n)
    return c


_C8 = None


def _C():
    global _C8
    if _C8 is None:
        _C8 = dct_matrix(8)
    return _C8


def dct2(block):
    c = _C()
    return c @ block @ c.T


def idct2(coeffs):
    c = _C()
    return c.T @ coeffs @ c


def make_block():
    """一块有平滑渐变 + 一条竖边的 8×8，模拟照片局部。"""
    yy, xx = np.mgrid[0:8, 0:8]
    block = 110 + 9 * xx + 4 * yy
    block[:, 5:] = block[:, 5:] + 35
    block[3:6, 2:4] = block[3:6, 2:4] - 18
    return np.clip(block, 0, 255).astype(float)


def quantize(coeffs, scale):
    q = np.maximum(JPEG_Q50 * scale, 1.0)
    return np.round(coeffs / q), q


def dequantize(levels, q):
    return levels * q


def jpeg_roundtrip(block, scale):
    coeffs = dct2(block - 128.0)  # JPEG 习惯先减 128
    levels, q = quantize(coeffs, scale)
    restored = idct2(dequantize(levels, q)) + 128.0
    zeros = int(np.sum(levels == 0))
    return coeffs, levels, np.clip(restored, 0, 255), zeros


def draw_grid(ax, data, title, fmt="{:.0f}", cmap="gray", vmin=None, vmax=None):
    if vmin is None:
        vmin = float(np.min(data))
    if vmax is None:
        vmax = float(np.max(data))
    if vmax <= vmin:
        vmax = vmin + 1
    ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    for r in range(data.shape[0]):
        for c in range(data.shape[1]):
            val = data[r, c]
            text = fmt.format(val)
            # 亮度判断用显示值
            norm = (val - vmin) / (vmax - vmin) if cmap == "gray" else 0.5
            color = "#1c1915" if (cmap != "gray" or norm >= 0.45) else "white"
            if cmap != "gray":
                color = "white" if abs(val) > 0.55 * max(abs(vmin), abs(vmax), 1) else INK
            ax.text(c, r, text, ha="center", va="center", color=color, fontsize=8)
    ax.set_xticks(range(8))
    ax.set_yticks(range(8))
    ax.tick_params(length=0, labelsize=8)
    ax.set_title(title, fontsize=11)


def plot_path(scale, name, title_note):
    block = make_block()
    coeffs, levels, restored, zeros = jpeg_roundtrip(block, scale)
    fig, axes = plt.subplots(2, 3, figsize=(10.2, 6.8), layout="constrained")
    draw_grid(axes[0, 0], block, "① 原 8×8 块")
    draw_grid(axes[0, 1], coeffs, "② DCT 系数", fmt="{:.0f}", cmap="coolwarm")
    draw_grid(axes[0, 2], levels, f"③ 量化后（零={zeros}）", fmt="{:.0f}", cmap="coolwarm")
    draw_grid(axes[1, 0], dequantize(levels, np.maximum(JPEG_Q50 * scale, 1.0)), "④ 反量化", fmt="{:.0f}", cmap="coolwarm")
    draw_grid(axes[1, 1], restored, "⑤ IDCT 重建")
    err = restored - block
    draw_grid(axes[1, 2], err, "重建 − 原图", fmt="{:.1f}", cmap="coolwarm", vmin=-20, vmax=20)
    fig.suptitle(title_note, fontsize=12, color=INK)
    save(fig, name)
    mse = float(np.mean(err ** 2))
    return zeros, mse, levels


def plot_energy_pack():
    """对比：空域像素分散，DCT 后能量挤到左上。"""
    block = make_block()
    coeffs = dct2(block - 128.0)
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.6), layout="constrained")
    draw_grid(axes[0], block, "空域：数值铺满整块")
    mag = np.abs(coeffs)
    draw_grid(axes[1], mag, "DCT 幅度：能量挤在左上角", fmt="{:.0f}", cmap="magma")
    save(fig, "ch13_energy.png")


def write_wav(path, y, sample_rate=44100):
    import wave

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = np.int16(np.clip(y, -1, 1) * 32767)
    with wave.open(str(path), "w") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())
    print(f"已保存 {path}")


def plot_masking_idea():
    """响音旁边的轻音难听见——压缩会丢掉被掩蔽的细节。合成：响 440 + 轻 880。"""
    sr = 44100
    t = np.arange(int(0.8 * sr)) / sr
    loud = 0.55 * np.sin(2 * np.pi * 440 * t)
    quiet = 0.04 * np.sin(2 * np.pi * 880 * t)
    mix = loud + quiet
    write_wav(AUDIO / "ch13_loud_quiet.wav", mix, sr)
    write_wav(AUDIO / "ch13_quiet_only.wav", quiet * 8, sr)  # 单独放大轻音便于对照听

    # 简易语谱图：短时 FFT
    win = 1024
    hop = 256
    n_frames = 1 + (len(mix) - win) // hop
    spec = np.empty((win // 2 + 1, n_frames))
    window = np.hanning(win)
    for i in range(n_frames):
        chunk = mix[i * hop : i * hop + win] * window
        spec[:, i] = np.abs(np.fft.rfft(chunk))
    freqs = np.fft.rfftfreq(win, d=1 / sr)
    times = np.arange(n_frames) * hop / sr

    fig, axes = plt.subplots(2, 1, figsize=(8.6, 5.6), layout="constrained")
    axes[0].plot(t[:4000], mix[:4000], color=WAVE, lw=1.0)
    finish(axes[0])
    axes[0].set_title("响 440 Hz + 很轻的 880 Hz（时域看不清轻音）")
    axes[0].set_xlabel("时间 t（秒）")
    axes[0].set_ylabel("强弱 y")

    shown = np.log1p(spec)
    # 只显示到 2 kHz
    f_max = 2000
    f_idx = freqs <= f_max
    axes[1].imshow(
        shown[f_idx],
        origin="lower",
        aspect="auto",
        extent=[times[0], times[-1], freqs[f_idx][0], freqs[f_idx][-1]],
        cmap="magma",
        interpolation="nearest",
    )
    axes[1].axhline(440, color=TEAL, ls="--", lw=1.0, alpha=0.85)
    axes[1].axhline(880, color=WAVE, ls="--", lw=1.0, alpha=0.85)
    axes[1].set_title("语谱图示意：亮的是响的 440；880 那条很暗（易被掩蔽）")
    axes[1].set_xlabel("时间 t（秒）")
    axes[1].set_ylabel("频率（Hz）")
    save(fig, "ch13_masking.png")


def plot_quality_compare():
    """同一块在不同量化倍率下的零的个数与观感。"""
    scales = (0.5, 1.0, 3.0)
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.2), layout="constrained")
    for ax, scale in zip(axes, scales):
        block = make_block()
        _c, levels, restored, zeros = jpeg_roundtrip(block, scale)
        ax.imshow(restored, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"倍率 {scale:g}\n零系数 {zeros}/64", fontsize=11)
    fig.suptitle("QUANT_SCALE 越大，零越多，块越糊", fontsize=12, color=INK)
    save(fig, "ch13_quality.png")


def self_check():
    block = make_block()
    coeffs, levels, restored, zeros = jpeg_roundtrip(block, TEXTBOOK_SCALE)
    if zeros < 20:
        raise SystemExit(f"自检失败：中等量化零太少（{zeros}），请检查量化表。")
    if zeros > 60:
        raise SystemExit(f"自检失败：零太多（{zeros}），块会糊得不像样。")
    # 无损路径：scale 极小
    _c, levels0, restored0, zeros0 = jpeg_roundtrip(block, 0.05)
    err0 = float(np.max(np.abs(restored0 - block)))
    if err0 > 8:
        raise SystemExit(f"自检失败：很细的量化误差仍有 {err0}")
    # DCT 往返（不量化）应几乎无损
    back = idct2(dct2(block - 128)) + 128
    if float(np.max(np.abs(back - block))) > 1e-6:
        raise SystemExit("自检失败：DCT/IDCT 往返误差过大。")
    print(f"自检通过：QUANT_SCALE={TEXTBOOK_SCALE:g} 时零系数 {zeros}/64；DCT 往返数值稳定。")
    return zeros


def main():
    if not (0.1 <= QUANT_SCALE <= 8.0):
        raise SystemExit("QUANT_SCALE 请用 0.1 到 8 之间的数。")

    use_style()
    zeros = self_check()
    plot_energy_pack()
    plot_path(
        TEXTBOOK_SCALE,
        "ch13_jpeg_path.png",
        f"JPEG 一块的路径（QUANT_SCALE={TEXTBOOK_SCALE:g}，零系数 {zeros}/64）",
    )
    plot_quality_compare()
    plot_masking_idea()
    my_zeros, my_mse, _ = plot_path(
        QUANT_SCALE,
        "ch13_my_block.png",
        f"你的量化倍率 QUANT_SCALE={QUANT_SCALE:g}",
    )
    print(f"课文零系数 {zeros}/64；你的零系数 {my_zeros}/64，均方误差约 {my_mse:.2f}。")
    print(
        "课文图：ch13_energy.png、ch13_jpeg_path.png、ch13_quality.png、ch13_masking.png"
    )
    print("你的图：ch13_my_block.png（跟随 QUANT_SCALE）")


if __name__ == "__main__":
    main()
