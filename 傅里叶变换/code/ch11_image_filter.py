"""第 11 篇：在频率上改照片。

在「傅里叶变换」文件夹里运行：

    python code/ch11_image_filter.py

课文对照图每次都按固定参数重画。
文件顶部的 CUTOFF、NOTCH_RADIUS 只影响 ch11_my_*.png。

低通留下中心低频 → 模糊；高通抠掉中心 → 边缘；
陷波挖掉条纹对应的亮点 → 去掉周期条纹。
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_style import WAVE, use_style

# ===== 改这里 =====
# 只影响 assets/figures/ch11_my_lowpass.png 和 ch11_my_notch.png
# CUTOFF：低通半径（像素，频谱中心到边缘约为 SIZE/2）
# NOTCH_RADIUS：陷波挖掉条纹峰的半径（像素）
CUTOFF = 18
NOTCH_RADIUS = 3
# ==================

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "assets" / "figures"
SIZE = 160
TEXTBOOK_CUTOFF = 18
TEXTBOOK_HIGHPASS = 10
TEXTBOOK_NOTCH = 3
STRIPE_PERIOD = 8


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    path = FIG / name
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"已保存 {path}")


def make_photo(size=SIZE):
    """合成一张「照片」：柔和天空、山丘、太阳。"""
    yy, xx = np.mgrid[0:size, 0:size]
    y = yy / (size - 1)
    x = xx / (size - 1)
    sky = 90 + 70 * (1 - y)
    hill = 55 + 35 * np.exp(-((x - 0.35) ** 2) / 0.08 - ((y - 0.78) ** 2) / 0.04)
    hill2 = 40 + 28 * np.exp(-((x - 0.72) ** 2) / 0.06 - ((y - 0.82) ** 2) / 0.035)
    sun = 55 * np.exp(-((x - 0.78) ** 2 + (y - 0.22) ** 2) / 0.012)
    photo = sky + hill + hill2 + sun
    photo = photo + 4 * np.sin(2 * np.pi * x * 3) * (y > 0.55)
    return np.clip(photo, 0, 255).astype(float)


def add_stripes(image, period=STRIPE_PERIOD, strength=42.0):
    """叠一层竖直周期条纹（像扫描干扰）。"""
    size = image.shape[0]
    xx = np.arange(size)
    stripe = strength * np.sin(2 * np.pi * xx / period)
    return np.clip(image + stripe[None, :], 0, 255)


def spectrum(image):
    """二维 FFT，中心是低频；返回移位后的复频谱和对数幅度。"""
    shifted = np.fft.fftshift(np.fft.fft2(image))
    mag = np.log1p(np.abs(shifted))
    return shifted, mag


def lowpass_mask(size, radius):
    yy, xx = np.mgrid[-size // 2 : size // 2, -size // 2 : size // 2]
    return (xx * xx + yy * yy) <= radius * radius


def highpass_mask(size, radius):
    return ~lowpass_mask(size, radius)


def notch_mask(size, period=STRIPE_PERIOD, radius=TEXTBOOK_NOTCH):
    """竖直条纹在频谱上是左右一对亮点，挖掉它们。"""
    yy, xx = np.mgrid[-size // 2 : size // 2, -size // 2 : size // 2]
    # 竖直条纹 → 水平频率轴上的峰；周期 P → 距中心约 size/P 格
    fx = size / period
    mask = np.ones((size, size), dtype=bool)
    for sign in (-1, 1):
        dist2 = (xx - sign * fx) ** 2 + yy ** 2
        mask &= dist2 > radius * radius
    return mask


def apply_mask(image, mask):
    shifted, _ = spectrum(image)
    filtered = np.fft.ifft2(np.fft.ifftshift(shifted * mask)).real
    return np.clip(filtered, 0, 255)


def show_gray(ax, image, title):
    ax.imshow(image, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title)


def show_spec(ax, mag, title, overlay=None):
    ax.imshow(mag, cmap="magma", interpolation="nearest")
    if overlay is not None:
        # 画出掩膜边界示意：被挖掉的区域略暗
        ax.imshow(np.where(overlay, 0, 0.45), cmap="gray", vmin=0, vmax=1, alpha=0.35)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title)


def plot_photo_and_stripes():
    photo = make_photo()
    striped = add_stripes(photo)
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 4.0), layout="constrained")
    show_gray(axes[0], photo, "合成「照片」")
    show_gray(axes[1], striped, f"叠上周期 {STRIPE_PERIOD} 的竖直条纹")
    save(fig, "ch11_photo_stripes.png")
    return photo, striped


def plot_trio(name, original, result, mag_masked, titles):
    fig, axes = plt.subplots(1, 3, figsize=(9.8, 3.4), layout="constrained")
    show_gray(axes[0], original, titles[0])
    show_spec(axes[1], mag_masked, titles[1])
    show_gray(axes[2], result, titles[2])
    save(fig, name)


def plot_lowpass(photo):
    mask = lowpass_mask(SIZE, TEXTBOOK_CUTOFF)
    result = apply_mask(photo, mask)
    shifted, _ = spectrum(photo)
    mag_masked = np.log1p(np.abs(shifted * mask))
    plot_trio(
        "ch11_lowpass.png",
        photo,
        result,
        mag_masked,
        ("原图", f"低通后频谱（半径 {TEXTBOOK_CUTOFF}）", "结果：模糊"),
    )
    return result


def plot_highpass(photo):
    mask = highpass_mask(SIZE, TEXTBOOK_HIGHPASS)
    result = apply_mask(photo, mask)
    # 高通结果对比度小，拉伸便于看见边缘
    shown = result - result.min()
    if shown.max() > 0:
        shown = shown / shown.max() * 255
    shifted, _ = spectrum(photo)
    mag_masked = np.log1p(np.abs(shifted * mask))
    plot_trio(
        "ch11_highpass.png",
        photo,
        shown,
        mag_masked,
        ("原图", f"高通后频谱（挖掉半径 {TEXTBOOK_HIGHPASS}）", "结果：边缘（已拉伸）"),
    )
    return shown


def plot_notch(striped):
    mask = notch_mask(SIZE, STRIPE_PERIOD, TEXTBOOK_NOTCH)
    result = apply_mask(striped, mask)
    shifted, mag0 = spectrum(striped)
    mag_masked = np.log1p(np.abs(shifted * mask))
    fig, axes = plt.subplots(1, 3, figsize=(9.8, 3.4), layout="constrained")
    show_gray(axes[0], striped, "带条纹的图")
    show_spec(axes[1], mag_masked, f"陷波后频谱（半径 {TEXTBOOK_NOTCH}）", overlay=mask)
    show_gray(axes[2], result, "去掉周期条纹")
    save(fig, "ch11_notch.png")

    # 单独一张频谱对照：改前能看见左右亮点
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.5), layout="constrained")
    show_spec(axes[0], mag0, "改前：左右一对亮点 = 竖直条纹")
    show_spec(axes[1], mag_masked, "陷波挖掉亮点")
    # 标出峰位置
    fx = SIZE / STRIPE_PERIOD
    for ax in axes:
        ax.plot([SIZE / 2 + fx], [SIZE / 2], "o", color=WAVE, ms=6, fillstyle="none", mew=1.5)
        ax.plot([SIZE / 2 - fx], [SIZE / 2], "o", color=WAVE, ms=6, fillstyle="none", mew=1.5)
    save(fig, "ch11_spectrum_peaks.png")
    return result


def plot_mine(photo, striped):
    lp = apply_mask(photo, lowpass_mask(SIZE, CUTOFF))
    shifted, _ = spectrum(photo)
    mag_lp = np.log1p(np.abs(shifted * lowpass_mask(SIZE, CUTOFF)))
    fig, axes = plt.subplots(1, 3, figsize=(9.8, 3.4), layout="constrained")
    show_gray(axes[0], photo, "原图")
    show_spec(axes[1], mag_lp, f"你的低通半径 CUTOFF={CUTOFF}")
    show_gray(axes[2], lp, "你的低通结果")
    save(fig, "ch11_my_lowpass.png")

    mask = notch_mask(SIZE, STRIPE_PERIOD, NOTCH_RADIUS)
    cleaned = apply_mask(striped, mask)
    shifted_s, _ = spectrum(striped)
    mag_n = np.log1p(np.abs(shifted_s * mask))
    fig, axes = plt.subplots(1, 3, figsize=(9.8, 3.4), layout="constrained")
    show_gray(axes[0], striped, "带条纹")
    show_spec(axes[1], mag_n, f"你的陷波半径 NOTCH_RADIUS={NOTCH_RADIUS}")
    show_gray(axes[2], cleaned, "你的去条纹结果")
    save(fig, "ch11_my_notch.png")


def self_check(photo, striped, low, high, cleaned):
    if photo.shape != (SIZE, SIZE):
        raise SystemExit(f"自检失败：照片尺寸是 {photo.shape}")
    # 低通应更平滑：高频能量下降
    _, mag_p = spectrum(photo)
    _, mag_l = spectrum(low)
    if float(mag_l.mean()) >= float(mag_p.mean()):
        raise SystemExit("自检失败：低通后频谱幅度均值应下降。")
    # 陷波后条纹应减弱：竖直方向相邻列差变小
    stripe_before = float(np.mean(np.abs(np.diff(striped, axis=1))))
    stripe_after = float(np.mean(np.abs(np.diff(cleaned, axis=1))))
    if stripe_after >= stripe_before * 0.85:
        raise SystemExit(
            f"自检失败：去条纹后列差应明显变小（前 {stripe_before:.2f}，后 {stripe_after:.2f}）。"
        )
    if float(np.std(high)) < 1:
        raise SystemExit("自检失败：高通结果几乎是平的。")
    print("自检通过：低通变糊，高通留边缘，陷波削弱周期条纹。")
    print(f"条纹列差：改前 {stripe_before:.2f} → 改后 {stripe_after:.2f}。")


def main():
    if not (1 <= CUTOFF <= SIZE // 2):
        raise SystemExit(f"CUTOFF 请用 1 到 {SIZE // 2} 的整数。")
    if not (1 <= NOTCH_RADIUS <= 20):
        raise SystemExit("NOTCH_RADIUS 请用 1 到 20 的整数。")

    use_style()
    photo, striped = plot_photo_and_stripes()
    low = plot_lowpass(photo)
    high = plot_highpass(photo)
    cleaned = plot_notch(striped)
    plot_mine(photo, striped)
    self_check(photo, striped, low, high, cleaned)
    print(
        "课文图：ch11_photo_stripes.png、ch11_lowpass.png、ch11_highpass.png、"
        "ch11_notch.png、ch11_spectrum_peaks.png"
    )
    print("你的图：ch11_my_lowpass.png、ch11_my_notch.png（跟随 CUTOFF、NOTCH_RADIUS）")


if __name__ == "__main__":
    main()
