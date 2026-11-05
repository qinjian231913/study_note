"""第 10 篇：图像的二维频谱。

在「傅里叶变换」文件夹里运行：

    python code/ch10_image_fft.py

课文对照图每次都按固定参数重画。
文件顶部的 STRIPE_CYCLES 只影响 ch10_mine.png。

灰度图是一张数字矩阵。二维 DFT 可以想成：先对每一行做一维变换，再对每一列做。
numpy.fft.fft2 一次做完；fftshift 把直流（平均亮度）挪到画面中央。
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

from plot_style import INK, TEAL, WAVE, use_style

# ===== 改这里 =====
# 只影响 assets/figures/ch10_mine.png
# 整幅图里竖条纹重复多少次。整数，建议 2 到 40。
STRIPE_CYCLES = 8
# ==================

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "assets" / "figures"
SIZE = 256

SPEC_CMAP = LinearSegmentedColormap.from_list(
    "textbook_2d",
    ["#fffdf8", "#f3e4c8", "#e8b87a", "#c24d28", "#0e6b66", "#1c1915"],
)


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    path = FIG / name
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"已保存 {path}")


def vertical_stripes(cycles, size=SIZE):
    """沿 x 方向变化：竖条纹。cycles 是整幅图里重复次数。"""
    x = np.arange(size)
    wave = 0.5 + 0.5 * np.cos(2 * np.pi * cycles * x / size)
    return np.tile(wave, (size, 1))


def horizontal_stripes(cycles, size=SIZE):
    """沿 y 方向变化：横条纹。"""
    return vertical_stripes(cycles, size).T


def diagonal_stripes(cycles, size=SIZE):
    """沿对角线变化。"""
    y = np.arange(size)[:, None]
    x = np.arange(size)[None, :]
    return 0.5 + 0.5 * np.cos(2 * np.pi * cycles * (x + y) / size)


def spectrum2d(image):
    """fft2 + fftshift，返回以分贝显示的幅度谱（相对最大值）。"""
    spectrum = np.fft.fftshift(np.fft.fft2(image))
    mag = np.abs(spectrum)
    db = 20.0 * np.log10(np.maximum(mag / np.max(mag), 1e-12))
    return np.clip(db, -60, 0)


def show_pair(ax_img, ax_spec, image, title_img, title_spec):
    ax_img.imshow(image, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    ax_img.set_title(title_img)
    ax_img.set_xticks([])
    ax_img.set_yticks([])
    for spine in ax_img.spines.values():
        spine.set_color("#2a241c")

    spec = spectrum2d(image)
    ax_spec.imshow(spec, cmap=SPEC_CMAP, vmin=-60, vmax=0, interpolation="nearest")
    ax_spec.set_title(title_spec)
    ax_spec.set_xticks([])
    ax_spec.set_yticks([])
    cy = (spec.shape[0] - 1) / 2
    cx = (spec.shape[1] - 1) / 2
    ax_spec.plot(cx, cy, marker="+", color=TEAL, ms=10, mew=1.4)
    for spine in ax_spec.spines.values():
        spine.set_color("#2a241c")


def plot_matrix_idea():
    """一小块灰度矩阵示意。"""
    tiny = np.array(
        [
            [0.1, 0.3, 0.8, 0.9],
            [0.15, 0.35, 0.75, 0.85],
            [0.2, 0.4, 0.7, 0.8],
            [0.25, 0.45, 0.65, 0.75],
        ]
    )
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.8), layout="constrained")
    axes[0].imshow(tiny, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    axes[0].set_title("灰度图 = 数字矩阵")
    for (row, col), value in np.ndenumerate(tiny):
        color = "white" if value > 0.55 else INK
        axes[0].text(col, row, f"{value:.2f}", ha="center", va="center", color=color, fontsize=11)
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    axes[1].axis("off")
    axes[1].text(
        0.02,
        0.72,
        "二维 DFT 可以想成两步：",
        transform=axes[1].transAxes,
        color=INK,
        fontsize=13,
    )
    axes[1].text(
        0.02,
        0.48,
        "1. 对每一行做一维傅里叶变换\n2. 再对每一列做一维傅里叶变换",
        transform=axes[1].transAxes,
        color=TEAL,
        fontsize=13,
        linespacing=1.8,
    )
    axes[1].text(
        0.02,
        0.18,
        "代码里一行：numpy.fft.fft2(image)\nfftshift 把直流挪到中央",
        transform=axes[1].transAxes,
        color=WAVE,
        fontsize=13,
        linespacing=1.8,
    )
    save(fig, "ch10_matrix.png")


def plot_stripes():
    cycles = 8
    specs = (
        (vertical_stripes(cycles), "竖条纹（左右变化）", "频谱亮点落在水平轴上"),
        (horizontal_stripes(cycles), "横条纹（上下变化）", "频谱亮点落在竖直轴上"),
        (diagonal_stripes(cycles), "斜条纹（对角变化）", "频谱亮点落在对角方向"),
    )
    fig, axes = plt.subplots(3, 2, figsize=(8.2, 9.4), layout="constrained")
    for row, (image, t_img, t_spec) in enumerate(specs):
        show_pair(axes[row, 0], axes[row, 1], image, t_img, t_spec)
    save(fig, "ch10_stripes.png")


def plot_shift_demo():
    """fftshift 前后对比。"""
    image = vertical_stripes(6)
    raw = np.abs(np.fft.fft2(image))
    shifted = np.fft.fftshift(raw)
    raw_db = 20 * np.log10(np.maximum(raw / np.max(raw), 1e-12))
    shift_db = 20 * np.log10(np.maximum(shifted / np.max(shifted), 1e-12))
    fig, axes = plt.subplots(1, 3, figsize=(9.2, 3.4), layout="constrained")
    axes[0].imshow(image, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    axes[0].set_title("原图：竖条纹")
    axes[1].imshow(np.clip(raw_db, -60, 0), cmap=SPEC_CMAP, vmin=-60, vmax=0, interpolation="nearest")
    axes[1].set_title("fft2 之后：直流在四角")
    axes[2].imshow(np.clip(shift_db, -60, 0), cmap=SPEC_CMAP, vmin=-60, vmax=0, interpolation="nearest")
    axes[2].set_title("fftshift：直流到中央")
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    save(fig, "ch10_fftshift.png")


def plot_mine():
    image = vertical_stripes(int(STRIPE_CYCLES))
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 4.0), layout="constrained")
    show_pair(
        axes[0],
        axes[1],
        image,
        f"你的竖条纹：整幅重复 {int(STRIPE_CYCLES)} 次",
        "频谱：亮点离中心的距离跟着次数变",
    )
    save(fig, "ch10_mine.png")


def peak_offset(image):
    """频谱上（除直流外）最亮点相对中心的偏移（像素）。"""
    mag = np.abs(np.fft.fftshift(np.fft.fft2(image)))
    cy, cx = np.array(mag.shape) // 2
    mag = mag.copy()
    mag[cy, cx] = 0
    # 也清掉紧邻直流的一点泄漏
    mag[max(0, cy - 1) : cy + 2, max(0, cx - 1) : cx + 2] = 0
    iy, ix = np.unravel_index(np.argmax(mag), mag.shape)
    return int(ix - cx), int(iy - cy)


def self_check():
    v = vertical_stripes(8)
    h = horizontal_stripes(8)
    d = diagonal_stripes(8)
    dx_v, dy_v = peak_offset(v)
    dx_h, dy_h = peak_offset(h)
    dx_d, dy_d = peak_offset(d)
    if abs(dx_v) < 2 or abs(dy_v) > 2:
        raise SystemExit(f"自检失败：竖条纹峰值应在水平轴上，得到偏移 ({dx_v}, {dy_v})")
    if abs(dy_h) < 2 or abs(dx_h) > 2:
        raise SystemExit(f"自检失败：横条纹峰值应在竖直轴上，得到偏移 ({dx_h}, {dy_h})")
    if abs(dx_d) < 2 or abs(dy_d) < 2:
        raise SystemExit(f"自检失败：斜条纹峰值应在对角方向，得到偏移 ({dx_d}, {dy_d})")
    if abs(abs(dx_v) - 8) > 1:
        raise SystemExit(f"自检失败：8 次竖条纹的水平偏移应约为 8，得到 {dx_v}")
    print(
        f"自检通过：竖条纹峰值偏移约 ({dx_v}, {dy_v})，"
        f"横条纹约 ({dx_h}, {dy_h})，斜条纹约 ({dx_d}, {dy_d})。"
    )


def main():
    if (
        not isinstance(STRIPE_CYCLES, int)
        or isinstance(STRIPE_CYCLES, bool)
        or not (2 <= STRIPE_CYCLES <= 40)
    ):
        raise SystemExit("STRIPE_CYCLES 请用 2 到 40 的整数。课文对照图不跟这个数走。")
    use_style()
    self_check()
    plot_matrix_idea()
    plot_shift_demo()
    plot_stripes()
    plot_mine()
    print("课文图：ch10_matrix.png、ch10_fftshift.png、ch10_stripes.png")
    print("你的图：ch10_mine.png（只跟随 STRIPE_CYCLES）")
    print(f"当前 STRIPE_CYCLES = {STRIPE_CYCLES}")


if __name__ == "__main__":
    main()
