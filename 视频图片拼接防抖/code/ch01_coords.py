"""第 1 篇：图像坐标与刚体直觉。

在「视频图片拼接防抖」文件夹里运行：

    python code/ch01_coords.py

图保存到 assets/figures/ 与 assets/images/。
顶部「改这里」只影响 ch01_my_transform.png。
"""

from pathlib import Path

SHOW_WINDOW = False

import matplotlib

if not SHOW_WINDOW:
    matplotlib.use("Agg")

import cv2
import matplotlib.pyplot as plt
import numpy as np

from plot_style import MUTED, TEAL, WAVE, finish, use_style

# ===== 改这里 =====
# 只影响 ch01_my_transform.png
TX = 40          # 向右平移像素（负数为向左）
TY = -15         # 向下平移像素（图像坐标 y 向下为正；这里负=向上）
ANGLE_DEG = 25   # 逆时针旋转角度（度）
SCALE = 1.15     # 缩放倍数
# ==================

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "assets" / "figures"
IMG = ROOT / "assets" / "images"

WAVE_BGR = (40, 77, 194)
TEAL_BGR = (102, 107, 14)


def save_fig(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    path = FIG / name
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"已保存 {path}")


def save_bgr(name, image_bgr):
    IMG.mkdir(parents=True, exist_ok=True)
    path = IMG / name
    if not cv2.imwrite(str(path), image_bgr):
        raise SystemExit(f"写图失败：{path}")
    print(f"已保存 {path}")


def make_card(width=220, height=160):
    """一张带十字、字母 L、格子的卡片，方便看出旋转方向与翻转。"""
    img = np.full((height, width, 3), 248, dtype=np.uint8)
    # 浅网格
    for x in range(0, width, 20):
        cv2.line(img, (x, 0), (x, height - 1), (230, 220, 205), 1)
    for y in range(0, height, 20):
        cv2.line(img, (0, y), (width - 1, y), (230, 220, 205), 1)

    # 外框
    cv2.rectangle(img, (8, 8), (width - 9, height - 9), (60, 55, 50), 2)

    # 粗 L：开口朝右下，旋转后方向感很强
    cv2.rectangle(img, (50, 40), (78, 120), WAVE_BGR, -1)
    cv2.rectangle(img, (50, 95), (150, 120), WAVE_BGR, -1)

    # 青点标出「角」：左上靠近原点一侧
    cv2.circle(img, (30, 30), 7, TEAL_BGR, -1)
    cv2.putText(img, "P", (38, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, TEAL_BGR, 2, cv2.LINE_AA)

    # 中心十字
    cx, cy = width // 2, height // 2
    cv2.drawMarker(img, (cx, cy), (30, 30, 30), markerType=cv2.MARKER_CROSS, markerSize=18, thickness=2)

    return img


def place_on_canvas(src, canvas_size=(420, 320), offset=(60, 50)):
    """把小图放到大画布，方便展示平移后仍看得见。"""
    canvas = np.full((canvas_size[1], canvas_size[0], 3), 251, dtype=np.uint8)
    x0, y0 = offset
    h, w = src.shape[:2]
    canvas[y0 : y0 + h, x0 : x0 + w] = src
    return canvas


def similarity_matrix(cx, cy, angle_deg, scale, tx, ty):
    """绕 (cx,cy) 缩放+旋转，再平移。返回 2×3 的仿射矩阵（OpenCV warpAffine 用）。"""
    m = cv2.getRotationMatrix2D((cx, cy), angle_deg, scale)
    m[0, 2] += tx
    m[1, 2] += ty
    return m


def warp(src, matrix, canvas_size):
    return cv2.warpAffine(
        src,
        matrix,
        canvas_size,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(251, 251, 251),
    )


def plot_coordinate_axes():
    """图像坐标 vs 数学坐标对照。"""
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.2), layout="constrained")

    ax = axes[0]
    ax.set_xlim(-0.5, 6.5)
    ax.set_ylim(4.5, -0.5)  # 倒置 y，模拟图像坐标
    ax.set_aspect("equal")
    for x in range(7):
        ax.axvline(x, color="#e6dccb", lw=0.8)
    for y in range(5):
        ax.axhline(y, color="#e6dccb", lw=0.8)
    ax.scatter([0], [0], s=60, color=WAVE, zorder=3)
    ax.scatter([4], [2], s=60, color=TEAL, zorder=3)
    ax.annotate("原点 (0,0)\n左上角", xy=(0, 0), xytext=(0.6, 1.4), color=WAVE, fontsize=11,
                arrowprops=dict(arrowstyle="->", color=WAVE))
    ax.annotate("像素 (4,2)", xy=(4, 2), xytext=(4.4, 3.3), color=TEAL, fontsize=11,
                arrowprops=dict(arrowstyle="->", color=TEAL))
    ax.annotate("", xy=(5.5, 0), xytext=(0.3, 0),
                arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.4))
    ax.annotate("", xy=(0, 3.8), xytext=(0, 0.3),
                arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.4))
    ax.text(5.2, -0.25, "x 向右", color=MUTED, fontsize=11)
    ax.text(-0.35, 3.5, "y 向下", color=MUTED, fontsize=11, rotation=90, va="center")
    ax.set_title("图像坐标（OpenCV / 大多数库）")
    ax.set_xlabel("列 x")
    ax.set_ylabel("行 y")

    ax = axes[1]
    ax.set_xlim(-0.5, 6.5)
    ax.set_ylim(-0.5, 4.5)
    ax.set_aspect("equal")
    finish(ax)
    ax.scatter([0], [0], s=60, color=WAVE, zorder=3)
    ax.scatter([4], [2], s=60, color=TEAL, zorder=3)
    ax.annotate("原点常在左下或中心", xy=(0, 0), xytext=(0.7, -0.2), color=WAVE, fontsize=10)
    ax.annotate("", xy=(5.5, 0), xytext=(0.3, 0),
                arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.4))
    ax.annotate("", xy=(0, 3.8), xytext=(0, 0.3),
                arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.4))
    ax.text(5.2, 0.25, "x 向右", color=MUTED, fontsize=11)
    ax.text(0.2, 3.6, "y 向上", color=MUTED, fontsize=11)
    ax.set_title("常见数学坐标（对比用）")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    save_fig(fig, "ch01_axes.png")


def plot_grid_indexed():
    """小网格标出部分像素坐标。"""
    rows, cols = 5, 7
    grid = np.zeros((rows, cols, 3), dtype=np.uint8)
    for r in range(rows):
        for c in range(cols):
            # 棋盘底色
            base = 235 if (r + c) % 2 == 0 else 210
            grid[r, c] = (base, base - 8, base - 16)
    grid[0, 0] = (40, 77, 194)
    grid[2, 4] = (102, 107, 14)

    # 放大显示
    scale = 48
    big = cv2.resize(grid, (cols * scale, rows * scale), interpolation=cv2.INTER_NEAREST)
    for r in range(rows):
        for c in range(cols):
            label = f"({c},{r})"
            color = (255, 255, 255) if (r, c) in ((0, 0), (2, 4)) else (40, 40, 40)
            cv2.putText(
                big,
                label,
                (c * scale + 4, r * scale + scale // 2 + 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                color,
                1,
                cv2.LINE_AA,
            )
    save_bgr("ch01_grid.png", big)

    fig, ax = plt.subplots(figsize=(7.2, 4.0), layout="constrained")
    ax.imshow(big[:, :, ::-1])
    ax.set_title("每个格子是一个像素；标签是 (x, y) = (列, 行)")
    ax.axis("off")
    save_fig(fig, "ch01_grid_caption.png")


def plot_translate_rotate_scale():
    card = make_card()
    canvas_w, canvas_h = 420, 320
    base = place_on_canvas(card, (canvas_w, canvas_h), (60, 50))
    cx, cy = 60 + card.shape[1] // 2, 50 + card.shape[0] // 2

    panels = [
        ("原图", np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)),
        ("平移 tx=50, ty=30", similarity_matrix(cx, cy, 0, 1.0, 50, 30)),
        ("旋转 35°（绕中心）", similarity_matrix(cx, cy, 35, 1.0, 0, 0)),
        ("缩放 0.7（绕中心）", similarity_matrix(cx, cy, 0, 0.7, 0, 0)),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(9.5, 7.2), layout="constrained")
    for ax, (title, m) in zip(axes.ravel(), panels):
        out = warp(base, m, (canvas_w, canvas_h))
        # 画中心点
        out = out.copy()
        cv2.circle(out, (int(cx), int(cy)), 4, TEAL_BGR, -1)
        ax.imshow(out[:, :, ::-1])
        ax.set_title(title)
        ax.axis("off")
    save_fig(fig, "ch01_trs.png")
    save_bgr("ch01_card.png", card)


def plot_forward_vs_inverse():
    """示意：正向映射会留洞；反向映射逐像素去取样。"""
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.0), layout="constrained")

    ax = axes[0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.set_aspect("equal")
    ax.set_title("正向：源像素推到目标（可能留洞）")
    # 源网格点
    for x in range(1, 5):
        for y in range(1, 5):
            ax.scatter([x], [y], color=WAVE, s=28)
            ax.annotate(
                "",
                xy=(x + 3.2 + 0.15 * x, y + 0.9 + 0.1 * y),
                xytext=(x, y),
                arrowprops=dict(arrowstyle="->", color=MUTED, lw=0.9),
            )
    ax.text(2, 0.4, "源图", color=WAVE, ha="center")
    ax.text(7.2, 0.4, "目标图（点与点之间可能空着）", color=TEAL, ha="center")
    ax.axis("off")

    ax = axes[1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.set_aspect("equal")
    ax.set_title("反向：目标像素拉回源图取样")
    for x in range(5, 9):
        for y in range(2, 6):
            ax.scatter([x], [y], color=TEAL, s=28)
            ax.annotate(
                "",
                xy=(x - 3.5, y - 0.8),
                xytext=(x, y),
                arrowprops=dict(arrowstyle="->", color=MUTED, lw=0.9),
            )
    ax.text(2.2, 0.4, "源图（在非整数位置插值）", color=WAVE, ha="center")
    ax.text(7, 0.4, "目标图（每个像素都填上）", color=TEAL, ha="center")
    ax.axis("off")

    save_fig(fig, "ch01_resample.png")


def plot_formula_card():
    """把三种刚体/相似变换写成坐标式。"""
    fig, ax = plt.subplots(figsize=(8.8, 3.8), layout="constrained")
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4.2)
    lines = [
        r"平移：  $x' = x + t_x$，  $y' = y + t_y$",
        r"缩放：  $x' = c_x + s(x-c_x)$，  $y' = c_y + s(y-c_y)$",
        "旋转：先把点移到相对中心，按角度转，再移回去",
        "OpenCV：getRotationMatrix2D + warpAffine（2×3 矩阵）",
    ]
    for i, text in enumerate(lines):
        ax.text(0.35, 3.45 - i * 0.9, text, fontsize=13, color="#1c1915")
    ax.set_title("本篇用到的坐标公式（人话版）")
    save_fig(fig, "ch01_formulas.png")


def plot_mine():
    card = make_card()
    canvas_w, canvas_h = 420, 320
    base = place_on_canvas(card, (canvas_w, canvas_h), (60, 50))
    cx, cy = 60 + card.shape[1] // 2, 50 + card.shape[0] // 2
    m = similarity_matrix(cx, cy, ANGLE_DEG, SCALE, TX, TY)
    out = warp(base, m, (canvas_w, canvas_h))
    out = out.copy()
    cv2.circle(out, (int(cx), int(cy)), 4, TEAL_BGR, -1)

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.0), layout="constrained")
    axes[0].imshow(base[:, :, ::-1])
    axes[0].set_title("原图")
    axes[0].axis("off")
    axes[1].imshow(out[:, :, ::-1])
    axes[1].set_title(f"你的参数：tx={TX}, ty={TY}, θ={ANGLE_DEG}°, s={SCALE}")
    axes[1].axis("off")
    save_fig(fig, "ch01_my_transform.png")


def self_check():
    # 平移：点 (10,20) + (3,-5) -> (13,15)
    x, y = 10, 20
    tx, ty = 3, -5
    assert (x + tx, y + ty) == (13, 15)

    # 绕原点旋转 90° 逆时针（数学坐标 y 向上）：(1,0)->(0,1)
    # 图像坐标 y 向下时，屏幕上的「逆时针」与矩阵符号要小心；这里只检查 OpenCV 矩阵可生成
    m = cv2.getRotationMatrix2D((0.0, 0.0), 90, 1.0)
    assert m.shape == (2, 3)

    card = make_card()
    assert card.shape[0] > 0 and card.shape[1] > 0
    print("自检通过：平移算术正确，OpenCV 能生成 2×3 旋转矩阵。")


def main():
    use_style()
    self_check()
    plot_coordinate_axes()
    plot_grid_indexed()
    plot_translate_rotate_scale()
    plot_forward_vs_inverse()
    plot_formula_card()
    plot_mine()
    print("课文对照图已写入 assets/figures/ 与 assets/images/")
    print("你的图：ch01_my_transform.png")


if __name__ == "__main__":
    main()
