"""第 2 篇：仿射、透视与齐次坐标。

    python code/ch02_affine.py

顶部「改这里」只影响 ch02_my_perspective.png。
"""

from pathlib import Path

SHOW_WINDOW = False

import matplotlib

if not SHOW_WINDOW:
    matplotlib.use("Agg")

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon

from plot_style import MUTED, TEAL, WAVE, use_style

# ===== 改这里 =====
# 目标四边形四个角（像素），顺序：左上、右上、右下、左下
DST_QUAD = np.array(
    [
        [80.0, 60.0],
        [340.0, 40.0],
        [360.0, 260.0],
        [60.0, 240.0],
    ],
    dtype=np.float32,
)
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


def make_floor_tile(w=200, h=160):
    """带透视感棋盘的「地板砖」源图，便于看出仿射/透视差别。"""
    img = np.full((h, w, 3), 245, dtype=np.uint8)
    cell = 20
    for y in range(0, h, cell):
        for x in range(0, w, cell):
            if ((x // cell) + (y // cell)) % 2 == 0:
                img[y : y + cell, x : x + cell] = (220, 208, 190)
    cv2.rectangle(img, (6, 6), (w - 7, h - 7), (50, 45, 40), 2)
    cv2.putText(img, "A", (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, WAVE_BGR, 2, cv2.LINE_AA)
    cv2.putText(img, "B", (w - 40, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, TEAL_BGR, 2, cv2.LINE_AA)
    cv2.putText(img, "D", (16, h - 16), cv2.FONT_HERSHEY_SIMPLEX, 1.0, WAVE_BGR, 2, cv2.LINE_AA)
    cv2.putText(img, "C", (w - 40, h - 16), cv2.FONT_HERSHEY_SIMPLEX, 1.0, TEAL_BGR, 2, cv2.LINE_AA)
    # 中线帮助看切变
    cv2.line(img, (w // 2, 8), (w // 2, h - 8), (120, 110, 100), 1)
    cv2.line(img, (8, h // 2), (w - 8, h // 2), (120, 110, 100), 1)
    return img


def plot_homogeneous_idea():
    fig, ax = plt.subplots(figsize=(8.8, 3.6), layout="constrained")
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.set_title("齐次坐标：多写一个 1，平移也能进矩阵")
    ax.text(0.4, 3.1, r"普通写法：$(x', y')$ 要「乘一个矩阵」再「加一个平移」", fontsize=13)
    ax.text(0.4, 2.2, r"齐次写法：把点写成 $(x, y, 1)$，用 $3\times3$ 矩阵一次乘完", fontsize=13)
    ax.text(0.4, 1.3, r"最后用第三维去除一化：$X=x_h/w_h,\ Y=y_h/w_h$", fontsize=13)
    ax.text(0.4, 0.5, "仿射：最后一行固定为 [0, 0, 1]；透视：最后一行也可学习", fontsize=12, color=MUTED)
    save_fig(fig, "ch02_homogeneous.png")


def plot_affine_vs_perspective_shapes():
    """几何形状：仿射保平行，透视可成任意凸四边形。"""
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.6), layout="constrained")

    def draw_quad(ax, pts, title, facecolor):
        ax.add_patch(Polygon(pts, closed=True, facecolor=facecolor, edgecolor="#1c1915", lw=1.6, alpha=0.85))
        for i, (x, y) in enumerate(pts):
            ax.scatter([x], [y], color=WAVE if i % 2 == 0 else TEAL, s=36, zorder=3)
            ax.text(x + 0.08, y + 0.08, "ABCD"[i], fontsize=11)
        ax.set_xlim(-0.5, 4.5)
        ax.set_ylim(-0.5, 3.5)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(title)

    square = [(0.8, 0.6), (3.2, 0.6), (3.2, 2.6), (0.8, 2.6)]
    parallelogram = [(0.6, 0.5), (3.0, 0.9), (3.4, 2.7), (1.0, 2.3)]
    taper = [(0.7, 0.5), (3.3, 0.8), (2.7, 2.8), (1.1, 2.5)]

    draw_quad(axes[0], square, "原矩形", "#f3e6d8")
    draw_quad(axes[1], parallelogram, "仿射后：仍是平行四边形", "#e5f3f1")
    draw_quad(axes[2], taper, "透视后：可成梯形", "#fde8df")
    save_fig(fig, "ch02_shapes.png")


def plot_affine_demo():
    src = make_floor_tile()
    h, w = src.shape[:2]
    src_tri = np.float32([[0, 0], [w - 1, 0], [0, h - 1]])
    # 切变 + 微旋转感：三个点够定仿射
    dst_tri = np.float32([[20, 30], [w + 10, 10], [50, h + 20]])
    m = cv2.getAffineTransform(src_tri, dst_tri)
    canvas = (360, 280)
    out = cv2.warpAffine(src, m, canvas, borderValue=(251, 251, 251))

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.0), layout="constrained")
    axes[0].imshow(src[:, :, ::-1])
    axes[0].set_title("源图（用 3 对点定仿射）")
    axes[0].axis("off")
    axes[1].imshow(out[:, :, ::-1])
    axes[1].set_title("仿射 warp：平行线仍平行")
    axes[1].axis("off")
    save_fig(fig, "ch02_affine_warp.png")
    save_bgr("ch02_tile.png", src)

    # 打印矩阵到一张注释图
    fig, ax = plt.subplots(figsize=(7.2, 2.8), layout="constrained")
    ax.axis("off")
    ax.set_title("本例 getAffineTransform 得到的 2×3 矩阵")
    text = "\n".join(
        "  ".join(f"{m[r, c]:8.3f}" for c in range(3)) for r in range(2)
    )
    ax.text(0.05, 0.45, text, fontsize=14, family="monospace", transform=ax.transAxes)
    ax.text(0.05, 0.12, "最后一列是平移；左侧 2×2 含旋转、缩放、切变", fontsize=12, color=MUTED, transform=ax.transAxes)
    save_fig(fig, "ch02_affine_matrix.png")


def plot_perspective_demo(dst_quad=None, name="ch02_perspective_warp.png", title_suffix=""):
    src = make_floor_tile()
    h, w = src.shape[:2]
    src_quad = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    if dst_quad is None:
        dst_quad = np.float32(
            [
                [70, 80],
                [300, 30],
                [330, 250],
                [40, 230],
            ]
        )
    else:
        dst_quad = np.asarray(dst_quad, dtype=np.float32)

    h_mat = cv2.getPerspectiveTransform(src_quad, dst_quad)
    canvas = (400, 300)
    out = cv2.warpPerspective(src, h_mat, canvas, borderValue=(251, 251, 251))

    # 在输出上描四边形
    vis = out.copy()
    cv2.polylines(vis, [dst_quad.astype(np.int32)], True, TEAL_BGR, 2)
    for i, (x, y) in enumerate(dst_quad):
        cv2.circle(vis, (int(x), int(y)), 5, WAVE_BGR if i % 2 == 0 else TEAL_BGR, -1)

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.0), layout="constrained")
    axes[0].imshow(src[:, :, ::-1])
    axes[0].set_title("源图四角 A-B-C-D")
    axes[0].axis("off")
    axes[1].imshow(vis[:, :, ::-1])
    axes[1].set_title("透视 warp 到目标四边形" + title_suffix)
    axes[1].axis("off")
    save_fig(fig, name)

    fig, ax = plt.subplots(figsize=(7.4, 3.2), layout="constrained")
    ax.axis("off")
    ax.set_title("本例 3×3 单应矩阵 H（getPerspectiveTransform）")
    lines = ["  ".join(f"{h_mat[r, c]:9.4f}" for c in range(3)) for r in range(3)]
    ax.text(0.04, 0.55, "\n".join(lines), fontsize=13, family="monospace", transform=ax.transAxes)
    ax.text(
        0.04,
        0.12,
        "第三行不再固定为 0,0,1 时，就出现近大远小（透视）",
        fontsize=12,
        color=MUTED,
        transform=ax.transAxes,
    )
    if name == "ch02_perspective_warp.png":
        save_fig(fig, "ch02_homography_matrix.png")
    else:
        plt.close(fig)
    return h_mat


def plot_dof_table():
    fig, ax = plt.subplots(figsize=(8.6, 3.2), layout="constrained")
    ax.axis("off")
    ax.set_title("自由度对照（二维）")
    col_labels = ["变换", "自由度", "最少对应点", "平行四边形？", "近大远小？"]
    rows = [
        ["平移", "2", "1", "保持", "不能"],
        ["相似（转+缩+移）", "4", "2", "保持", "不能"],
        ["仿射", "6", "3", "保持平行", "不能"],
        ["透视 / 单应", "8", "4", "不要求", "能"],
    ]
    table = ax.table(cellText=rows, colLabels=col_labels, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.15, 1.55)
    save_fig(fig, "ch02_dof.png")


def plot_mine():
    plot_perspective_demo(DST_QUAD, name="ch02_my_perspective.png", title_suffix="（你的四角）")


def self_check():
    # 齐次：平移矩阵乘 (x,y,1)
    tx, ty = 3.0, -2.0
    H = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]], dtype=np.float64)
    p = np.array([10.0, 20.0, 1.0])
    q = H @ p
    assert abs(q[0] - 13) < 1e-9 and abs(q[1] - 18) < 1e-9 and abs(q[2] - 1) < 1e-9

    src = np.float32([[0, 0], [10, 0], [0, 10]])
    dst = np.float32([[1, 2], [11, 3], [2, 12]])
    m = cv2.getAffineTransform(src, dst)
    assert m.shape == (2, 3)

    sq = np.float32([[0, 0], [10, 0], [10, 10], [0, 10]])
    H2 = cv2.getPerspectiveTransform(sq, sq)
    assert H2.shape == (3, 3)
    print("自检通过：齐次平移、仿射与透视矩阵可生成。")


def main():
    use_style()
    self_check()
    plot_homogeneous_idea()
    plot_affine_vs_perspective_shapes()
    plot_dof_table()
    plot_affine_demo()
    plot_perspective_demo()
    plot_mine()
    print("课文对照图已写入 assets/figures/")
    print("你的图：ch02_my_perspective.png")


if __name__ == "__main__":
    main()
