"""第 14 篇：光流是什么——亮度守恒与孔径问题。

    MPLCONFIGDIR=/tmp/mpl-cache .venv/bin/python code/ch14_flow_intro.py

顶部「改这里」只影响 ch14_my_blob.png / ch14_my_constraint.png。
本篇几乎不算算法，只生成概念对照图。
"""

from pathlib import Path

SHOW_WINDOW = False

import matplotlib

if not SHOW_WINDOW:
    matplotlib.use("Agg")

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Rectangle

from plot_style import INK, MUTED, TEAL, WAVE, use_style

# ===== 改这里 =====
# 小亮斑在两帧之间的像素位移（表观运动）
BLOB_SHIFT_X = 12.0
BLOB_SHIFT_Y = -6.0
# 约束方程示意：I_x u + I_y v = -I_t 里的空间/时间梯度
GRADIENT_IX = 2.0
GRADIENT_IY = 0.5
GRADIENT_IT = -8.0
# ==================

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "assets" / "figures"
IMG = ROOT / "assets" / "images"
WAVE_BGR = (40, 77, 194)
TEAL_BGR = (102, 107, 14)
INK_BGR = (21, 25, 28)


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


def make_blob_frame(h=160, w=240, cx=90.0, cy=80.0, radius=18.0):
    """合成一帧：暗底 + 一个高斯亮斑。"""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    dist2 = (xx - cx) ** 2 + (yy - cy) ** 2
    blob = np.exp(-dist2 / (2.0 * radius * radius))
    gray = (40 + 200 * blob).astype(np.uint8)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def plot_blob_shift(dx=12.0, dy=-6.0, name="ch14_blob_shift.png", title_note=""):
    """图：两帧亮斑与表观位移箭头。"""
    h, w = 160, 240
    cx0, cy0 = 90.0, 80.0
    f0 = make_blob_frame(h, w, cx0, cy0)
    f1 = make_blob_frame(h, w, cx0 + dx, cy0 + dy)

    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.5), layout="constrained")
    axes[0].imshow(f0[:, :, ::-1])
    axes[0].set_title("第 t 帧")
    axes[0].axis("off")
    axes[1].imshow(f1[:, :, ::-1])
    axes[1].set_title("第 t+1 帧")
    axes[1].axis("off")

    overlay = (0.55 * f0.astype(np.float32) + 0.45 * f1.astype(np.float32)).astype(np.uint8)
    axes[2].imshow(overlay[:, :, ::-1])
    axes[2].annotate(
        "",
        xy=(cx0 + dx, cy0 + dy),
        xytext=(cx0, cy0),
        arrowprops=dict(arrowstyle="->", color=WAVE, lw=2.2),
    )
    axes[2].plot([cx0], [cy0], "o", color=TEAL, ms=6)
    axes[2].plot([cx0 + dx], [cy0 + dy], "o", color=WAVE, ms=6)
    axes[2].set_title(f"表观位移 (u, v)=({dx:.0f}, {dy:.0f}){title_note}")
    axes[2].axis("off")
    save_fig(fig, name)
    save_bgr("ch14_blob_t0.png", f0)
    save_bgr("ch14_blob_t1.png", f1)
    return dx, dy


def plot_apparent_vs_3d():
    """钉死：光流是图像平面表观运动，不等于真实 3D 速度。"""
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.2), layout="constrained")

    ax = axes[0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("同一表观位移，可以有很多 3D 原因")
    # 相机与成像平面
    ax.add_patch(Rectangle((0.6, 2.8), 0.5, 1.4, facecolor="#ddd4c4", edgecolor=INK, lw=1.2))
    ax.text(0.35, 4.5, "相机", fontsize=11, color=MUTED)
    ax.plot([1.1, 9.2], [3.5, 6.2], color=MUTED, lw=1, ls="--")
    ax.plot([1.1, 9.2], [3.5, 0.8], color=MUTED, lw=1, ls="--")
    ax.plot([8.5, 8.5], [0.6, 6.4], color=INK, lw=1.5)
    ax.text(8.7, 6.6, "图像平面", fontsize=11)
    # 两种 3D 运动投到同一像素位移
    ax.annotate("", xy=(7.6, 4.4), xytext=(6.2, 4.4), arrowprops=dict(arrowstyle="->", color=TEAL, lw=2))
    ax.text(5.4, 4.85, "物体右移", color=TEAL, fontsize=11)
    ax.annotate("", xy=(7.6, 2.4), xytext=(7.6, 3.6), arrowprops=dict(arrowstyle="->", color=WAVE, lw=2))
    ax.text(6.5, 1.7, "相机左移\n（背景相对右移）", color=WAVE, fontsize=11, ha="center")
    ax.annotate(
        "",
        xy=(8.5, 3.9),
        xytext=(8.5, 3.1),
        arrowprops=dict(arrowstyle="->", color=INK, lw=2.4),
    )
    ax.text(8.7, 3.4, "屏幕上\n都像向下/右移", fontsize=10, color=MUTED)

    ax = axes[1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("旋转也能骗过人眼：条纹柱")
    # 简化「理发店灯柱」：竖条纹 + 旋转暗示斜向表观
    for i, x in enumerate(np.linspace(2.2, 5.0, 7)):
        color = WAVE if i % 2 == 0 else TEAL
        ax.add_patch(Rectangle((x, 1.2), 0.35, 4.6, facecolor=color, edgecolor="none", alpha=0.55))
    ax.add_patch(Rectangle((2.1, 1.1), 3.4, 4.8, fill=False, edgecolor=INK, lw=1.5))
    ax.annotate("", xy=(7.8, 4.8), xytext=(6.2, 3.0), arrowprops=dict(arrowstyle="->", color=WAVE, lw=2.2))
    ax.text(6.0, 5.4, "柱子绕竖直轴转", fontsize=11)
    ax.text(6.0, 2.2, "图像上条纹却像\n斜着往上爬", fontsize=11, color=MUTED)
    ax.text(0.4, 0.4, "结论：光流只描述「像素看上去怎么挪」，不问「世界里真实怎么动」。", fontsize=11)
    save_fig(fig, "ch14_apparent_vs_3d.png")


def plot_brightness_constancy():
    """亮度守恒：同一物理点灰度近似不变。"""
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.8), layout="constrained")

    # 左：跟随物理点，灰度不变
    ax = axes[0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")
    ax.set_title("跟随同一个物理点：灰度差不多")
    ax.add_patch(Circle((3.0, 3.0), 0.7, facecolor="#f3c7b6", edgecolor=WAVE, lw=2))
    ax.add_patch(Circle((6.8, 3.4), 0.7, facecolor="#f3c7b6", edgecolor=WAVE, lw=2))
    ax.annotate("", xy=(6.0, 3.3), xytext=(3.8, 3.1), arrowprops=dict(arrowstyle="->", color=TEAL, lw=2))
    ax.text(3.0, 1.7, "I = 180", ha="center", fontsize=12)
    ax.text(6.8, 1.7, "I ≈ 180", ha="center", fontsize=12)
    ax.text(5.0, 5.2, "亮度守恒假设", ha="center", fontsize=12, color=TEAL)

    # 右：固定像素看，灰度会变
    ax = axes[1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")
    ax.set_title("盯着同一个像素坐标：灰度会变")
    ax.add_patch(Rectangle((4.2, 2.2), 1.6, 1.6, fill=False, edgecolor=INK, lw=1.8, ls="--"))
    ax.add_patch(Circle((3.0, 3.0), 0.7, facecolor="#f3c7b6", edgecolor=WAVE, lw=1.5, alpha=0.45))
    ax.add_patch(Circle((7.0, 3.0), 0.7, facecolor="#f3c7b6", edgecolor=WAVE, lw=2))
    ax.text(5.0, 1.3, "这个格子从「暗」变成「亮」", ha="center", fontsize=11, color=MUTED)
    ax.text(5.0, 5.2, "时间梯度 I_t ≠ 0", ha="center", fontsize=12, color=WAVE)
    save_fig(fig, "ch14_brightness_constancy.png")


def plot_brightness_breaks():
    """何时亮度守恒破掉。"""
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.6), layout="constrained")
    titles = ["光照突变", "镜面高光挪位", "遮挡 / 露出"]
    notes = [
        "开灯瞬间，同一点灰度跳变",
        "高光跟着视角走，不是物体材质",
        "后面的点突然出现/消失",
    ]
    for ax, title, note in zip(axes, titles, notes):
        ax.set_xlim(0, 8)
        ax.set_ylim(0, 6)
        ax.axis("off")
        ax.set_title(title)
        if title.startswith("光照"):
            ax.add_patch(Circle((2.5, 3.0), 0.9, facecolor="#c8b8a0", edgecolor=INK, lw=1.2))
            ax.add_patch(Circle((5.5, 3.0), 0.9, facecolor="#fff3a8", edgecolor=WAVE, lw=2))
            ax.text(2.5, 1.4, "暗", ha="center")
            ax.text(5.5, 1.4, "亮", ha="center", color=WAVE)
        elif title.startswith("镜面"):
            ax.add_patch(Rectangle((1.5, 1.8), 5, 2.6, facecolor="#dfe9e8", edgecolor=INK, lw=1.2))
            ax.add_patch(Circle((3.0, 3.1), 0.45, facecolor="#fffdf8", edgecolor=WAVE, lw=2))
            ax.add_patch(Circle((5.2, 3.1), 0.45, facecolor="#fffdf8", edgecolor=TEAL, lw=2))
            ax.annotate("", xy=(4.7, 3.1), xytext=(3.5, 3.1), arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.5))
        else:
            ax.add_patch(Rectangle((1.2, 1.5), 2.4, 3.2, facecolor="#c24d28", edgecolor=INK, alpha=0.35))
            ax.add_patch(Rectangle((3.4, 1.5), 2.4, 3.2, facecolor="#0e6b66", edgecolor=INK, alpha=0.35))
            ax.add_patch(Rectangle((4.2, 2.0), 2.2, 2.4, facecolor="#c24d28", edgecolor=INK, alpha=0.7))
            ax.text(4.0, 0.7, "前景挡住背景", ha="center", fontsize=11)
        ax.text(4.0, 5.3, note, ha="center", fontsize=10, color=MUTED)
    save_fig(fig, "ch14_brightness_breaks.png")


def constraint_line_points(ix, iy, it, n=40):
    """I_x u + I_y v + I_t = 0 在 (u,v) 平面上是一条直线。"""
    # 法向方向 (ix, iy)，直线：ix*u + iy*v = -it
    normal = np.array([ix, iy], dtype=np.float64)
    nrm = np.linalg.norm(normal)
    if nrm < 1e-9:
        return None, None, None
    normal = normal / nrm
    # 特解：沿法向的「法向光流」
    # ix*u + iy*v = -it  →  u = -it * ix / (ix^2+iy^2), v 同理
    denom = ix * ix + iy * iy
    u_n = -it * ix / denom
    v_n = -it * iy / denom
    tangent = np.array([-normal[1], normal[0]])
    ts = np.linspace(-6, 6, n)
    us = u_n + ts * tangent[0] * 1.2
    vs = v_n + ts * tangent[1] * 1.2
    return (us, vs), (u_n, v_n), tangent


def plot_constraint(ix=2.0, iy=0.5, it=-8.0, name="ch14_constraint_line.png", note=""):
    line, normal_flow, tangent = constraint_line_points(ix, iy, it)
    fig, ax = plt.subplots(figsize=(6.4, 5.2), layout="constrained")
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.axvline(0, color=MUTED, lw=0.8)
    if line is not None:
        us, vs = line
        ax.plot(us, vs, color=TEAL, lw=2.4, label="约束直线：满足亮度守恒的全部 (u,v)")
        un, vn = normal_flow
        ax.annotate(
            "",
            xy=(un, vn),
            xytext=(0, 0),
            arrowprops=dict(arrowstyle="->", color=WAVE, lw=2.4),
        )
        ax.plot([un], [vn], "o", color=WAVE, ms=8, zorder=5)
        ax.text(un + 0.15, vn + 0.25, "法向光流\n（最短解）", color=WAVE, fontsize=11)
        # 切向不确定：同一直线上另画两个候选
        for s, lab in ((2.2, "也合法"), (-2.2, "也合法")):
            u2 = un + s * tangent[0]
            v2 = vn + s * tangent[1]
            ax.plot([u2], [v2], "o", color=TEAL, ms=7)
            ax.text(u2 + 0.1, v2 - 0.45, lab, color=TEAL, fontsize=10)
    ax.set_xlim(-6, 6)
    ax.set_ylim(-6, 6)
    ax.set_aspect("equal")
    ax.set_xlabel("u（水平表观速度）")
    ax.set_ylabel("v（竖直表观速度）")
    ax.set_title(f"一个像素只能钉住一条直线{note}")
    ax.text(
        0.02,
        0.02,
        f"$I_x={ix:.1f},\ I_y={iy:.1f},\ I_t={it:.1f}$",
        transform=ax.transAxes,
        fontsize=11,
        color=MUTED,
    )
    ax.legend(loc="upper right", fontsize=9, frameon=False)
    ax.grid(True, alpha=0.35)
    save_fig(fig, name)
    return normal_flow


def plot_aperture_story():
    """孔径问题：小孔只看见边 → 只看法向。"""
    fig, axes = plt.subplots(1, 3, figsize=(10.6, 3.7), layout="constrained")

    def draw_stripes(ax, phase=0.0, aperture=None):
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 7)
        ax.set_aspect("equal")
        ax.axis("off")
        for i in range(-2, 14):
            x0 = i * 0.85 + phase
            xs = [x0, x0 + 0.45, x0 + 0.45 + 0.3, x0 + 0.3]
            ys = [0, 0, 7, 7]
            # 竖条近似斜边：用矩形竖条
            ax.add_patch(Rectangle((x0, 0), 0.42, 7, facecolor=WAVE if i % 2 == 0 else "#f3e6d8", edgecolor="none"))
        if aperture is not None:
            # 遮罩：孔外变暗
            ax.add_patch(Rectangle((0, 0), 10, 7, facecolor="#1c1915", alpha=0.55))
            ax.add_patch(Circle(aperture[:2], aperture[2], facecolor="#fffdf8", edgecolor=TEAL, lw=2, alpha=1.0, zorder=3))
            # 孔内再画一次条纹局部（简化：白底+竖线）
            cx, cy, r = aperture
            ax.add_patch(Circle((cx, cy), r - 0.05, facecolor="#f3e6d8", edgecolor="none", zorder=4))
            for i in range(-2, 14):
                x0 = i * 0.85 + phase
                if abs(x0 + 0.2 - cx) < r + 0.3:
                    ax.plot([x0 + 0.2, x0 + 0.2], [cy - r * 0.85, cy + r * 0.85], color=WAVE, lw=3, zorder=5, solid_capstyle="round")

    draw_stripes(axes[0], phase=0.0, aperture=None)
    axes[0].set_title("整幅：看得出条纹怎么动")
    axes[0].annotate("", xy=(7.2, 3.5), xytext=(4.5, 3.5), arrowprops=dict(arrowstyle="->", color=TEAL, lw=2.2))
    axes[0].text(5.5, 4.2, "真实右移", color=TEAL)

    draw_stripes(axes[1], phase=0.35, aperture=(5.0, 3.5, 1.35))
    axes[1].set_title("小孔只看见一条边")
    axes[1].text(5.0, 0.55, "边沿线方向怎么滑，看起来几乎一样", ha="center", fontsize=10, color=MUTED)

    # 右：法向确定、切向扇形不确定
    ax = axes[2]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("只定法向，切向不确定")
    ax.plot([2, 8], [2.2, 5.8], color=INK, lw=3)
    ax.text(7.2, 6.2, "边缘", fontsize=11)
    # 法向箭头
    ax.annotate("", xy=(6.2, 2.8), xytext=(4.5, 3.8), arrowprops=dict(arrowstyle="->", color=WAVE, lw=2.5))
    ax.text(6.3, 2.5, "法向\n（能看出）", color=WAVE, fontsize=11)
    # 切向候选
    for ang, col in ((-25, TEAL), (0, TEAL), (25, TEAL)):
        rad = np.deg2rad(ang)
        # 沿边缘方向的扰动
        dx, dy = 1.6 * np.cos(rad), 1.0 * np.sin(rad)
        ax.annotate(
            "",
            xy=(5.0 + dx, 4.0 + dy),
            xytext=(5.0, 4.0),
            arrowprops=dict(arrowstyle="->", color=col, lw=1.5, alpha=0.85),
        )
    ax.text(5.0, 1.0, "橙色=法向分量；青色扇形=切向候选", ha="center", fontsize=10, color=MUTED)
    save_fig(fig, "ch14_aperture.png")


def plot_corner_vs_edge():
    """角点能钉死二维运动，纯边缘不能。"""
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0), layout="constrained")

    ax = axes[0]
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 6)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("只看见竖直边 → 水平能估，上下滑不定")
    ax.add_patch(Rectangle((3.5, 0.8), 0.35, 4.4, facecolor=WAVE, edgecolor=INK))
    ax.add_patch(Circle((3.7, 3.0), 1.3, fill=False, edgecolor=TEAL, lw=2, ls="--"))
    ax.annotate("", xy=(5.6, 3.0), xytext=(4.2, 3.0), arrowprops=dict(arrowstyle="->", color=WAVE, lw=2))
    ax.text(5.7, 3.2, "法向", color=WAVE)
    ax.annotate("", xy=(3.7, 4.4), xytext=(3.7, 3.3), arrowprops=dict(arrowstyle="->", color=TEAL, lw=1.6, alpha=0.7))
    ax.annotate("", xy=(3.7, 1.6), xytext=(3.7, 2.7), arrowprops=dict(arrowstyle="->", color=TEAL, lw=1.6, alpha=0.7))
    ax.text(4.2, 4.6, "切向？", color=TEAL)

    ax = axes[1]
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 6)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("看见角点 → 两个方向都有纹理变化")
    ax.plot([2.2, 5.5], [3.0, 3.0], color=WAVE, lw=4, solid_capstyle="round")
    ax.plot([5.5, 5.5], [3.0, 5.2], color=WAVE, lw=4, solid_capstyle="round")
    ax.add_patch(Circle((5.5, 3.0), 1.35, fill=False, edgecolor=TEAL, lw=2, ls="--"))
    ax.annotate("", xy=(6.8, 3.9), xytext=(5.5, 3.0), arrowprops=dict(arrowstyle="->", color=INK, lw=2.2))
    ax.text(6.9, 4.1, "二维位移\n可唯一", fontsize=11)
    ax.text(4.0, 0.7, "这就是第 15 篇要用「窗口 + 角点」的原因", ha="center", fontsize=11, color=MUTED)
    save_fig(fig, "ch14_corner_vs_edge.png")


def plot_bridge_to_lk():
    """为第 15 篇铺路：一个像素不够，窗口里叠约束。"""
    fig, ax = plt.subplots(figsize=(8.8, 3.6), layout="constrained")
    ax.axis("off")
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.set_title("从本篇到 Lucas-Kanade：还差两块积木")
    boxes = [
        (0.4, 1.5, "单像素\n一条约束直线"),
        (4.2, 1.5, "小窗口里\n假设同一 (u,v)"),
        (8.0, 1.5, "多条直线相交\n最小二乘求解"),
    ]
    for x, y, text in boxes:
        ax.add_patch(Rectangle((x, y), 3.2, 2.2, facecolor="#f7f1e8", edgecolor=INK, lw=1.3))
        ax.text(x + 1.6, y + 1.1, text, ha="center", va="center", fontsize=12)
    ax.annotate("", xy=(4.1, 2.6), xytext=(3.7, 2.6), arrowprops=dict(arrowstyle="->", color=WAVE, lw=2))
    ax.annotate("", xy=(7.9, 2.6), xytext=(7.5, 2.6), arrowprops=dict(arrowstyle="->", color=WAVE, lw=2))
    ax.text(6.0, 0.6, "第 15 篇正式做：窗口 + 最小二乘 → 稀疏光流 LK", ha="center", fontsize=12, color=TEAL)
    save_fig(fig, "ch14_bridge_lk.png")


def self_check():
    # 1) 位移语义：正向右、负向上（图像 y 向下时脚本用图像坐标）
    assert isinstance(BLOB_SHIFT_X, (int, float))

    # 2) 法向光流公式
    ix, iy, it = 3.0, 4.0, -10.0
    denom = ix * ix + iy * iy
    u_n = -it * ix / denom
    v_n = -it * iy / denom
    assert abs(u_n - 1.2) < 1e-9
    assert abs(v_n - 1.6) < 1e-9
    # 落在约束直线上
    assert abs(ix * u_n + iy * v_n + it) < 1e-9

    # 3) 切向扰动仍满足约束
    tx, ty = -iy, ix  # 切向
    for s in (-0.5, 0.5, 1.7):
        u = u_n + s * tx
        v = v_n + s * ty
        assert abs(ix * u + iy * v + it) < 1e-8

    # 4) 合成图可生成
    img = make_blob_frame()
    assert img.shape[2] == 3 and img.dtype == np.uint8

    print("自检通过：法向光流、切向不确定性、合成帧均可复现。")


def main():
    use_style()
    self_check()
    plot_blob_shift()
    plot_apparent_vs_3d()
    plot_brightness_constancy()
    plot_brightness_breaks()
    plot_constraint()
    plot_aperture_story()
    plot_corner_vs_edge()
    plot_bridge_to_lk()
    # 用户可调参数输出
    plot_blob_shift(BLOB_SHIFT_X, BLOB_SHIFT_Y, name="ch14_my_blob.png", title_note="（你的位移）")
    plot_constraint(GRADIENT_IX, GRADIENT_IY, GRADIENT_IT, name="ch14_my_constraint.png", note="（你的梯度）")
    print("课文对照图已写入 assets/figures/")
    print("你的图：ch14_my_blob.png、ch14_my_constraint.png")


if __name__ == "__main__":
    main()
