"""第 17 篇：稠密光流——从方程到 Farneback。

在「视频图片拼接防抖」文件夹里运行：

    MPLCONFIGDIR=/tmp/mpl-cache .venv/bin/python code/ch17_dense_flow.py

图会保存到 assets/figures/ 与 assets/images/。
顶部「改这里」只影响 ch17_my_flow.png（以及同名说明）。
课文对照图每次都按固定参数重画，方便和网页正文对得上。
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

from plot_style import INK, MUTED, TEAL, WAVE, finish, use_style

WAVE_BGR = (40, 77, 194)
TEAL_BGR = (102, 107, 14)

# ===== 改这里 =====
# 只影响 ch17_my_flow.png：合成第二帧的平移，以及 Farneback 参数
MY_SHIFT_X = 6.0  # 第二帧相对第一帧向右平移多少像素（真值）
MY_SHIFT_Y = -3.0  # 向上为负（图像坐标 y 向下为正）
MY_PYR_SCALE = 0.5
MY_LEVELS = 3
MY_WINSIZE = 15
MY_ITERATIONS = 3
MY_POLY_N = 5
MY_POLY_SIGMA = 1.2
# ==================

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "assets" / "figures"
IMG = ROOT / "assets" / "images"


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


def make_textured_scene(w=320, h=240):
    """带斜条纹、圆点、方块的合成场景，方便光流「抓住」纹理。"""
    img = np.full((h, w), 210, dtype=np.uint8)
    yy, xx = np.mgrid[0:h, 0:w]
    # 斜条纹：梯度方向与位移不完全平行，避免孔径问题太严重
    stripe = ((xx * 0.35 + yy * 0.55).astype(np.int32) // 12) % 2
    img[stripe == 0] = 170
    img[stripe == 1] = 230

    # 几块实心几何：边界强梯度
    cv2.rectangle(img, (40, 50), (110, 130), 90, -1)
    cv2.rectangle(img, (40, 50), (110, 130), 40, 2)
    cv2.circle(img, (210, 90), 38, 60, -1)
    cv2.circle(img, (210, 90), 38, 30, 2)
    # 棋盘角：利于多项式拟合
    for i in range(6):
        for j in range(4):
            x0, y0 = 50 + i * 18, 160 + j * 16
            if (i + j) % 2 == 0:
                img[y0 : y0 + 14, x0 : x0 + 16] = 40
            else:
                img[y0 : y0 + 14, x0 : x0 + 16] = 250

    # 轻微噪声，更接近真实图
    rng = np.random.default_rng(7)
    noise = rng.normal(0, 3.0, img.shape)
    img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return img


def shift_gray(img, dx, dy):
    """亚像素平移；用 warpAffine，边界用复制。"""
    h, w = img.shape[:2]
    m = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(
        img,
        m,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def make_pair(dx=5.0, dy=-2.0, w=320, h=240):
    prev = make_textured_scene(w, h)
    nxt = shift_gray(prev, dx, dy)
    return prev, nxt


def flow_to_bgr(flow, max_mag=None):
    """Middlebury 风格色轮：色相=方向，明度=幅度。返回 BGR。"""
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1], angleInDegrees=True)
    if max_mag is None:
        max_mag = float(np.percentile(mag, 99))
        max_mag = max(max_mag, 1e-3)
    hsv = np.zeros((*flow.shape[:2], 3), dtype=np.uint8)
    hsv[..., 0] = (ang / 2).astype(np.uint8)  # OpenCV H: 0–180
    hsv[..., 1] = 255
    hsv[..., 2] = np.clip(mag / max_mag * 255, 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def draw_flow_arrows(gray_bgr, flow, step=16, scale=2.5, color=WAVE_BGR):
    """在灰图上画稀疏箭头，便于肉眼读方向。"""
    out = gray_bgr.copy()
    h, w = flow.shape[:2]
    for y in range(step // 2, h, step):
        for x in range(step // 2, w, step):
            fx, fy = flow[y, x]
            if abs(fx) + abs(fy) < 0.15:
                continue
            x2 = int(round(x + fx * scale))
            y2 = int(round(y + fy * scale))
            cv2.arrowedLine(out, (x, y), (x2, y2), color, 1, tipLength=0.35)
    return out


def run_farneback(
    prev,
    nxt,
    pyr_scale=0.5,
    levels=3,
    winsize=15,
    iterations=3,
    poly_n=5,
    poly_sigma=1.2,
    flags=0,
):
    return cv2.calcOpticalFlowFarneback(
        prev,
        nxt,
        None,
        pyr_scale,
        levels,
        winsize,
        iterations,
        poly_n,
        poly_sigma,
        flags,
    )


def plot_sparse_vs_dense():
    """概念：稀疏几个箭头 vs 每像素一张「地毯」。"""
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.0), layout="constrained")

    ax = axes[0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.set_aspect("equal")
    ax.add_patch(Rectangle((1, 1), 8, 5, fill=False, ec=INK, lw=1.6))
    # 稀疏点
    pts = [(2.5, 2.2), (4.0, 4.5), (6.2, 3.0), (7.5, 5.2), (3.2, 5.5)]
    for x, y in pts:
        ax.annotate(
            "",
            xy=(x + 1.1, y + 0.35),
            xytext=(x, y),
            arrowprops=dict(arrowstyle="->", color=WAVE, lw=2.0),
        )
        ax.scatter([x], [y], color=WAVE, s=28, zorder=3)
    ax.set_title("稀疏光流：只在少数点上估 (u, v)")
    ax.text(5, 0.35, "适合跟踪特征点、估全局变换", ha="center", color=MUTED, fontsize=11)
    ax.axis("off")

    ax = axes[1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.set_aspect("equal")
    ax.add_patch(Rectangle((1, 1), 8, 5, fill=False, ec=INK, lw=1.6))
    rng = np.random.default_rng(1)
    for y in np.linspace(1.6, 5.5, 7):
        for x in np.linspace(1.6, 8.4, 10):
            jx = 0.15 * rng.normal()
            jy = 0.1 * rng.normal()
            ax.annotate(
                "",
                xy=(x + 0.55 + jx, y + 0.18 + jy),
                xytext=(x, y),
                arrowprops=dict(arrowstyle="->", color=TEAL, lw=1.0),
            )
    ax.set_title("稠密光流：几乎每个像素都有 (u, v)")
    ax.text(5, 0.35, "适合看整幅运动场、运动分割", ha="center", color=MUTED, fontsize=11)
    ax.axis("off")

    save_fig(fig, "ch17_sparse_vs_dense.png")


def plot_brightness_and_aperture():
    """亮度守恒 + 孔径问题：一条线只能约束法向运动。"""
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.0), layout="constrained")

    ax = axes[0]
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 6)
    ax.set_aspect("equal")
    # 灰度坡：左暗右亮
    for i, g in enumerate(np.linspace(0.35, 0.9, 8)):
        ax.add_patch(Rectangle((0.5 + i * 0.85, 1.2), 0.85, 3.6, color=str(g), ec="none"))
    ax.annotate(
        "",
        xy=(5.2, 3.0),
        xytext=(3.0, 3.0),
        arrowprops=dict(arrowstyle="->", color=WAVE, lw=2.4),
    )
    ax.text(4.1, 3.5, "物体右移", color=WAVE, ha="center", fontsize=12)
    ax.text(4.0, 0.45, "亮度守恒：同一点灰度沿运动轨迹不变", ha="center", color=MUTED, fontsize=11)
    ax.set_title("亮度守恒直觉")
    ax.axis("off")

    ax = axes[1]
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 6)
    ax.set_aspect("equal")
    # 竖直边缘
    ax.add_patch(Rectangle((3.4, 1.0), 0.35, 4.0, color="#3a3530", ec="none"))
    ax.add_patch(Circle((3.55, 3.0), 1.1, fill=False, ec=TEAL, lw=2.0, ls="--"))
    # 多种可能位移：法向确定、切向自由
    for dy, label, col in ((0, "法向", WAVE), (0.9, "斜上？", MUTED), (-0.9, "斜下？", MUTED)):
        ax.annotate(
            "",
            xy=(3.55 + 1.4, 3.0 + dy),
            xytext=(3.55, 3.0),
            arrowprops=dict(arrowstyle="->", color=col, lw=1.8),
        )
        ax.text(5.3, 3.0 + dy, label, color=col, va="center", fontsize=11)
    ax.text(4.0, 0.35, "小孔只看见边缘：切向怎么滑都像同一条线", ha="center", color=MUTED, fontsize=11)
    ax.set_title("孔径问题：一个方程，两个未知数")
    ax.axis("off")

    save_fig(fig, "ch17_brightness_aperture.png")


def plot_horn_schunck_energy():
    """数据项 + 平滑项：不用变分，只画「拉扯」直觉。"""
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.2), layout="constrained")

    ax = axes[0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.set_aspect("equal")
    ax.set_title("数据项：让亮度方程尽量成立")
    # 网格上的向量，故意杂乱
    rng = np.random.default_rng(2)
    for y in np.linspace(1.5, 5.5, 5):
        for x in np.linspace(1.5, 8.5, 7):
            ux = 0.9 + 0.6 * rng.normal()
            uy = 0.1 + 0.5 * rng.normal()
            ax.annotate(
                "",
                xy=(x + ux, y + uy),
                xytext=(x, y),
                arrowprops=dict(arrowstyle="->", color=WAVE, lw=1.2),
            )
    ax.text(5, 0.5, "只信亮度 → 平坦区噪声大、边缘切向不稳", ha="center", color=MUTED, fontsize=11)
    ax.axis("off")

    ax = axes[1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.set_aspect("equal")
    ax.set_title("再加平滑项：邻居的 (u,v) 不要差太狠")
    for y in np.linspace(1.5, 5.5, 5):
        for x in np.linspace(1.5, 8.5, 7):
            ax.annotate(
                "",
                xy=(x + 1.05, y + 0.25),
                xytext=(x, y),
                arrowprops=dict(arrowstyle="->", color=TEAL, lw=1.4),
            )
    # 弹簧示意
    ax.plot([3.2, 4.3], [3.5, 3.7], color=INK, lw=1.2, ls=":")
    ax.plot([4.3, 5.4], [3.7, 3.9], color=INK, lw=1.2, ls=":")
    ax.text(5, 0.5, "像邻居手拉手：场变平滑，但运动边界也会被抹一点", ha="center", color=MUTED, fontsize=11)
    ax.axis("off")

    save_fig(fig, "ch17_horn_schunck_energy.png")


def plot_farneback_poly_idea():
    """多项式曲面：局部用二次曲面拟合灰度，再比两帧系数。"""
    fig = plt.figure(figsize=(10.2, 4.0), layout="constrained")
    ax0 = fig.add_subplot(1, 2, 1, projection="3d")
    ax1 = fig.add_subplot(1, 2, 2)

    xs = np.linspace(-1.2, 1.2, 40)
    ys = np.linspace(-1.2, 1.2, 40)
    X, Y = np.meshgrid(xs, ys)
    # 二次曲面 f = c1 + c2 x + c3 y + c4 x^2 + c5 y^2 + c6 xy
    Z1 = 0.2 + 0.4 * X - 0.15 * Y + 0.25 * X**2 + 0.1 * Y**2 - 0.05 * X * Y
    Z2 = 0.2 + 0.4 * (X - 0.35) - 0.15 * (Y + 0.1) + 0.25 * (X - 0.35) ** 2 + 0.1 * (Y + 0.1) ** 2

    ax0.plot_surface(X, Y, Z1, color="#e8a090", alpha=0.85, linewidth=0, antialiased=True)
    ax0.plot_surface(X, Y, Z2, color="#7eb8b4", alpha=0.55, linewidth=0, antialiased=True)
    ax0.set_title("局部灰度 ≈ 二次多项式曲面")
    ax0.set_xticks([])
    ax0.set_yticks([])
    ax0.set_zticks([])
    ax0.set_xlabel("x")
    ax0.set_ylabel("y")

    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 7)
    ax1.set_aspect("equal")
    ax1.set_title("白话：先拟合「小丘」，再把丘对齐")
    # 两帧窗口示意
    ax1.add_patch(Rectangle((1.2, 2.0), 3.2, 3.2, fill=False, ec=WAVE, lw=2))
    ax1.add_patch(Rectangle((5.6, 2.0), 3.2, 3.2, fill=False, ec=TEAL, lw=2))
    ax1.text(2.8, 5.5, "帧 t 系数", color=WAVE, ha="center")
    ax1.text(7.2, 5.5, "帧 t+1 系数", color=TEAL, ha="center")
    ax1.annotate(
        "",
        xy=(5.4, 3.6),
        xytext=(4.5, 3.6),
        arrowprops=dict(arrowstyle="->", color=INK, lw=2),
    )
    ax1.text(5.0, 4.1, "位移", ha="center", fontsize=11)
    ax1.text(
        5,
        0.55,
        "Farneback：系数差 → 推 (u,v)，再在邻域里平滑",
        ha="center",
        color=MUTED,
        fontsize=11,
    )
    ax1.axis("off")

    save_fig(fig, "ch17_farneback_poly.png")


def make_color_wheel_bgr(size=220):
    """标准色轮：中心幅度 0（黑），外圈幅度大。"""
    half = size // 2
    yy, xx = np.mgrid[-half:half, -half:half]
    fx = xx.astype(np.float32) / half
    fy = yy.astype(np.float32) / half
    flow = np.dstack([fx, fy])
    # 圆外置零
    mag = np.sqrt(fx**2 + fy**2)
    flow[mag > 1] = 0
    bgr = flow_to_bgr(flow, max_mag=1.0)
    bgr[mag > 1] = 251
    return bgr


def plot_colorwheel():
    wheel = make_color_wheel_bgr(240)
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.2), layout="constrained")
    axes[0].imshow(wheel[:, :, ::-1])
    axes[0].set_title("色轮：颜色=方向，亮度=快慢")
    axes[0].axis("off")
    # 标注四个主方向
    axes[0].annotate("→ +u", xy=(200, 120), xytext=(210, 40), color=INK, fontsize=11,
                     arrowprops=dict(arrowstyle="->", color=INK))
    axes[0].annotate("↓ +v", xy=(120, 200), xytext=(20, 210), color=INK, fontsize=11,
                     arrowprops=dict(arrowstyle="->", color=INK))

    ax = axes[1]
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.set_title("读图口诀")
    tips = [
        "色相转一圈 = 方向转 360°",
        "越亮（或越饱和）= 位移越大",
        "接近黑色 = 几乎静止",
        "箭头图适合抽样子看；色轮适合看整幅场",
    ]
    for i, t in enumerate(tips):
        ax.text(0.6, 5.5 - i * 1.2, f"· {t}", fontsize=13, color=INK)
    save_fig(fig, "ch17_flow_colorwheel.png")
    save_bgr("ch17_colorwheel.png", wheel)


def farneback_demo_figures():
    """主结果：真值平移 vs Farneback 估计（色轮 + 箭头）。"""
    dx, dy = 5.0, -2.5
    prev, nxt = make_pair(dx, dy)
    flow = run_farneback(prev, nxt)

    prev_bgr = cv2.cvtColor(prev, cv2.COLOR_GRAY2BGR)
    nxt_bgr = cv2.cvtColor(nxt, cv2.COLOR_GRAY2BGR)
    pair = np.full((240, 320 * 2 + 12, 3), 251, dtype=np.uint8)
    pair[:, :320] = prev_bgr
    pair[:, 332:] = nxt_bgr
    cv2.putText(pair, "t", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, WAVE_BGR, 2, cv2.LINE_AA)
    cv2.putText(pair, "t+1", (344, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, TEAL_BGR, 2, cv2.LINE_AA)
    save_bgr("ch17_pair.png", pair)

    color = flow_to_bgr(flow, max_mag=8.0)
    arrows = draw_flow_arrows(prev_bgr, flow, step=14, scale=3.0)

    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.6), layout="constrained")
    axes[0].imshow(prev, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title(f"帧 t（真值位移 ≈ ({dx:.1f}, {dy:.1f})）")
    axes[0].axis("off")
    axes[1].imshow(color[:, :, ::-1])
    axes[1].set_title("Farneback 色轮图")
    axes[1].axis("off")
    axes[2].imshow(arrows[:, :, ::-1])
    axes[2].set_title("抽样子箭头")
    axes[2].axis("off")
    save_fig(fig, "ch17_farneback_result.png")
    save_bgr("ch17_flow_color.png", color)
    save_bgr("ch17_flow_arrows.png", arrows)

    # 误差图：与真值常数场比较
    gt = np.zeros_like(flow)
    gt[..., 0] = dx
    gt[..., 1] = dy
    err = np.linalg.norm(flow - gt, axis=2)
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8), layout="constrained")
    im0 = axes[0].imshow(err, cmap="magma")
    axes[0].set_title("端点误差 |flow − 真值|（像素）")
    axes[0].axis("off")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    # 直方图
    axes[1].hist(err.ravel(), bins=40, color=TEAL, alpha=0.85)
    finish(axes[1])
    axes[1].set_xlabel("EPE（像素）")
    axes[1].set_ylabel("像素个数")
    axes[1].set_title(f"中位误差 ≈ {np.median(err):.2f} px")
    save_fig(fig, "ch17_flow_error.png")

    return flow, dx, dy


def plot_params_compare():
    """拧 winsize / levels：看场变平滑或糊边界。"""
    prev, nxt = make_pair(6.0, -3.0)
    configs = [
        ("winsize=5\n（更贴细节，更噪）", dict(winsize=5, levels=3)),
        ("winsize=21\n（更平滑）", dict(winsize=21, levels=3)),
        ("levels=1\n（无金字塔，大位移吃力）", dict(winsize=15, levels=1)),
        ("levels=4\n（由粗到细更稳）", dict(winsize=15, levels=4)),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(9.8, 7.2), layout="constrained")
    for ax, (title, kw) in zip(axes.ravel(), configs):
        flow = run_farneback(prev, nxt, **kw)
        color = flow_to_bgr(flow, max_mag=9.0)
        ax.imshow(color[:, :, ::-1])
        ax.set_title(title, fontsize=12)
        ax.axis("off")
    save_fig(fig, "ch17_params_compare.png")


def plot_param_table_card():
    """一张「参数速查」图，正文也会有表。"""
    fig, ax = plt.subplots(figsize=(9.6, 4.6), layout="constrained")
    ax.axis("off")
    ax.set_title("calcOpticalFlowFarneback 参数速记", fontsize=14, pad=12)
    rows = [
        ("pyr_scale", "每层相对上一层的缩放，常用 0.5"),
        ("levels", "金字塔层数；位移大就加层"),
        ("winsize", "平均窗口；越大越平滑、越慢"),
        ("iterations", "每层迭代次数"),
        ("poly_n", "多项式邻域边长，常 5 或 7"),
        ("poly_sigma", "给多项式邻域的高斯 σ"),
        ("flags", "0 或 OPTFLOW_FARNEBACK_GAUSSIAN 等"),
    ]
    y = 0.88
    for name, tip in rows:
        ax.text(0.04, y, name, transform=ax.transAxes, fontsize=12, color=WAVE, fontfamily="monospace")
        ax.text(0.28, y, tip, transform=ax.transAxes, fontsize=12, color=INK)
        y -= 0.12
    save_fig(fig, "ch17_params_card.png")


def plot_mine():
    prev, nxt = make_pair(MY_SHIFT_X, MY_SHIFT_Y)
    flow = run_farneback(
        prev,
        nxt,
        pyr_scale=MY_PYR_SCALE,
        levels=MY_LEVELS,
        winsize=MY_WINSIZE,
        iterations=MY_ITERATIONS,
        poly_n=MY_POLY_N,
        poly_sigma=MY_POLY_SIGMA,
    )
    color = flow_to_bgr(flow, max_mag=max(8.0, abs(MY_SHIFT_X) + abs(MY_SHIFT_Y) + 2))
    arrows = draw_flow_arrows(cv2.cvtColor(prev, cv2.COLOR_GRAY2BGR), flow, step=14, scale=3.0)
    med_u = float(np.median(flow[..., 0]))
    med_v = float(np.median(flow[..., 1]))

    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.6), layout="constrained")
    axes[0].imshow(prev, cmap="gray")
    axes[0].set_title(f"真值 ({MY_SHIFT_X:.1f}, {MY_SHIFT_Y:.1f})")
    axes[0].axis("off")
    axes[1].imshow(color[:, :, ::-1])
    axes[1].set_title(f"估计中位 ({med_u:.2f}, {med_v:.2f})")
    axes[1].axis("off")
    axes[2].imshow(arrows[:, :, ::-1])
    axes[2].set_title(
        f"win={MY_WINSIZE}, lev={MY_LEVELS}, poly_n={MY_POLY_N}"
    )
    axes[2].axis("off")
    save_fig(fig, "ch17_my_flow.png")
    return med_u, med_v


def self_check():
    prev, nxt = make_pair(4.0, -2.0, w=160, h=120)
    if prev.shape != (120, 160) or nxt.shape != (120, 160):
        raise SystemExit("合成帧尺寸不对")

    flow = run_farneback(prev, nxt, levels=2, winsize=11)
    if flow.shape != (120, 160, 2):
        raise SystemExit(f"光流形状不对：{flow.shape}")
    if flow.dtype not in (np.float32, np.float64):
        raise SystemExit(f"光流 dtype 不对：{flow.dtype}")

    med = np.median(flow.reshape(-1, 2), axis=0)
    # 允许边界与算法误差：中位估计应靠近真值
    if abs(med[0] - 4.0) > 1.5 or abs(med[1] - (-2.0)) > 1.5:
        raise SystemExit(f"平移恢复偏差过大：median={med}")

    color = flow_to_bgr(flow)
    if color.shape != (120, 160, 3):
        raise SystemExit("色轮可视化失败")

    # 齐次/多项式无关：确认 BGR 写读
    probe = np.zeros((8, 8, 3), dtype=np.uint8)
    probe[:] = WAVE_BGR
    IMG.mkdir(parents=True, exist_ok=True)
    p = IMG / "_ch17_probe.png"
    cv2.imwrite(str(p), probe)
    back = cv2.imread(str(p))
    p.unlink(missing_ok=True)
    if back is None:
        raise SystemExit("OpenCV 读写失败")

    print("自检通过：Farneback 输出 HxWx2，合成平移可恢复，色轮可生成。")


def main():
    use_style()
    self_check()
    plot_sparse_vs_dense()
    plot_brightness_and_aperture()
    plot_horn_schunck_energy()
    plot_farneback_poly_idea()
    plot_colorwheel()
    farneback_demo_figures()
    plot_params_compare()
    plot_param_table_card()
    plot_mine()
    print("课文对照图已写入 assets/figures/ 与 assets/images/")
    print("你的图：ch17_my_flow.png（只跟随顶部「改这里」）")


if __name__ == "__main__":
    main()
