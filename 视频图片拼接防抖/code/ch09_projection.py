"""第 9 篇：投影模型——平面、柱面、球面。

在「视频图片拼接防抖」文件夹里运行：

    MPLCONFIGDIR=/tmp/mpl-cache .venv/bin/python code/ch09_projection.py

图会保存到 assets/figures/ 与 assets/images/。
顶部「改这里」只影响 ch09_my_*.png；课文对照图按固定参数重画。
"""

from pathlib import Path

SHOW_WINDOW = False

import matplotlib

if not SHOW_WINDOW:
    matplotlib.use("Agg")

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Arc, Circle, FancyBboxPatch, Rectangle

from plot_style import INK, MUTED, TEAL, WAVE, use_style

WAVE_BGR = (40, 77, 194)
TEAL_BGR = (102, 107, 14)
INK_BGR = (21, 25, 28)
MUTED_BGR = (76, 86, 92)

# ===== 改这里 =====
# 只影响 ch09_my_cylinder.png / ch09_my_sphere.png / ch09_my_compare.png
MY_FOCAL = 220.0  # 虚拟焦距（像素）。越大越「望远」；越小畸变越明显
MY_YAW_DEG = 38.0  # 第二张「照片」相对第一张水平转角（度）。越大越考验平面投影
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


# ---------------------------------------------------------------------------
# 几何：相机射线 ↔ 三种投影画布
# ---------------------------------------------------------------------------


def pixel_to_ray(x, y, cx, cy, f):
    """像素 → 相机坐标系下的方向向量（未归一化）。Z 朝前。"""
    return np.array([x - cx, y - cy, f], dtype=np.float64)


def rotate_yaw(ray, yaw_rad):
    """绕竖直 Y 轴旋转（水平转头）。"""
    c, s = np.cos(yaw_rad), np.sin(yaw_rad)
    x, y, z = ray
    return np.array([c * x + s * z, y, -s * x + c * z], dtype=np.float64)


def ray_to_planar(ray, cx, cy, f):
    """平面投影：把射线打到 Z=f 的平面，再映回像素。背后的点返回 None。"""
    x, y, z = ray
    if z <= 1e-6:
        return None
    return (f * x / z + cx, f * y / z + cy)


def ray_to_cylindrical(ray, cx, cy, f):
    """柱面投影：半径为 f 的圆柱展开。θ 水平，h 竖直。"""
    x, y, z = ray
    theta = np.arctan2(x, z)
    denom = np.hypot(x, z)
    if denom < 1e-9:
        return None
    h = y / denom
    return (f * theta + cx, f * h + cy)


def ray_to_spherical(ray, cx, cy, f):
    """球面投影：单位球经纬度展开。θ 经度，φ 纬度。"""
    x, y, z = ray
    r = np.linalg.norm(ray)
    if r < 1e-9:
        return None
    x, y, z = x / r, y / r, z / r
    theta = np.arctan2(x, z)
    phi = np.arcsin(np.clip(y, -1.0, 1.0))
    return (f * theta + cx, f * phi + cy)


def project_map(h, w, yaw_rad, f, mode, cx=None, cy=None):
    """为「从已旋转相机取景 → 映到统一投影画布」建 remap 表。

    输出画布与输入同尺寸。对画布每个像素，反求它在「当前相机图像」里的采样坐标。
    mode: planar / cylindrical / spherical
    """
    if cx is None:
        cx = (w - 1) / 2.0
    if cy is None:
        cy = (h - 1) / 2.0

    ys, xs = np.meshgrid(np.arange(h, dtype=np.float64), np.arange(w, dtype=np.float64), indexing="ij")

    # 画布像素 → 世界射线（相机朝向 0° 时）
    if mode == "planar":
        rays = np.stack([xs - cx, ys - cy, np.full_like(xs, f)], axis=-1)
    elif mode == "cylindrical":
        theta = (xs - cx) / f
        hh = (ys - cy) / f
        rays = np.stack([np.sin(theta), hh, np.cos(theta)], axis=-1)
    elif mode == "spherical":
        theta = (xs - cx) / f
        phi = (ys - cy) / f
        # 注意：经纬展开下，方向向量用球坐标
        cos_phi = np.cos(phi)
        rays = np.stack([cos_phi * np.sin(theta), np.sin(phi), cos_phi * np.cos(theta)], axis=-1)
    else:
        raise ValueError(mode)

    # 世界射线 → 当前相机坐标系（相机绕 Y 转了 yaw）
    # 相机坐标系相对世界：R_cam = R_yaw^{-1}，即对射线做 -yaw
    c, s = np.cos(-yaw_rad), np.sin(-yaw_rad)
    x, y, z = rays[..., 0], rays[..., 1], rays[..., 2]
    x2 = c * x + s * z
    z2 = -s * x + c * z
    y2 = y

    # 透视投影回当前图像平面 Z=f
    valid = z2 > 1e-6
    map_x = np.full((h, w), -1.0, dtype=np.float32)
    map_y = np.full((h, w), -1.0, dtype=np.float32)
    map_x[valid] = (f * x2[valid] / z2[valid] + cx).astype(np.float32)
    map_y[valid] = (f * y2[valid] / z2[valid] + cy).astype(np.float32)
    return map_x, map_y


def warp_view(image, yaw_deg, f, mode):
    h, w = image.shape[:2]
    map_x, map_y = project_map(h, w, np.deg2rad(yaw_deg), f, mode)
    return cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(245, 245, 245))


# ---------------------------------------------------------------------------
# 合成场景：一圈「环拍」用的水平街景条带
# ---------------------------------------------------------------------------


def make_pano_strip(width=1200, height=280):
    """横向超宽街景，用来裁出多张「转头拍」的照片。"""
    img = np.full((height, width, 3), 245, dtype=np.uint8)
    img[: height // 2] = (236, 228, 214)
    img[height // 2 :] = (205, 192, 172)

    # 远山起伏
    hills = []
    for i in range(0, width + 80, 80):
        hills.append([i, 90 + int(35 * np.sin(i * 0.02))])
    hills = np.array([[0, height // 2]] + hills + [[width, height // 2]], np.int32)
    cv2.fillPoly(img, [hills], (175, 165, 148))

    # 彩色楼房：不同色块，方便看出拉伸/弯曲
    rng = np.random.default_rng(9)
    x = 20
    while x < width - 40:
        w = int(rng.integers(36, 70))
        h = int(rng.integers(70, 140))
        color = (
            int(rng.integers(60, 140)),
            int(rng.integers(70, 150)),
            int(rng.integers(80, 160)),
        )
        top = height // 2 - h
        cv2.rectangle(img, (x, top), (x + w, height // 2), color, -1)
        cv2.rectangle(img, (x, top), (x + w, height // 2), (40, 40, 40), 1)
        # 窗户网格
        for wy in range(top + 10, height // 2 - 8, 18):
            for wx in range(x + 8, x + w - 8, 14):
                cv2.rectangle(img, (wx, wy), (wx + 6, wy + 8), (220, 230, 235), -1)
        x += w + int(rng.integers(18, 40))

    # 路面标线
    y = height // 2 + 36
    for x0 in range(10, width - 10, 44):
        cv2.line(img, (x0, y), (x0 + 24, y), (250, 250, 250), 3)

    # 等距竖线：投影后看直线弯不弯
    for gx in range(0, width, 60):
        cv2.line(img, (gx, 8), (gx, height - 8), (150, 140, 125), 1)

    # 编号地标，方便对照
    for i, lab in enumerate("ABCDEFGHIJKL"):
        px = 40 + i * 95
        if px >= width - 20:
            break
        cv2.circle(img, (px, height // 2 - 20), 14, WAVE_BGR if i % 2 == 0 else TEAL_BGR, -1)
        cv2.putText(img, lab, (px - 6, height // 2 - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    return img


def crop_views(strip, centers_x, view_w=320, view_h=240):
    """从超宽条带按中心裁出若干「照片」。"""
    H, W = strip.shape[:2]
    views = []
    for cx in centers_x:
        x0 = int(cx - view_w // 2)
        y0 = (H - view_h) // 2
        x0 = max(0, min(W - view_w, x0))
        y0 = max(0, min(H - view_h, y0))
        views.append(strip[y0 : y0 + view_h, x0 : x0 + view_w].copy())
    return views


def make_camera_photo(view, label, f_draw=None):
    """给照片加边框与标签。"""
    out = view.copy()
    h, w = out.shape[:2]
    cv2.rectangle(out, (2, 2), (w - 3, h - 3), INK_BGR, 2)
    cv2.putText(out, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, WAVE_BGR, 2, cv2.LINE_AA)
    if f_draw is not None:
        cv2.putText(out, f"f~{int(f_draw)}", (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, MUTED_BGR, 1, cv2.LINE_AA)
    return out


# ---------------------------------------------------------------------------
# 课文配图
# ---------------------------------------------------------------------------


def plot_planar_blowup():
    """广角环拍时，平面单应为何「撑破」。"""
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.8), layout="constrained")

    # 左：小转角，平面还行
    ax = axes[0]
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-0.2, 1.35)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("转角小：平面差不多够用")
    # 相机
    ax.plot([0], [0], "o", color=INK, ms=6)
    ax.text(0.05, -0.12, "相机", fontsize=10, color=MUTED)
    # 小扇形视野
    wedge_xs = [0, np.sin(np.deg2rad(-25)), np.sin(np.deg2rad(25)), 0]
    wedge_ys = [0, np.cos(np.deg2rad(-25)), np.cos(np.deg2rad(25)), 0]
    ax.fill(wedge_xs, wedge_ys, color="#e5f3f1", alpha=0.8)
    ax.plot([-0.55, 0.55], [1.0, 1.0], color=TEAL, lw=2.5)
    ax.text(0, 1.12, "投影平面", ha="center", color=TEAL, fontsize=11)
    ax.text(0, 0.45, "画面\n几乎均匀", ha="center", fontsize=11, color=MUTED)

    # 中：大转角，平面边缘飞出去
    ax = axes[1]
    ax.set_xlim(-2.2, 2.2)
    ax.set_ylim(-0.2, 1.5)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("转角大：边缘被撑到无穷远")
    ax.plot([0], [0], "o", color=INK, ms=6)
    # 大扇形
    angs = np.linspace(-80, 80, 40)
    ax.fill(
        np.concatenate([[0], np.sin(np.deg2rad(angs)), [0]]),
        np.concatenate([[0], np.cos(np.deg2rad(angs)), [0]]),
        color="#fde8df",
        alpha=0.85,
    )
    # 平面上的采样点：θ 越大，平面坐标 tanθ 爆炸
    thetas = np.deg2rad([-70, -45, -20, 0, 20, 45, 70])
    xs = np.tan(thetas)
    ys = np.ones_like(xs)
    ax.plot([-1.9, 1.9], [1.0, 1.0], color=WAVE, lw=2.0)
    ax.scatter(xs, ys, c=[WAVE if abs(t) > 0.5 else TEAL for t in thetas], s=28, zorder=3)
    for t, x in zip(thetas, xs):
        ax.plot([0, x], [0, 1], color=MUTED, lw=0.7, alpha=0.7)
    ax.text(0, 1.22, "同一平面上：两侧间距被拉爆", ha="center", color=WAVE, fontsize=11)
    ax.annotate("", xy=(1.7, 1.0), xytext=(0.6, 1.0), arrowprops=dict(arrowstyle="->", color=WAVE, lw=1.6))
    ax.text(1.15, 1.08, "越远越稀", color=WAVE, fontsize=10)

    # 右：拼出来的「爆炸」示意
    ax = axes[2]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")
    ax.set_title("拼到平面画布上的观感")
    # 中间正常、两侧拉伸的格子
    for i, (x0, scale) in enumerate([(1.2, 0.55), (3.5, 1.0), (6.6, 0.55)]):
        w = 1.6 / scale
        h = 3.2 * scale
        y0 = 3 - h / 2
        color = "#fde8df" if scale < 0.9 else "#e5f3f1"
        ax.add_patch(FancyBboxPatch((x0, y0), w, h, boxstyle="round,pad=0.02", facecolor=color, edgecolor=INK, lw=1.2))
        # 格子
        for gx in np.linspace(x0, x0 + w, 4):
            ax.plot([gx, gx], [y0, y0 + h], color=MUTED, lw=0.5, alpha=0.6)
        for gy in np.linspace(y0, y0 + h, 5):
            ax.plot([x0, x0 + w], [gy, gy], color=MUTED, lw=0.5, alpha=0.6)
        lab = ["侧图\n拉扁", "中间\n正常", "侧图\n拉扁"][i]
        ax.text(x0 + w / 2, 0.6, lab, ha="center", fontsize=10, color=WAVE if scale < 0.9 else TEAL)
    ax.text(5, 5.4, "平面单应硬拼广角环拍 → 边缘畸变失控", ha="center", fontsize=11, color=INK)

    save_fig(fig, "ch09_planar_blowup.png")


def plot_cylinder_idea():
    fig, ax = plt.subplots(figsize=(9.6, 4.2), layout="constrained")
    ax.set_xlim(-1.6, 8.2)
    ax.set_ylim(-1.4, 1.6)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("柱面投影直觉：光线先打到圆柱，再把圆柱「剪开铺平」")

    # 左：俯视圆柱
    circ = Circle((0, 0), 1.0, fill=False, edgecolor=TEAL, lw=2.2)
    ax.add_patch(circ)
    ax.plot([0], [0], "o", color=INK, ms=7)
    ax.text(0.08, -0.25, "相机", fontsize=10)
    # 若干射线与落点
    for deg in (-50, -25, 0, 25, 50):
        th = np.deg2rad(deg)
        ax.plot([0, np.sin(th)], [0, np.cos(th)], color=MUTED, lw=0.9)
        ax.plot(np.sin(th), np.cos(th), "o", color=WAVE, ms=5)
    ax.text(-0.2, 1.2, "圆柱壁", color=TEAL, fontsize=11)
    ax.text(-1.5, -1.2, "俯视：水平角 θ 均匀占圆柱周长", fontsize=11, color=MUTED)

    # 箭头
    ax.annotate("", xy=(2.4, 0.1), xytext=(1.3, 0.1), arrowprops=dict(arrowstyle="->", color=INK, lw=1.5))
    ax.text(1.7, 0.35, "展开", fontsize=11)

    # 右：展开成矩形
    ax.add_patch(Rectangle((2.8, -1.0), 4.8, 2.0, fill=True, facecolor="#e5f3f1", edgecolor=TEAL, lw=2.0))
    for i, deg in enumerate((-50, -25, 0, 25, 50)):
        u = 5.2 + deg / 50 * 2.0
        ax.plot([u, u], [-1.0, 1.0], color=WAVE, lw=1.0, alpha=0.8)
        ax.plot(u, 0.0, "o", color=WAVE, ms=5)
        ax.text(u, 1.15, f"{deg}°", ha="center", fontsize=9, color=MUTED)
    ax.text(5.2, -1.25, "铺平后：横坐标 ∝ θ，竖线间距更均匀", ha="center", fontsize=11, color=INK)
    ax.text(5.2, 1.4, "柱面展开图（全景条带）", ha="center", fontsize=12, color=TEAL)

    save_fig(fig, "ch09_cylinder_idea.png")


def plot_sphere_idea():
    fig, ax = plt.subplots(figsize=(9.6, 4.4), layout="constrained")
    ax.set_xlim(-2.2, 8.5)
    ax.set_ylim(-2.0, 2.2)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("球面投影直觉：光线打到球面，再按经纬度摊成矩形")

    # 左：球示意（椭圆+经纬）
    circ = Circle((0, 0), 1.15, fill=False, edgecolor=TEAL, lw=2.0)
    ax.add_patch(circ)
    ax.add_patch(Arc((0, 0), 2.3, 0.7, theta1=0, theta2=360, edgecolor=MUTED, lw=1.0, ls="--"))
    ax.add_patch(Arc((0, 0), 0.7, 2.3, theta1=0, theta2=360, edgecolor=MUTED, lw=1.0, ls="--"))
    ax.plot([0], [0], "o", color=INK, ms=6)
    ax.text(0.1, -0.35, "相机", fontsize=10)
    # 若干落点
    for lon, lat in [(-40, 20), (0, 35), (35, -15), (50, 25), (-55, -25)]:
        th = np.deg2rad(lon)
        ph = np.deg2rad(lat)
        x = np.cos(ph) * np.sin(th)
        y = np.sin(ph)
        # 透视压扁一点显示
        ax.plot(1.15 * x, 1.15 * y, "o", color=WAVE, ms=5)
        ax.plot([0, 1.15 * x], [0, 1.15 * y], color=MUTED, lw=0.7, alpha=0.7)
    ax.text(-2.1, -1.7, "上下也要大转角时，球面更合适", fontsize=11, color=MUTED)

    ax.annotate("", xy=(2.6, 0), xytext=(1.5, 0), arrowprops=dict(arrowstyle="->", color=INK, lw=1.5))
    ax.text(1.9, 0.25, "展开", fontsize=11)

    # 右：经纬矩形
    ax.add_patch(Rectangle((3.0, -1.5), 5.0, 3.0, fill=True, facecolor="#fde8df", edgecolor=WAVE, lw=2.0))
    for i in range(6):
        ax.plot([3.0 + i, 3.0 + i], [-1.5, 1.5], color=MUTED, lw=0.5, alpha=0.5)
    for j in range(5):
        ax.plot([3.0, 8.0], [-1.5 + 0.75 * j, -1.5 + 0.75 * j], color=MUTED, lw=0.5, alpha=0.5)
    ax.text(5.5, 1.7, "经度 θ → 横轴", ha="center", color=WAVE, fontsize=11)
    ax.text(8.15, 0, "纬度 φ\n→ 纵轴", va="center", fontsize=11, color=WAVE)
    ax.text(5.5, -1.85, "可覆盖接近整球视野（含抬头低头）", ha="center", fontsize=11, color=INK)

    save_fig(fig, "ch09_sphere_idea.png")


def plot_formula_board():
    """公式白话板。"""
    fig, ax = plt.subplots(figsize=(10.0, 5.2), layout="constrained")
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_title("三种投影：先统一成「射线」，再落到不同画布")

    boxes = [
        (0.3, 7.2, 9.4, 2.2, "#f7f1e8", "共用第一步：像素 (x,y) → 射线", "X=(x-cx)/f，  Y=(y-cy)/f，  Z=1\n（f 是焦距像素；cx,cy 是光心/主点）"),
        (0.3, 4.6, 3.0, 2.2, "#e5f3f1", "平面", "x' = f*X/Z + cx\ny' = f*Y/Z + cy\n等价于打到 Z=f 平面"),
        (3.5, 4.6, 3.0, 2.2, "#fde8df", "柱面", "theta = atan2(X, Z)\nh = Y / sqrt(X^2+Z^2)\nx'=f*theta+cx， y'=f*h+cy"),
        (6.7, 4.6, 3.0, 2.2, "#e8eef7", "球面", "theta = atan2(X, Z)\nphi = arcsin(Y/|r|)\nx'=f*theta+cx， y'=f*phi+cy"),
        (0.3, 0.6, 9.4, 3.4, "#fffdf8", "白话对照", "平面：像把照片贴在一块玻璃板上——左右转太狠，板边不够用，只能把图「扯」过去。\n柱面：像把照片贴在灯笼纸筒上再剪开——水平 360° 很自然，上下仍受限。\n球面：像贴在地球仪上再按经纬摊开——抬头低头也能收进同一张展开图。"),
    ]
    for x, y, w, h, fc, title, body in boxes:
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05", facecolor=fc, edgecolor=INK, lw=1.1))
        ax.text(x + 0.15, y + h - 0.45, title, fontsize=13, fontweight="bold", color=INK)
        ax.text(x + 0.15, y + 0.25, body, fontsize=11, color=INK, va="bottom", linespacing=1.45)

    save_fig(fig, "ch09_formulas.png")


def plot_selection_table():
    fig, ax = plt.subplots(figsize=(10.2, 3.8), layout="constrained")
    ax.axis("off")
    ax.set_title("选型速查：场景 → 投影")

    col_labels = ["场景", "推荐投影", "为什么", "别踩的坑"]
    cell_text = [
        ["墙/海报/桌面\n（近似一个平面）", "平面 + 单应 H", "本来就是平面到平面", "近景大平移仍有视差"],
        ["原地水平环拍\n（抬头不多）", "柱面", "水平角均匀展开，\n两侧不爆炸", "把柱面当平面再估 H\n仍会有弯曲残差"],
        ["含大幅俯仰\n或接近整球", "球面", "经纬度覆盖上下", "两极附近拉伸大；\n存储/交互更重"],
        ["手机「扫一圈」\n日常全景", "多数用柱面", "实现简单、观感够用", "手平移过大 → 视差鬼影"],
    ]
    table = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        loc="center",
        cellLoc="left",
        colColours=["#e5f3f1"] * 4,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 2.35)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#d7ccba")
        if r == 0:
            cell.set_text_props(weight="bold", color=INK)
        cell.set_facecolor("#fffdf8" if r % 2 == 0 else "#f7f1e8")
        if r == 0:
            cell.set_facecolor("#e5f3f1")
    save_fig(fig, "ch09_selection.png")


def stitch_side_by_side(views, labels, gap=8):
    """把多张等高图横拼，中间留缝。"""
    h = max(v.shape[0] for v in views)
    chunks = []
    for i, (v, lab) in enumerate(zip(views, labels)):
        if v.shape[0] != h:
            v = cv2.resize(v, (int(v.shape[1] * h / v.shape[0]), h))
        tagged = make_camera_photo(v, lab)
        chunks.append(tagged)
        if i < len(views) - 1:
            chunks.append(np.full((h, gap, 3), 230, dtype=np.uint8))
    return np.hstack(chunks)


def compose_canvas(warped_list, f, mode_title):
    """把多张已投影图按水平偏移叠到一张宽画布（示意，非完整缝合）。"""
    h, w = warped_list[0].shape[:2]
    # 相邻图在柱面/球面上的位移约 f * Δθ；平面上用更大偏移模拟「撑开」
    n = len(warped_list)
    if "平面" in mode_title:
        # 人为加大间距 + 只取中部，突出边缘崩溃感
        shifts = [int(0.55 * w * i) for i in range(n)]
    else:
        shifts = [int(0.42 * w * i) for i in range(n)]
    canvas_w = shifts[-1] + w + 20
    canvas = np.full((h + 20, canvas_w, 3), 248, dtype=np.uint8)
    for img, sx in zip(warped_list, shifts):
        roi = canvas[10 : 10 + h, 10 + sx : 10 + sx + w]
        mask = np.any(img < 240, axis=2)
        # 简单平均融合重叠区
        both = mask & np.any(roi < 240, axis=2)
        only = mask & ~both
        roi[only] = img[only]
        roi[both] = ((roi[both].astype(np.int16) + img[both].astype(np.int16)) // 2).astype(np.uint8)
    cv2.putText(canvas, mode_title, (12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, WAVE_BGR, 2, cv2.LINE_AA)
    return canvas


def make_grid_photo(w=320, h=240, yaw_hint=0):
    """带网格与建筑物的单帧，用于投影对照。"""
    img = np.full((h, w, 3), 245, dtype=np.uint8)
    img[: h // 2] = (236, 228, 214)
    img[h // 2 :] = (208, 196, 176)
    # 竖线网格
    for x in range(0, w, 20):
        cv2.line(img, (x, 0), (x, h), (170, 160, 145), 1)
    for y in range(0, h, 20):
        cv2.line(img, (0, y), (w, y), (170, 160, 145), 1)
    # 中央十字
    cv2.line(img, (w // 2, 10), (w // 2, h - 10), TEAL_BGR, 2)
    cv2.line(img, (10, h // 2), (w - 10, h // 2), TEAL_BGR, 2)
    # 几栋楼
    for x0, ww, hh, col in (
        (40, 36, 90, (90, 110, 150)),
        (130, 44, 120, (70, 95, 130)),
        (230, 40, 100, (100, 90, 70)),
    ):
        top = h // 2 - hh
        cv2.rectangle(img, (x0, top), (x0 + ww, h // 2), col, -1)
        cv2.rectangle(img, (x0, top), (x0 + ww, h // 2), (40, 40, 40), 1)
    cv2.putText(img, f"yaw~{yaw_hint}", (8, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, MUTED_BGR, 1, cv2.LINE_AA)
    return img


def build_source_views():
    strip = make_pano_strip()
    # 五个重叠取景，模拟水平环拍
    centers = [180, 340, 500, 660, 820]
    views = crop_views(strip, centers, view_w=300, view_h=220)
    return strip, views


def figure_source_views():
    strip, views = build_source_views()
    save_bgr("ch09_pano_strip.png", strip)
    row = stitch_side_by_side(views, [f"V{i}" for i in range(len(views))])
    save_bgr("ch09_source_views.png", row)

    fig, ax = plt.subplots(figsize=(11.0, 2.8), layout="constrained")
    ax.imshow(cv2.cvtColor(row, cv2.COLOR_BGR2RGB))
    ax.axis("off")
    ax.set_title("同一组环拍照片 V0…V4（后面三种投影都用它们）")
    save_fig(fig, "ch09_source_views_fig.png")
    return views


def figure_three_way_compare(views, f=240.0, yaws=None, tag="ch09_compare_three"):
    """对同一组图，分别映到平面 / 柱面 / 球面画布并横拼示意。"""
    if yaws is None:
        # 相对中心帧的水平角（度）
        yaws = [-48, -24, 0, 24, 48]

    modes = [
        ("planar", "平面投影：两侧拉伸/撑破"),
        ("cylindrical", "柱面投影：水平更均匀"),
        ("spherical", "球面投影：上下也按纬度收"),
    ]
    canvases = []
    for mode, title in modes:
        warped = []
        for view, yaw in zip(views, yaws):
            warped.append(warp_view(view, yaw, f, mode))
        canvases.append(compose_canvas(warped, f, title))

    # 统一高度后竖排
    max_w = max(c.shape[1] for c in canvases)
    padded = []
    for c in canvases:
        if c.shape[1] < max_w:
            pad = np.full((c.shape[0], max_w - c.shape[1], 3), 248, dtype=np.uint8)
            c = np.hstack([c, pad])
        padded.append(c)
    stack = np.vstack(padded)
    save_bgr(f"{tag}.png", stack)

    fig, axes = plt.subplots(3, 1, figsize=(11.0, 7.2), layout="constrained")
    for ax, c, (_, title) in zip(axes, padded, modes):
        ax.imshow(cv2.cvtColor(c, cv2.COLOR_BGR2RGB))
        ax.set_title(title)
        ax.axis("off")
    save_fig(fig, f"{tag}_fig.png")
    return stack


def figure_single_warp_demo():
    """单张网格图在三种投影下的形状变化（直观）。"""
    photo = make_grid_photo()
    f = 180.0
    # 用「把正前方图直接按投影公式重采样」——这里用 identity yaw，但用广角 f 小来强调
    # 更清楚：显示「从透视图 → 柱/球展开」的 remap
    h, w = photo.shape[:2]
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0

    def forward_remap(mode):
        map_x = np.zeros((h, w), np.float32)
        map_y = np.zeros((h, w), np.float32)
        for y in range(h):
            for x in range(w):
                # 输出是投影画布；求原透视像素
                if mode == "planar":
                    # 恒等
                    map_x[y, x] = x
                    map_y[y, x] = y
                else:
                    # 画布 (x,y) → 射线 → 透视像素
                    if mode == "cylindrical":
                        theta = (x - cx) / f
                        hh = (y - cy) / f
                        ray = np.array([np.sin(theta), hh, np.cos(theta)])
                    else:
                        theta = (x - cx) / f
                        phi = (y - cy) / f
                        cp = np.cos(phi)
                        ray = np.array([cp * np.sin(theta), np.sin(phi), cp * np.cos(theta)])
                    if ray[2] <= 1e-6:
                        map_x[y, x] = -1
                        map_y[y, x] = -1
                    else:
                        map_x[y, x] = f * ray[0] / ray[2] + cx
                        map_y[y, x] = f * ray[1] / ray[2] + cy
        return cv2.remap(photo, map_x, map_y, cv2.INTER_LINEAR, borderValue=(245, 245, 245))

    planar = forward_remap("planar")
    cyl = forward_remap("cylindrical")
    sph = forward_remap("spherical")

    row = stitch_side_by_side([planar, cyl, sph], ["平面(原图)", "柱面展开", "球面展开"])
    save_bgr("ch09_single_warp.png", row)

    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.4), layout="constrained")
    for ax, im, t in zip(axes, [planar, cyl, sph], ["平面＝原透视", "柱面：边缘水平收拢", "球面：四角进一步收"]):
        ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        ax.set_title(t)
        ax.axis("off")
    save_fig(fig, "ch09_single_warp_fig.png")


def figure_my_params(views):
    """用户可调参数输出。"""
    f = float(MY_FOCAL)
    yaw = float(MY_YAW_DEG)
    yaws = [-2 * yaw, -yaw, 0.0, yaw, 2 * yaw]

    center = views[len(views) // 2]
    for mode, name in (("cylindrical", "ch09_my_cylinder.png"), ("spherical", "ch09_my_sphere.png")):
        out = warp_view(center, 0.0, f, mode)
        # 再叠一张偏转后的，看对齐感
        out2 = warp_view(views[0], -yaw, f, mode)
        blend = out.copy()
        m = np.any(out2 < 240, axis=2)
        blend[m] = ((blend[m].astype(np.int16) + out2[m].astype(np.int16)) // 2).astype(np.uint8)
        cv2.putText(blend, f"f={f:.0f} yaw={yaw:.0f}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, WAVE_BGR, 2, cv2.LINE_AA)
        save_bgr(name, blend)

    figure_three_way_compare(views, f=f, yaws=yaws, tag="ch09_my_compare")


# ---------------------------------------------------------------------------
# 自检
# ---------------------------------------------------------------------------


def self_check():
    f = 200.0
    cx = cy = 0.0

    # 正前方点应映到原点附近
    ray = pixel_to_ray(0, 0, cx, cy, f)
    assert ray_to_planar(ray, cx, cy, f) == (0.0, 0.0)
    assert np.allclose(ray_to_cylindrical(ray, cx, cy, f), (0.0, 0.0))
    assert np.allclose(ray_to_spherical(ray, cx, cy, f), (0.0, 0.0))

    # 水平 45°：柱面 x′ = f * π/4
    ray45 = np.array([1.0, 0.0, 1.0])  # 45°
    u, v = ray_to_cylindrical(ray45, 0, 0, f)
    assert abs(u - f * np.pi / 4) < 1e-6 and abs(v) < 1e-6

    # 平面 45°：x′ = f * tan(45°) = f
    up, vp = ray_to_planar(ray45, 0, 0, f)
    assert abs(up - f) < 1e-6 and abs(vp) < 1e-6

    # 同水平角下，平面比柱面「更靠外」（|tan θ| > |θ|）
    assert abs(up) > abs(u)

    # 大角度：平面趋向无穷，柱面仍有限
    ray80 = rotate_yaw(np.array([0.0, 0.0, 1.0]), np.deg2rad(80))
    up80, _ = ray_to_planar(ray80, 0, 0, f)
    uc80, _ = ray_to_cylindrical(ray80, 0, 0, f)
    assert abs(up80) > abs(uc80) * 2

    # 旋转往返
    r = pixel_to_ray(30, -20, 0, 0, f)
    back = rotate_yaw(rotate_yaw(r, 0.4), -0.4)
    assert np.allclose(r, back, atol=1e-9)

    print("自检通过")


def main():
    use_style()
    FIG.mkdir(parents=True, exist_ok=True)
    IMG.mkdir(parents=True, exist_ok=True)

    self_check()

    plot_planar_blowup()
    plot_cylinder_idea()
    plot_sphere_idea()
    plot_formula_board()
    plot_selection_table()
    figure_single_warp_demo()

    views = figure_source_views()
    figure_three_way_compare(views, f=240.0, yaws=[-48, -24, 0, 24, 48], tag="ch09_compare_three")
    figure_my_params(views)

    print("第 9 篇全部配图已生成。")


if __name__ == "__main__":
    main()
