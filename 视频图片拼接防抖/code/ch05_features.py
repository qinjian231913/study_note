"""第 5 篇：角点与特征点。

在「视频图片拼接防抖」文件夹里运行：

    MPLCONFIGDIR=/tmp/mpl-cache .venv/bin/python code/ch05_features.py

图会保存到 assets/figures/。
顶部「改这里」只影响 ch05_my_features.png。
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

from plot_style import INK, MUTED, TEAL, WAVE, use_style

WAVE_BGR = (40, 77, 194)
TEAL_BGR = (102, 107, 14)
INK_BGR = (21, 25, 28)

# ===== 改这里 =====
# 只影响 ch05_my_features.png
MY_MAX_CORNERS = 80          # goodFeaturesToTrack 最多取多少点
MY_QUALITY = 0.01            # 质量阈值，越小点越多（0.001～0.1）
MY_MIN_DISTANCE = 12         # 两点之间最少像素距离
MY_BLUR = 0                  # 检测前高斯模糊核（奇数，0 表示不模糊）
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


def make_shape_board(w=420, h=280):
    """几何形状合成图：平坦、边缘、角点都有。"""
    img = np.full((h, w, 3), 248, dtype=np.uint8)
    # 大块平坦灰
    cv2.rectangle(img, (20, 20), (150, 120), (210, 205, 198), -1)
    # 竖直边缘（黑白分界）
    cv2.rectangle(img, (180, 30), (250, 130), (40, 40, 40), -1)
    cv2.rectangle(img, (250, 30), (320, 130), (235, 230, 220), -1)
    # L 形角点
    cv2.rectangle(img, (40, 150), (140, 250), (70, 110, 150), -1)
    cv2.rectangle(img, (140, 200), (220, 250), (70, 110, 150), -1)
    # 棋盘纹理（很多角）
    cell = 18
    ox, oy = 250, 160
    for j in range(5):
        for i in range(7):
            if (i + j) % 2 == 0:
                x0, y0 = ox + i * cell, oy + j * cell
                cv2.rectangle(img, (x0, y0), (x0 + cell - 1, y0 + cell - 1), (50, 48, 45), -1)
    # 圆形（边缘有曲率，但不如尖角「两个方向都强」）
    cv2.circle(img, (360, 70), 36, (90, 140, 120), -1)
    return img


def make_realistic_scene(w=560, h=320):
    """真实感合成场景：楼房、窗、路面、树、纹理墙。"""
    img = np.full((h, w, 3), 236, dtype=np.uint8)
    # 天空 / 地面
    img[: h // 2] = (228, 214, 196)
    img[h // 2 :] = (175, 168, 158)

    # 远山
    hills = np.array(
        [[0, 155], [90, 110], [180, 140], [300, 95], [420, 130], [560, 105], [560, 160], [0, 160]],
        np.int32,
    )
    cv2.fillPoly(img, [hills], (168, 158, 140))

    # 楼房
    buildings = (
        (40, 70, 90, 160, (95, 105, 125)),
        (150, 95, 70, 135, (80, 90, 70)),
        (260, 55, 110, 175, (110, 95, 85)),
        (400, 85, 85, 145, (70, 100, 115)),
        (490, 75, 55, 155, (100, 85, 95)),
    )
    for x, top, bw, bh, color in buildings:
        base = h // 2
        cv2.rectangle(img, (x, top), (x + bw, base), color, -1)
        cv2.rectangle(img, (x, top), (x + bw, base), (35, 35, 35), 1)
        # 窗户网格 → 大量角点
        for wy in range(top + 12, base - 10, 18):
            for wx in range(x + 8, x + bw - 8, 16):
                cv2.rectangle(img, (wx, wy), (wx + 8, wy + 10), (220, 210, 180), -1)
                cv2.rectangle(img, (wx, wy), (wx + 8, wy + 10), (40, 40, 40), 1)

    # 纹理墙（砖纹）
    wall_x, wall_y = 30, h // 2 + 10
    for row in range(4):
        for col in range(8):
            bx = wall_x + col * 22 + (11 if row % 2 else 0)
            by = wall_y + row * 14
            shade = 140 + ((col + row) % 3) * 18
            cv2.rectangle(img, (bx, by), (bx + 20, by + 12), (shade, shade - 10, shade - 20), -1)
            cv2.rectangle(img, (bx, by), (bx + 20, by + 12), (60, 55, 50), 1)

    # 路面虚线
    y = h // 2 + 55
    for x0 in range(200, w - 30, 48):
        cv2.line(img, (x0, y), (x0 + 26, y), (245, 245, 245), 3)

    # 树
    cv2.rectangle(img, (330, 200), (342, 280), (55, 75, 40), -1)
    cv2.circle(img, (336, 185), 32, (45, 115, 65), -1)

    # 路牌矩形（清晰角点）
    cv2.rectangle(img, (470, 200), (530, 245), (40, 90, 180), -1)
    cv2.putText(img, "P", (485, 233), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (240, 240, 240), 2, cv2.LINE_AA)

    return img


def draw_points_bgr(img, pts, color=WAVE_BGR, radius=4):
    vis = img.copy()
    if pts is None:
        return vis
    for p in np.asarray(pts).reshape(-1, 2):
        cv2.circle(vis, (int(round(p[0])), int(round(p[1]))), radius, color, -1, lineType=cv2.LINE_AA)
        cv2.circle(vis, (int(round(p[0])), int(round(p[1]))), radius + 1, INK_BGR, 1, lineType=cv2.LINE_AA)
    return vis


def detect_gft(gray, max_corners=100, quality=0.01, min_dist=10):
    pts = cv2.goodFeaturesToTrack(
        gray,
        maxCorners=max_corners,
        qualityLevel=quality,
        minDistance=min_dist,
        blockSize=5,
        useHarrisDetector=False,
    )
    return pts


def detect_harris_points(gray, threshold_ratio=0.01, max_points=120):
    """Harris 响应图取局部极大值，返回 (N,1,2) float32。"""
    gray_f = np.float32(gray)
    resp = cv2.cornerHarris(gray_f, blockSize=2, ksize=3, k=0.04)
    resp = cv2.dilate(resp, None)
    thresh = threshold_ratio * float(resp.max() + 1e-12)
    ys, xs = np.where(resp > thresh)
    if len(xs) == 0:
        return None
    scores = resp[ys, xs]
    order = np.argsort(-scores)
    xs, ys = xs[order], ys[order]
    # 非极大抑制：按分数贪心，保持间距
    kept = []
    for x, y in zip(xs, ys):
        if all((x - kx) ** 2 + (y - ky) ** 2 >= 64 for kx, ky in kept):
            kept.append((float(x), float(y)))
        if len(kept) >= max_points:
            break
    if not kept:
        return None
    return np.array(kept, dtype=np.float32).reshape(-1, 1, 2)


def detect_fast(gray, threshold=25, nonmax=True):
    detector = cv2.FastFeatureDetector_create(threshold=threshold, nonmaxSuppression=nonmax)
    kps = detector.detect(gray, None)
    if not kps:
        return None
    pts = np.array([[kp.pt[0], kp.pt[1]] for kp in kps], dtype=np.float32).reshape(-1, 1, 2)
    return pts


def plot_sliding_window_intuition():
    """平坦 / 边缘 / 角点：小窗平移时灰度变化直觉。"""
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.8), layout="constrained")

    # 构造三块局部图案
    patches = []
    flat = np.full((80, 80), 180, dtype=np.uint8)
    patches.append(("平坦区：怎么滑都差不多", flat, "变化≈0 → 难认"))

    edge = np.full((80, 80), 200, dtype=np.uint8)
    edge[:, 40:] = 40
    patches.append(("边缘：垂直滑变化大，平行滑小", edge, "只在一个方向变化"))

    corner = np.full((80, 80), 200, dtype=np.uint8)
    corner[40:, :] = 40
    corner[:, 40:] = 40
    corner[40:, 40:] = 40
    patches.append(("角点：两个方向都变化大", corner, "最容易当「路标」"))

    for ax, (title, patch, note) in zip(axes, patches):
        ax.imshow(patch, cmap="gray", vmin=0, vmax=255)
        # 画一个橙窗 + 平移后的青窗
        ax.add_patch(Rectangle((28, 28), 22, 22, fill=False, edgecolor=WAVE, lw=2))
        ax.add_patch(Rectangle((36, 34), 22, 22, fill=False, edgecolor=TEAL, lw=1.8, linestyle="--"))
        ax.annotate(
            "",
            xy=(47, 45),
            xytext=(39, 39),
            arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.4),
        )
        ax.set_title(title, fontsize=11)
        ax.set_xlabel(note, fontsize=10, color=MUTED)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle("滑动窗口：看小块灰度差（SSD）够不够大", fontsize=13, color=INK)
    save_fig(fig, "ch05_flat_vs_corner.png")


def plot_harris_idea():
    """Harris：两个主方向都有大变化 → 角点。"""
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.7), layout="constrained")

    cases = [
        ("平坦", "两个方向变化都小\n→ R 很小，不是特征", False, False),
        ("边缘", "只有一个方向变化大\n→ 像「一条线」，不是角", True, False),
        ("角点", "两个方向都变化大\n→ R 很大，判为角点", True, True),
    ]
    for ax, (title, note, strong_x, strong_y) in zip(axes, cases):
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(title)

        # 中心点
        ax.scatter([5], [5], s=80, color=WAVE, zorder=5)
        # 方向箭头：粗细表示变化强度
        lx = 3.2 if strong_x else 1.2
        ly = 3.2 if strong_y else 1.2
        ax.annotate(
            "",
            xy=(5 + lx, 5),
            xytext=(5 - lx, 5),
            arrowprops=dict(arrowstyle="<->", color=TEAL if strong_x else MUTED, lw=2.4 if strong_x else 1.0),
        )
        ax.annotate(
            "",
            xy=(5, 5 + ly),
            xytext=(5, 5 - ly),
            arrowprops=dict(arrowstyle="<->", color=TEAL if strong_y else MUTED, lw=2.4 if strong_y else 1.0),
        )
        ax.text(5, 0.6, note, ha="center", va="bottom", fontsize=10, color=MUTED)

        # 示意小窗
        ax.add_patch(Rectangle((3.2, 3.2), 3.6, 3.6, fill=False, edgecolor=INK, lw=1.2, linestyle=":"))

    fig.suptitle("Harris 白话：窗口里「横竖都变」才算角", fontsize=13)
    save_fig(fig, "ch05_harris_idea.png")


def plot_fast_idea():
    """FAST：圆周上连续亮/暗像素。"""
    fig, ax = plt.subplots(figsize=(7.2, 5.0), layout="constrained")
    ax.set_xlim(-3.2, 3.2)
    ax.set_ylim(-3.4, 3.2)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("FAST 直觉：圆心像素 vs 周围一圈")

    # 圆心
    ax.add_patch(Circle((0, 0), 0.28, color=WAVE, zorder=5))
    ax.text(0, -0.55, "候选点 p", ha="center", fontsize=11, color=WAVE)

    # 16 个圆周点（Bresenham 圆示意）
    n = 16
    r = 2.0
    bright = {0, 1, 2, 3, 4, 5, 6, 7, 8}  # 连续一段更亮
    for i in range(n):
        ang = -np.pi / 2 + i * 2 * np.pi / n
        x, y = r * np.cos(ang), r * np.sin(ang)
        col = TEAL if i in bright else MUTED
        ax.add_patch(Circle((x, y), 0.22, color=col, zorder=4))
        ax.text(x * 1.28, y * 1.28, str(i + 1), ha="center", va="center", fontsize=9, color=INK)

    ax.add_patch(Circle((0, 0), r, fill=False, edgecolor="#c9bda8", lw=1.2, linestyle="--", zorder=1))
    ax.text(
        0,
        -3.1,
        "若圆周上有一段连续 ≥9 个像素都比 p「明显更亮」或「明显更暗」\n→ 认为 p 是角点。只比灰度阈值，所以很快。",
        ha="center",
        fontsize=11,
        color=MUTED,
    )
    save_fig(fig, "ch05_fast_idea.png")


def plot_detections_on_shapes():
    img = make_shape_board()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gft = detect_gft(gray, max_corners=60, quality=0.01, min_dist=8)
    harris = detect_harris_points(gray, threshold_ratio=0.02, max_points=60)
    fast = detect_fast(gray, threshold=20)

    panels = [
        ("goodFeaturesToTrack\n（Shi-Tomasi）", draw_points_bgr(img, gft, WAVE_BGR)),
        ("cornerHarris\n（响应阈值）", draw_points_bgr(img, harris, TEAL_BGR)),
        ("FastFeatureDetector", draw_points_bgr(img, fast, WAVE_BGR)),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.9), layout="constrained")
    for ax, (title, vis) in zip(axes, panels):
        ax.imshow(vis[:, :, ::-1])
        ax.set_title(title, fontsize=11)
        ax.axis("off")
    fig.suptitle("合成几何图：平坦处几乎无点，角与棋盘上很密", fontsize=13)
    save_fig(fig, "ch05_synthetic_corners.png")
    save_bgr("ch05_shape_board.png", img)


def plot_realistic_features():
    img = make_realistic_scene()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    pts = detect_gft(gray, max_corners=120, quality=0.01, min_dist=10)
    vis = draw_points_bgr(img, pts, WAVE_BGR, radius=3)

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.0), layout="constrained")
    axes[0].imshow(img[:, :, ::-1])
    axes[0].set_title("真实感合成场景")
    axes[0].axis("off")
    n = 0 if pts is None else len(pts)
    axes[1].imshow(vis[:, :, ::-1])
    axes[1].set_title(f"goodFeaturesToTrack：{n} 个点")
    axes[1].axis("off")
    fig.suptitle("窗户、砖缝、路牌角容易被选中；天空/路面很少", fontsize=13)
    save_fig(fig, "ch05_realistic_features.png")
    save_bgr("ch05_realistic_scene.png", img)


def plot_scale_blur_bright():
    img = make_realistic_scene()
    gray0 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 尺度：缩小再放大回原尺寸（丢细节）
    small = cv2.resize(img, (img.shape[1] // 3, img.shape[0] // 3), interpolation=cv2.INTER_AREA)
    scaled = cv2.resize(small, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
    # 模糊
    blurred = cv2.GaussianBlur(img, (15, 15), 0)
    # 变暗
    bright = np.clip(img.astype(np.float32) * 0.45, 0, 255).astype(np.uint8)

    variants = [
        ("原图", img, gray0),
        ("尺度变小（丢细节）", scaled, cv2.cvtColor(scaled, cv2.COLOR_BGR2GRAY)),
        ("强模糊", blurred, cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)),
        ("整体变暗", bright, cv2.cvtColor(bright, cv2.COLOR_BGR2GRAY)),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.2), layout="constrained")
    for ax, (title, bgr, gray) in zip(axes.ravel(), variants):
        pts = detect_gft(gray, max_corners=100, quality=0.01, min_dist=10)
        vis = draw_points_bgr(bgr, pts, WAVE_BGR, radius=3)
        n = 0 if pts is None else len(pts)
        ax.imshow(vis[:, :, ::-1])
        ax.set_title(f"{title} · {n} 点")
        ax.axis("off")
    fig.suptitle("同一套参数下：尺度/模糊/亮度怎样影响检测数量与位置", fontsize=13)
    save_fig(fig, "ch05_scale_blur_bright.png")


def plot_opencv_api_compare():
    img = make_realistic_scene()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    gft = detect_gft(gray, max_corners=100, quality=0.01, min_dist=10)
    harris = detect_harris_points(gray, threshold_ratio=0.015, max_points=100)
    fast = detect_fast(gray, threshold=28)

    panels = [
        (f"goodFeaturesToTrack\nn={0 if gft is None else len(gft)}", draw_points_bgr(img, gft)),
        (f"cornerHarris + 阈值\nn={0 if harris is None else len(harris)}", draw_points_bgr(img, harris, TEAL_BGR)),
        (f"FastFeatureDetector\nn={0 if fast is None else len(fast)}", draw_points_bgr(img, fast)),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.9), layout="constrained")
    for ax, (title, vis) in zip(axes, panels):
        ax.imshow(vis[:, :, ::-1])
        ax.set_title(title, fontsize=11)
        ax.axis("off")
    fig.suptitle("OpenCV 三种常用入口（同一场景，参数各自调到「看起来合理」）", fontsize=13)
    save_fig(fig, "ch05_opencv_compare.png")


def plot_mine():
    img = make_realistic_scene()
    work = img
    if MY_BLUR and MY_BLUR >= 3:
        k = MY_BLUR if MY_BLUR % 2 == 1 else MY_BLUR + 1
        work = cv2.GaussianBlur(img, (k, k), 0)
    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
    pts = detect_gft(
        gray,
        max_corners=int(MY_MAX_CORNERS),
        quality=float(MY_QUALITY),
        min_dist=float(MY_MIN_DISTANCE),
    )
    vis = draw_points_bgr(work, pts, WAVE_BGR, radius=3)
    n = 0 if pts is None else len(pts)

    fig, ax = plt.subplots(figsize=(9.0, 5.0), layout="constrained")
    ax.imshow(vis[:, :, ::-1])
    ax.set_title(
        f"你的参数：max={MY_MAX_CORNERS}, quality={MY_QUALITY}, "
        f"minDist={MY_MIN_DISTANCE}, blur={MY_BLUR} → {n} 点"
    )
    ax.axis("off")
    save_fig(fig, "ch05_my_features.png")


def self_check():
    img = make_shape_board()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 1) 平坦大块里不应有大量点；棋盘区域应有点
    pts = detect_gft(gray, max_corners=80, quality=0.01, min_dist=6)
    assert pts is not None and len(pts) >= 8, "合成图应能检出若干角点"

    # 2) Harris 响应：平坦灰块中心几乎为 0；L 形内角附近应显著更大
    resp = cv2.cornerHarris(np.float32(gray), 2, 3, 0.04)
    flat_peak = float(np.abs(resp[50:90, 50:110]).max())
    # L 形内角约在 (140, 200)
    corner_peak = float(np.abs(resp[190:210, 130:150]).max())
    assert corner_peak > flat_peak * 5, "角点附近 Harris 响应应远大于平坦区"

    # 3) FAST 在棋盘上应检出点
    fast = detect_fast(gray, threshold=20)
    assert fast is not None and len(fast) > 0, "FAST 应能检出点"

    # 4) 模糊后点数通常下降（同参数）
    blur = cv2.GaussianBlur(gray, (21, 21), 0)
    p0 = detect_gft(gray, 100, 0.01, 8)
    p1 = detect_gft(blur, 100, 0.01, 8)
    n0 = 0 if p0 is None else len(p0)
    n1 = 0 if p1 is None else len(p1)
    assert n1 <= n0, "强模糊后特征点不应变多"

    # 5) API 形状检查
    assert cv2.FastFeatureDetector_create(25) is not None
    print("自检通过：平坦/角点响应、三种检测器、模糊影响均符合预期。")


def main():
    use_style()
    self_check()
    plot_sliding_window_intuition()
    plot_harris_idea()
    plot_fast_idea()
    plot_detections_on_shapes()
    plot_realistic_features()
    plot_scale_blur_bright()
    plot_opencv_api_compare()
    plot_mine()
    print("课文对照图已写入 assets/figures/")
    print("你的图：ch05_my_features.png")


if __name__ == "__main__":
    main()
