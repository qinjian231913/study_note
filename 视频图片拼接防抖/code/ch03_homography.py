"""第 3 篇：单应性 H——两平面之间的通行证。

    python code/ch03_homography.py

顶部「改这里」只影响 ch03_my_overlay.png。
"""

from pathlib import Path

SHOW_WINDOW = False

import matplotlib

if not SHOW_WINDOW:
    matplotlib.use("Agg")

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

from plot_style import INK, MUTED, TEAL, WAVE, use_style

# ===== 改这里 =====
# 图 B 四角在图 A 坐标系里的目标位置（像素）。
# 顺序：左上、右上、右下、左下。改完再跑脚本，看 ch03_my_overlay.png。
MY_DST_QUAD = np.array(
    [
        [180.0, 40.0],
        [420.0, 50.0],
        [430.0, 260.0],
        [160.0, 250.0],
    ],
    dtype=np.float32,
)
# 叠图时图 B 的透明度（0~1）
MY_ALPHA = 0.45
# ==================

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "assets" / "figures"
IMG = ROOT / "assets" / "images"
WAVE_BGR = (40, 77, 194)
TEAL_BGR = (102, 107, 14)
INK_BGR = (28, 25, 21)


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


def make_poster(w=280, h=200, label="A", accent=WAVE_BGR):
    """模拟墙上海报：棋盘 + 边框 + 角标，便于看出对齐。"""
    img = np.full((h, w, 3), 248, dtype=np.uint8)
    cell = 28
    for y in range(0, h, cell):
        for x in range(0, w, cell):
            if ((x // cell) + (y // cell)) % 2 == 0:
                img[y : y + cell, x : x + cell] = (228, 216, 198)
    cv2.rectangle(img, (8, 8), (w - 9, h - 9), (55, 48, 42), 3)
    # 对角线帮助看扭曲
    cv2.line(img, (8, 8), (w - 9, h - 9), (160, 150, 140), 1)
    cv2.line(img, (w - 9, 8), (8, h - 9), (160, 150, 140), 1)
    cv2.putText(
        img, label, (18, 48), cv2.FONT_HERSHEY_SIMPLEX, 1.4, accent, 3, cv2.LINE_AA
    )
    cv2.putText(
        img,
        "POSTER",
        (w // 2 - 55, h // 2 + 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (90, 85, 80),
        2,
        cv2.LINE_AA,
    )
    return img


def make_pair_planar(shift_x=90, yaw_scale=0.92):
    """同一平面海报：图 A 正面，图 B 略斜视+平移，可用一张 H 对齐。

    返回 (canvas_a, canvas_b, pts_a, pts_b)，pts 是海报四角在各自画布上的像素坐标。
    """
    base = make_poster(label="WALL", accent=WAVE_BGR)
    h, w = base.shape[:2]
    ax0, ay0 = 20, 20
    canvas_h, canvas_w = h + 40, w + 160
    canvas_a = np.full((canvas_h, canvas_w, 3), 251, dtype=np.uint8)
    canvas_a[ay0 : ay0 + h, ax0 : ax0 + w] = base
    pts_a = np.float32(
        [
            [ax0, ay0],
            [ax0 + w - 1, ay0],
            [ax0 + w - 1, ay0 + h - 1],
            [ax0, ay0 + h - 1],
        ]
    )

    src = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    dst_local = np.float32(
        [
            [10, 15],
            [w * yaw_scale - 5, 5],
            [w * yaw_scale + 8, h - 8],
            [5, h - 12],
        ]
    )
    h_gen = cv2.getPerspectiveTransform(src, dst_local)
    warp_w, warp_h = w + 40, h + 20
    warped = cv2.warpPerspective(base, h_gen, (warp_w, warp_h), borderValue=(251, 251, 251))
    canvas_b = np.full((canvas_h, canvas_w, 3), 251, dtype=np.uint8)
    y0, x0 = 10, shift_x
    bh, bw = warped.shape[:2]
    y1 = min(y0 + bh, canvas_h)
    x1 = min(x0 + bw, canvas_w)
    canvas_b[y0:y1, x0:x1] = warped[: y1 - y0, : x1 - x0]
    # 海报四角在 B 画布上 = 局部透视角 + 平移
    pts_b = dst_local.copy()
    pts_b[:, 0] += x0
    pts_b[:, 1] += y0
    cv2.putText(
        canvas_b, "B", (x0 + 18, y0 + 48), cv2.FONT_HERSHEY_SIMPLEX, 1.2, TEAL_BGR, 3, cv2.LINE_AA
    )
    cv2.putText(
        canvas_a, "A", (28, 58), cv2.FONT_HERSHEY_SIMPLEX, 1.2, WAVE_BGR, 3, cv2.LINE_AA
    )
    return canvas_a, canvas_b, pts_a, pts_b


def make_pair_parallax():
    """近景双层：远墙用同一 H 可对齐，近处立柱深度不同 → 单应失败。"""
    h, w = 220, 360
    # 远景墙（平面纹理）
    wall = np.full((h, w, 3), 235, dtype=np.uint8)
    for x in range(0, w, 40):
        cv2.line(wall, (x, 0), (x, h), (200, 190, 180), 1)
    for y in range(0, h, 40):
        cv2.line(wall, (0, y), (w, y), (200, 190, 180), 1)
    cv2.putText(wall, "FAR WALL", (110, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (90, 85, 80), 2)

    # 近景立柱（矩形色块，两张图里水平位移不同）
    def place_pole(bg, pole_x, color):
        out = bg.copy()
        cv2.rectangle(out, (pole_x, 40), (pole_x + 36, h - 20), color, -1)
        cv2.putText(out, "NEAR", (pole_x - 4, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
        return out

    img_a = place_pole(wall, 140, WAVE_BGR)
    # 图 B：墙整体右移少，立柱右移多（视差）
    wall_b = np.full_like(wall, 251)
    # 墙用轻微平移模拟另一视角下的远景
    M_wall = np.float32([[1, 0, 18], [0, 1, 0]])
    wall_shifted = cv2.warpAffine(wall, M_wall, (w, h), borderValue=(251, 251, 251))
    img_b = place_pole(wall_shifted, 200, TEAL_BGR)  # 立柱位移 60，墙只移 18
    cv2.putText(img_a, "A", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, WAVE_BGR, 2)
    cv2.putText(img_b, "B", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, TEAL_BGR, 2)
    return img_a, img_b


def corners_of(img):
    h, w = img.shape[:2]
    return np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])


def apply_h_pts(H, pts):
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 1, 2)
    out = cv2.perspectiveTransform(pts, H)
    return out.reshape(-1, 2)


def canvas_from_corners(corners_a, corners_b_warped, margin=8):
    """根据图 A 四角与 warp 后的图 B 四角，算能装下两者的画布与平移。"""
    all_pts = np.vstack([corners_a, corners_b_warped])
    min_xy = all_pts.min(axis=0)
    max_xy = all_pts.max(axis=0)
    # 若有负坐标，整体平移
    offset = np.array([0.0, 0.0])
    if min_xy[0] < 0:
        offset[0] = -min_xy[0] + margin
    else:
        offset[0] = margin
    if min_xy[1] < 0:
        offset[1] = -min_xy[1] + margin
    else:
        offset[1] = margin
    size = (
        int(np.ceil(max_xy[0] - min_xy[0] + 2 * margin)),
        int(np.ceil(max_xy[1] - min_xy[1] + 2 * margin)),
    )
    T = np.array([[1, 0, offset[0]], [0, 1, offset[1]], [0, 0, 1]], dtype=np.float64)
    return size, T


def overlay_images(img_a, img_b, H, alpha=0.45):
    """把 B 经 H 画到 A 坐标系，自动扩画布，半透明叠图。"""
    ha, wa = img_a.shape[:2]
    corners_a = corners_of(img_a)
    corners_b = apply_h_pts(H, corners_of(img_b))
    size, T = canvas_from_corners(corners_a, corners_b)
    H_adj = T @ H
    warped_b = cv2.warpPerspective(img_b, H_adj, size, borderValue=(251, 251, 251))
    canvas_a = cv2.warpPerspective(img_a, T, size, borderValue=(251, 251, 251))
    # 只在 B 有内容处混合
    mask = (warped_b.sum(axis=2) < 250 * 3).astype(np.float32)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)[..., None]
    blend = canvas_a.astype(np.float32) * (1 - alpha * mask) + warped_b.astype(np.float32) * (
        alpha * mask
    )
    # A 独占区域保持清晰
    only_a = (canvas_a.sum(axis=2) < 250 * 3) & (warped_b.sum(axis=2) >= 250 * 3)
    blend[only_a] = canvas_a[only_a]
    return blend.astype(np.uint8), warped_b, canvas_a, size, T, H_adj


def plot_naming():
    fig, ax = plt.subplots(figsize=(9.0, 3.8), layout="constrained")
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.set_title("第 2 篇的透视矩阵，正式名叫单应性 Homography")
    ax.add_patch(
        FancyBboxPatch(
            (0.4, 1.2), 3.2, 2.0, boxstyle="round,pad=0.15", facecolor="#fde8df", edgecolor=WAVE, lw=1.5
        )
    )
    ax.text(2.0, 2.6, "第 2 篇", ha="center", fontsize=12, color=MUTED)
    ax.text(2.0, 2.0, "透视 / 射影变换", ha="center", fontsize=14)
    ax.text(2.0, 1.5, r"$3\times3$ 矩阵 $H$", ha="center", fontsize=13)

    ax.add_patch(
        FancyBboxPatch(
            (6.4, 1.2), 3.2, 2.0, boxstyle="round,pad=0.15", facecolor="#e5f3f1", edgecolor=TEAL, lw=1.5
        )
    )
    ax.text(8.0, 2.6, "第 3 篇正式名", ha="center", fontsize=12, color=MUTED)
    ax.text(8.0, 2.0, "单应性 Homography", ha="center", fontsize=14)
    ax.text(8.0, 1.5, "两平面之间的通行证", ha="center", fontsize=13)

    ax.annotate(
        "",
        xy=(6.3, 2.2),
        xytext=(3.7, 2.2),
        arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.8),
    )
    ax.text(5.0, 2.55, "同一个东西", ha="center", fontsize=12, color=MUTED)
    ax.text(
        5.0,
        0.45,
        r"写法：$(x',y',1)^\top \sim H\,(x,y,1)^\top$，再除以第三维回到像素",
        ha="center",
        fontsize=12,
    )
    save_fig(fig, "ch03_naming.png")


def plot_when_ok():
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.8), layout="constrained")
    titles = [
        "① 真正的平面\n墙、海报、桌面",
        "② 远景近似平面\n山、楼群很远",
        "③ 相机几乎纯旋转\n脚不挪，只转头",
    ]
    notes = [
        "一张 H 足够\n（理想条件）",
        "深度差相对很小\n近似一张 H",
        "所有点可看成\n在无穷远平面上",
    ]
    colors = ["#fde8df", "#e5f3f1", "#f3e6d8"]
    for ax, title, note, c in zip(axes, titles, notes, colors):
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis("off")
        ax.add_patch(
            FancyBboxPatch(
                (0.6, 2.5), 8.8, 6.0, boxstyle="round,pad=0.2", facecolor=c, edgecolor="#1c1915", lw=1.2
            )
        )
        ax.text(5, 7.2, title, ha="center", va="center", fontsize=12)
        ax.text(5, 4.2, note, ha="center", va="center", fontsize=11, color=MUTED)
        ax.set_title("")
    fig.suptitle("何时一张 3×3 的 H 就够用？", fontsize=14)
    save_fig(fig, "ch03_when_ok.png")


def plot_estimate_demo():
    img_a, img_b, pts_a, pts_b = make_pair_planar()
    H = cv2.getPerspectiveTransform(pts_b, pts_a)

    vis_a = img_a.copy()
    vis_b = img_b.copy()
    for i, (pa, pb) in enumerate(zip(pts_a, pts_b)):
        color = WAVE_BGR if i % 2 == 0 else TEAL_BGR
        cv2.circle(vis_a, (int(pa[0]), int(pa[1])), 6, color, -1)
        cv2.circle(vis_b, (int(pb[0]), int(pb[1])), 6, color, -1)
        cv2.putText(vis_a, "ABCD"[i], (int(pa[0]) + 6, int(pa[1]) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        cv2.putText(vis_b, "ABCD"[i], (int(pb[0]) + 6, int(pb[1]) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    cv2.polylines(vis_a, [pts_a.astype(np.int32)], True, INK_BGR, 2)
    cv2.polylines(vis_b, [pts_b.astype(np.int32)], True, INK_BGR, 2)

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2), layout="constrained")
    axes[0].imshow(vis_b[:, :, ::-1])
    axes[0].set_title("图 B：标 4 个对应点（源）")
    axes[0].axis("off")
    axes[1].imshow(vis_a[:, :, ::-1])
    axes[1].set_title("图 A：同一物体的 4 个点（目标）")
    axes[1].axis("off")
    save_fig(fig, "ch03_correspondences.png")
    save_bgr("ch03_img_a.png", img_a)
    save_bgr("ch03_img_b.png", img_b)

    # 矩阵图
    fig, ax = plt.subplots(figsize=(7.6, 3.4), layout="constrained")
    ax.axis("off")
    ax.set_title("用 4 对点估出的 H（getPerspectiveTransform）")
    lines = ["  ".join(f"{H[r, c]:9.4f}" for c in range(3)) for r in range(3)]
    ax.text(0.04, 0.55, "\n".join(lines), fontsize=13, family="monospace", transform=ax.transAxes)
    ax.text(
        0.04,
        0.15,
        "点更多时用 findHomography（常配 RANSAC，见第 7 篇）",
        fontsize=12,
        color=MUTED,
        transform=ax.transAxes,
    )
    save_fig(fig, "ch03_h_matrix.png")
    return img_a, img_b, H, pts_a, pts_b


def plot_warp_and_canvas(img_a, img_b, H):
    blend, warped_b, canvas_a, size, T, H_adj = overlay_images(img_a, img_b, H, alpha=0.5)

    # 固定小画布裁切对比
    ha, wa = img_a.shape[:2]
    cropped = cv2.warpPerspective(img_b, H, (wa, ha), borderValue=(251, 251, 251))
    crop_blend = cv2.addWeighted(img_a, 0.55, cropped, 0.45, 0)

    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.8), layout="constrained")
    axes[0].imshow(img_a[:, :, ::-1])
    axes[0].set_title("图 A（参考坐标系）")
    axes[0].axis("off")
    axes[1].imshow(warped_b[:, :, ::-1])
    axes[1].set_title("图 B 经 H 变到 A 系\n（已扩画布）")
    axes[1].axis("off")
    axes[2].imshow(blend[:, :, ::-1])
    axes[2].set_title("半透明叠图：对齐效果")
    axes[2].axis("off")
    save_fig(fig, "ch03_warp_overlay.png")

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0), layout="constrained")
    axes[0].imshow(crop_blend[:, :, ::-1])
    axes[0].set_title("画布 = 图 A 大小：B 超出部分被裁掉")
    axes[0].axis("off")
    axes[1].imshow(blend[:, :, ::-1])
    axes[1].set_title(f"按四角包络扩画布 {size[0]}×{size[1]}：两边都保住")
    axes[1].axis("off")
    save_fig(fig, "ch03_canvas_size.png")

    # 文字步骤图
    fig, ax = plt.subplots(figsize=(9.2, 3.6), layout="constrained")
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.set_title("算画布大小的三步（避免裁切）")
    steps = [
        "1. 用 H 变换\n图 B 的四个角",
        "2. 与图 A 四角\n一起取包围盒",
        "3. 若有负坐标\n平移，再 warp",
    ]
    xs = [1.2, 5.0, 8.4]
    for x, s in zip(xs, steps):
        ax.add_patch(
            FancyBboxPatch(
                (x - 1.35, 1.0),
                2.7,
                2.2,
                boxstyle="round,pad=0.12",
                facecolor="#fffdf8",
                edgecolor=TEAL,
                lw=1.4,
            )
        )
        ax.text(x, 2.1, s, ha="center", va="center", fontsize=11)
    for i in range(2):
        ax.annotate(
            "",
            xy=(xs[i + 1] - 1.45, 2.1),
            xytext=(xs[i] + 1.45, 2.1),
            arrowprops=dict(arrowstyle="->", color=WAVE, lw=1.6),
        )
    save_fig(fig, "ch03_canvas_steps.png")
    return blend


def plot_parallax_fail():
    img_a, img_b = make_pair_parallax()
    # 只用远墙上的 4 点估 H（墙能对齐），近景立柱必然错位
    pts_b = np.float32([[40, 40], [320, 40], [320, 180], [40, 180]])
    pts_a = np.float32([[22, 40], [302, 40], [302, 180], [22, 180]])
    H = cv2.getPerspectiveTransform(pts_b, pts_a)
    blend, _, _, _, _, _ = overlay_images(img_a, img_b, H, alpha=0.55)

    # 在叠图上圈出失败处
    vis = blend.copy()
    cv2.rectangle(vis, (130, 30), (250, 210), WAVE_BGR, 2)
    cv2.putText(vis, "NEAR: ghost / offset", (100, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, WAVE_BGR, 2)
    cv2.putText(vis, "FAR wall ~aligned", (100, 215), cv2.FONT_HERSHEY_SIMPLEX, 0.55, TEAL_BGR, 2)

    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.6), layout="constrained")
    axes[0].imshow(img_a[:, :, ::-1])
    axes[0].set_title("图 A：远墙 + 近柱")
    axes[0].axis("off")
    axes[1].imshow(img_b[:, :, ::-1])
    axes[1].set_title("图 B：柱移得比墙多")
    axes[1].axis("off")
    axes[2].imshow(vis[:, :, ::-1])
    axes[2].set_title("一张 H：墙还行，柱重影")
    axes[2].axis("off")
    save_fig(fig, "ch03_parallax_fail.png")
    save_bgr("ch03_parallax_a.png", img_a)
    save_bgr("ch03_parallax_b.png", img_b)

    # 白话示意图
    fig, ax = plt.subplots(figsize=(8.8, 3.8), layout="constrained")
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.set_title("视差白话：近处物体「滑动」得更远")
    ax.text(5, 4.2, "同一侧移相机：近柱在画面上位移大，远墙位移小", ha="center", fontsize=12)
    # 简化相机与两深度
    ax.plot([1.5, 1.5], [0.8, 2.2], color=INK, lw=3)
    ax.text(1.5, 0.4, "相机", ha="center", fontsize=11)
    ax.plot([5.5, 5.5], [0.8, 3.5], color=WAVE, lw=8, solid_capstyle="butt")
    ax.text(5.5, 0.4, "近柱", ha="center", color=WAVE, fontsize=11)
    ax.plot([8.5, 8.5], [0.8, 4.0], color=TEAL, lw=14, solid_capstyle="butt", alpha=0.5)
    ax.text(8.5, 0.4, "远墙", ha="center", color=TEAL, fontsize=11)
    ax.annotate("", xy=(6.3, 2.5), xytext=(5.5, 2.5), arrowprops=dict(arrowstyle="->", color=WAVE, lw=2))
    ax.annotate("", xy=(8.9, 2.5), xytext=(8.5, 2.5), arrowprops=dict(arrowstyle="->", color=TEAL, lw=2))
    ax.text(6.0, 3.0, "大位移", color=WAVE, fontsize=11)
    ax.text(8.7, 3.0, "小位移", color=TEAL, fontsize=11)
    ax.text(5, 4.85, "", fontsize=1)
    ax.text(
        5,
        4.55,
        "一张 H 只能选一种「滑动速度」→ 不能同时服侍近与远",
        ha="center",
        fontsize=12,
        color=MUTED,
    )
    save_fig(fig, "ch03_parallax_idea.png")


def plot_mine():
    img_a, img_b, _, _ = make_pair_planar()
    src = corners_of(img_b)
    # 用用户四角作为「图 B 整幅映到 A 画布上的目标」
    H = cv2.getPerspectiveTransform(src, MY_DST_QUAD)
    blend, _, _, _, _, _ = overlay_images(img_a, img_b, H, alpha=MY_ALPHA)
    vis = blend.copy()
    mapped = apply_h_pts(H, src)
    # 画到扩画布坐标系需要 T——这里直接在 blend 上标 MY_DST 可能不准；
    # 改用 overlay 返回的方式：重新算
    corners_a = corners_of(img_a)
    corners_bw = apply_h_pts(H, src)
    _, T = canvas_from_corners(corners_a, corners_bw)
    mapped_c = apply_h_pts(T @ H, src)
    cv2.polylines(vis, [mapped_c.astype(np.int32)], True, TEAL_BGR, 2)
    for i, (x, y) in enumerate(mapped_c):
        cv2.circle(vis, (int(x), int(y)), 5, WAVE_BGR if i % 2 == 0 else TEAL_BGR, -1)

    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.0), layout="constrained")
    axes[0].imshow(img_b[:, :, ::-1])
    axes[0].set_title("图 B（整幅四角作源）")
    axes[0].axis("off")
    axes[1].imshow(vis[:, :, ::-1])
    axes[1].set_title(f"你的 MY_DST_QUAD 叠图（alpha={MY_ALPHA}）")
    axes[1].axis("off")
    save_fig(fig, "ch03_my_overlay.png")


def plot_api_cheat():
    fig, ax = plt.subplots(figsize=(9.0, 4.0), layout="constrained")
    ax.axis("off")
    ax.set_title("OpenCV 两件套（本篇核心 API）")
    ax.text(
        0.03,
        0.78,
        "H = cv2.getPerspectiveTransform(src4, dst4)",
        fontsize=12,
        family="monospace",
        transform=ax.transAxes,
        va="top",
    )
    ax.text(0.03, 0.68, "→ 正好 4 对点，精确解", fontsize=12, color=MUTED, transform=ax.transAxes, va="top")
    ax.text(
        0.03,
        0.52,
        "H, mask = cv2.findHomography(srcN, dstN, cv2.RANSAC)",
        fontsize=12,
        family="monospace",
        transform=ax.transAxes,
        va="top",
    )
    ax.text(0.03, 0.42, "→ N≥4，可剔野点（第 7 篇）", fontsize=12, color=MUTED, transform=ax.transAxes, va="top")
    ax.text(
        0.03,
        0.26,
        "out = cv2.warpPerspective(img_b, H, (W, H_out))",
        fontsize=12,
        family="monospace",
        transform=ax.transAxes,
        va="top",
    )
    ax.text(0.03, 0.16, "→ 把图 B 画到新画布", fontsize=12, color=MUTED, transform=ax.transAxes, va="top")
    ax.text(
        0.03,
        0.04,
        "口诀：先估 H（点从 B → A），再 warpPerspective（图 B 进 A 的坐标系）",
        fontsize=12,
        color=MUTED,
        transform=ax.transAxes,
    )
    save_fig(fig, "ch03_api.png")


def self_check():
    # 恒等单应
    sq = np.float32([[0, 0], [10, 0], [10, 10], [0, 10]])
    H = cv2.getPerspectiveTransform(sq, sq)
    assert H.shape == (3, 3)
    # 尺度等价：λH 与 H 表示同一变换（对一点）
    p = np.array([3.0, 4.0, 1.0])
    q1 = H @ p
    q1 = q1[:2] / q1[2]
    q2 = (2.5 * H) @ p
    q2 = q2[:2] / q2[2]
    assert np.allclose(q1, q2)

    # findHomography 多点
    src = np.float32([[0, 0], [20, 0], [20, 10], [0, 10], [10, 5]])
    dst = src + np.float32([5, -3])
    H2, _ = cv2.findHomography(src, dst, 0)
    assert H2 is not None and H2.shape == (3, 3)

    # 画布偏移：负坐标应被平移
    corners = np.array([[-10.0, -5.0], [100.0, -5.0], [100.0, 80.0], [-10.0, 80.0]])
    size, T = canvas_from_corners(np.array([[0.0, 0.0], [50, 0], [50, 40], [0, 40]]), corners)
    assert T[0, 2] > 0 and T[1, 2] > 0
    assert size[0] > 50 and size[1] > 40

    print("自检通过：恒等 H、尺度等价、findHomography、画布平移。")


def main():
    use_style()
    self_check()
    plot_naming()
    plot_when_ok()
    plot_api_cheat()
    img_a, img_b, H, _, _ = plot_estimate_demo()
    plot_warp_and_canvas(img_a, img_b, H)
    plot_parallax_fail()
    plot_mine()
    print("课文对照图已写入 assets/figures/")
    print("你的图：ch03_my_overlay.png")


if __name__ == "__main__":
    main()
