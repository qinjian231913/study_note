"""第 8 篇：全景拼接流水线总览。

把「检测 → 匹配 → RANSAC 估 H → 算画布 → warp 叠图」串成可运行的两图拼接，
并用合成重叠图演示多图参照帧与误差累积直觉。

    python code/ch08_stitch.py

顶部「改这里」只影响 ch08_my_result.png 等「你的」结果图。
"""

from pathlib import Path

SHOW_WINDOW = False

import matplotlib

if not SHOW_WINDOW:
    matplotlib.use("Agg")

import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.patches import FancyBboxPatch

from plot_style import INK, MUTED, TEAL, WAVE, use_style

# ===== 改这里 =====
# 合成右图相对左图的水平平移（像素，越大重叠越少）
SHIFT_X = 140
# 额外轻微旋转（度），模拟手持转头时不完全是纯平移
SHIFT_ANGLE_DEG = 2.5
# ORB 特征点数；太少 → 匹配不稳；太多 → 稍慢
N_FEATURES = 1200
# Lowe 比率：越小匹配越「严」
LOWE_RATIO = 0.75
# RANSAC 重投影阈值（像素）
RANSAC_THRESH = 4.0
# 叠图时右图透明度（0=全透明，1=完全盖住）
BLEND_ALPHA = 0.55
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


# ---------------------------------------------------------------------------
# 合成场景与两张重叠图
# ---------------------------------------------------------------------------


def make_scene(w=640, h=320):
    """画一张带纹理的「街景」大图，方便 ORB 找到角点。"""
    img = np.full((h, w, 3), 236, dtype=np.uint8)

    # 天空 / 地面
    img[: h // 2, :] = (232, 214, 188)
    img[h // 2 :, :] = (210, 200, 186)

    # 棋盘地面纹理
    cell = 18
    for y in range(h // 2, h, cell):
        for x in range(0, w, cell):
            if ((x // cell) + (y // cell)) % 2 == 0:
                img[y : y + cell, x : x + cell] = (198, 188, 172)

    # 几栋「楼」
    buildings = [
        (40, 70, 110, h // 2, (120, 98, 78)),
        (170, 50, 90, h // 2, (95, 110, 130)),
        (280, 80, 100, h // 2, (110, 88, 70)),
        (400, 45, 95, h // 2, (88, 105, 92)),
        (520, 75, 90, h // 2, (125, 100, 85)),
    ]
    for x, y, bw, bh, color in buildings:
        cv2.rectangle(img, (x, y), (x + bw, y + bh), color, -1)
        # 窗户格子
        for wy in range(y + 12, y + bh - 10, 22):
            for wx in range(x + 10, x + bw - 10, 20):
                cv2.rectangle(img, (wx, wy), (wx + 10, wy + 12), (210, 220, 230), -1)

    # 路标 / 竖线，给特征点多一些「钩子」
    for x in (60, 200, 340, 480, 600):
        cv2.line(img, (x, h // 2), (x, h - 10), (70, 65, 60), 2)
    cv2.putText(img, "A", (55, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, WAVE_BGR, 2, cv2.LINE_AA)
    cv2.putText(img, "B", (300, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.2, TEAL_BGR, 2, cv2.LINE_AA)
    cv2.putText(img, "C", (530, 65), cv2.FONT_HERSHEY_SIMPLEX, 1.2, WAVE_BGR, 2, cv2.LINE_AA)
    return img


def crop_pair(scene, shift_x=140, angle_deg=2.5, view_w=360, view_h=240):
    """从大场景裁出左图，并把右图做成「平移+微旋」后再裁，保证有重叠。"""
    sh, sw = scene.shape[:2]
    x0, y0 = 40, 40
    left = scene[y0 : y0 + view_h, x0 : x0 + view_w].copy()

    # 以场景中心附近为轴，把整张场景旋转再平移，再裁右图
    cx, cy = sw / 2.0, sh / 2.0
    m = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)
    m[0, 2] += shift_x
    warped = cv2.warpAffine(scene, m, (sw, sh), borderValue=(236, 236, 236))
    x1 = x0 + int(shift_x * 0.55)
    y1 = y0 + 8
    # 保证裁切框在范围内
    x1 = int(np.clip(x1, 0, sw - view_w))
    y1 = int(np.clip(y1, 0, sh - view_h))
    right = warped[y1 : y1 + view_h, x1 : x1 + view_w].copy()
    return left, right


# ---------------------------------------------------------------------------
# 流水线核心
# ---------------------------------------------------------------------------


def detect_and_match(img_a, img_b, n_features=1200, ratio=0.75):
    orb = cv2.ORB_create(nfeatures=n_features)
    kp1, des1 = orb.detectAndCompute(img_a, None)
    kp2, des2 = orb.detectAndCompute(img_b, None)
    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        return kp1 or [], kp2 or [], [], []

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    raw = bf.knnMatch(des1, des2, k=2)
    good = []
    for pair in raw:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < ratio * n.distance:
            good.append(m)
    return kp1, kp2, good, raw


def estimate_homography(kp1, kp2, matches, ransac_thresh=4.0):
    if len(matches) < 4:
        return None, None
    # 把「右图」变到「左图」坐标系：src=右图点，dst=左图点 → 左 ≈ H · 右
    src_r = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_l = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src_r, dst_l, cv2.RANSAC, ransac_thresh)
    return H, mask


def compute_canvas(img_a, img_b, H):
    """把右图四角经 H 映到左图坐标系，再与左图外接矩形求并，得到画布尺寸与偏移。"""
    h_a, w_a = img_a.shape[:2]
    h_b, w_b = img_b.shape[:2]
    corners_b = np.float32([[0, 0], [w_b, 0], [w_b, h_b], [0, h_b]]).reshape(-1, 1, 2)
    warped_corners = cv2.perspectiveTransform(corners_b, H)
    corners_a = np.float32([[0, 0], [w_a, 0], [w_a, h_a], [0, h_a]]).reshape(-1, 1, 2)
    all_corners = np.concatenate([corners_a, warped_corners], axis=0)
    xs = all_corners[:, 0, 0]
    ys = all_corners[:, 0, 1]
    min_x, max_x = np.floor(xs.min()), np.ceil(xs.max())
    min_y, max_y = np.floor(ys.min()), np.ceil(ys.max())
    # 平移矩阵：把负坐标挪到画布原点
    tx, ty = -min_x, -min_y
    T = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]], dtype=np.float64)
    canvas_w = int(max_x - min_x)
    canvas_h = int(max_y - min_y)
    return canvas_w, canvas_h, T, warped_corners


def warp_and_blend(img_a, img_b, H, canvas_w, canvas_h, T, alpha=0.55):
    H_on_canvas = T @ H
    warped_b = cv2.warpPerspective(img_b, H_on_canvas, (canvas_w, canvas_h))
    canvas_a = cv2.warpPerspective(img_a, T, (canvas_w, canvas_h))

    mask_a = (canvas_a.sum(axis=2) > 0).astype(np.float32)
    mask_b = (warped_b.sum(axis=2) > 0).astype(np.float32)
    both = (mask_a * mask_b)[..., None]
    only_a = (mask_a * (1 - mask_b))[..., None]
    only_b = (mask_b * (1 - mask_a))[..., None]

    a_f = canvas_a.astype(np.float32)
    b_f = warped_b.astype(np.float32)
    out = (
        only_a * a_f
        + only_b * b_f
        + both * ((1 - alpha) * a_f + alpha * b_f)
    )
    return out.astype(np.uint8), canvas_a, warped_b


def stitch_pair(img_a, img_b, n_features, ratio, ransac_thresh, alpha):
    kp1, kp2, good, _ = detect_and_match(img_a, img_b, n_features, ratio)
    H, mask = estimate_homography(kp1, kp2, good, ransac_thresh)
    info = {
        "kp1": kp1,
        "kp2": kp2,
        "good": good,
        "H": H,
        "mask": mask,
        "n_kp1": len(kp1),
        "n_kp2": len(kp2),
        "n_good": len(good),
        "n_inliers": int(mask.sum()) if mask is not None else 0,
    }
    if H is None:
        return None, info
    cw, ch, T, warped_corners = compute_canvas(img_a, img_b, H)
    result, canvas_a, warped_b = warp_and_blend(img_a, img_b, H, cw, ch, T, alpha)
    info.update(
        {
            "canvas": (cw, ch),
            "T": T,
            "warped_corners": warped_corners,
            "canvas_a": canvas_a,
            "warped_b": warped_b,
        }
    )
    return result, info


# ---------------------------------------------------------------------------
# 配图
# ---------------------------------------------------------------------------


def plot_pipeline():
    fig, ax = plt.subplots(figsize=(10.2, 3.4), layout="constrained")
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 3.2)
    ax.axis("off")
    ax.set_title("全景拼接流水线（两图版）")

    steps = [
        (0.3, "1 检测\n特征点"),
        (2.5, "2 匹配\n对应点"),
        (4.7, "3 RANSAC\n估 H"),
        (6.9, "4 算\n画布"),
        (9.1, "5 warp\n叠图"),
    ]
    colors = ["#fde8df", "#e5f3f1", "#fde8df", "#e5f3f1", "#fde8df"]
    for (x, label), facecolor in zip(steps, colors):
        box = FancyBboxPatch(
            (x, 0.9),
            1.9,
            1.5,
            boxstyle="round,pad=0.08,rounding_size=0.18",
            linewidth=1.4,
            edgecolor=INK,
            facecolor=facecolor,
        )
        ax.add_patch(box)
        ax.text(x + 0.95, 1.65, label, ha="center", va="center", fontsize=12)

    for i in range(len(steps) - 1):
        x0 = steps[i][0] + 1.9
        x1 = steps[i + 1][0]
        ax.annotate(
            "",
            xy=(x1 - 0.05, 1.65),
            xytext=(x0 + 0.05, 1.65),
            arrowprops=dict(arrowstyle="->", color=WAVE, lw=1.8),
        )

    ax.text(0.3, 0.35, "输入：两张有重叠的图　→　输出：一张更宽的拼图", fontsize=12, color=MUTED)
    save_fig(fig, "ch08_pipeline.png")


def plot_pair(left, right):
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.8), layout="constrained")
    axes[0].imshow(left[:, :, ::-1])
    axes[0].set_title("左图（参照帧）")
    axes[0].axis("off")
    axes[1].imshow(right[:, :, ::-1])
    axes[1].set_title("右图（待对齐）")
    axes[1].axis("off")
    save_fig(fig, "ch08_pair.png")
    save_bgr("ch08_left.png", left)
    save_bgr("ch08_right.png", right)


def plot_matches(left, right, kp1, kp2, good, mask=None, name="ch08_matches.png", title="匹配连线"):
    # 只画一部分，避免糊成一团
    if mask is not None:
        chosen = [m for m, keep in zip(good, mask.ravel()) if keep]
        color = TEAL_BGR
        title = title + f"（内点 {len(chosen)}）"
    else:
        chosen = good[:80]
        color = WAVE_BGR
    vis = cv2.drawMatches(
        left,
        kp1,
        right,
        kp2,
        chosen,
        None,
        matchColor=color,
        singlePointColor=(180, 170, 160),
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    fig, ax = plt.subplots(figsize=(10.5, 4.0), layout="constrained")
    ax.imshow(vis[:, :, ::-1])
    ax.set_title(title)
    ax.axis("off")
    save_fig(fig, name)


def plot_canvas_idea(left, info):
    """示意：右图四角映到左图坐标系，外接矩形 = 画布。"""
    H = info["H"]
    warped = info["warped_corners"].reshape(-1, 2)
    h_a, w_a = left.shape[:2]

    fig, ax = plt.subplots(figsize=(8.6, 4.4), layout="constrained")
    ax.set_aspect("equal")
    ax.set_title("算画布：右图四角经 H 变换后，与左图取外接矩形")

    # 左图矩形
    ax.add_patch(
        mpatches.Rectangle((0, 0), w_a, h_a, fill=True, facecolor="#e5f3f1", edgecolor=TEAL, lw=2, alpha=0.7)
    )
    ax.text(w_a / 2, h_a / 2, "左图", ha="center", va="center", color=TEAL, fontsize=13)

    # 右图变换后四边形
    poly = np.vstack([warped, warped[0]])
    ax.plot(poly[:, 0], poly[:, 1], color=WAVE, lw=2)
    ax.fill(warped[:, 0], warped[:, 1], color="#fde8df", alpha=0.55)
    ax.text(warped[:, 0].mean(), warped[:, 1].mean(), "右图经 H", ha="center", va="center", color=WAVE, fontsize=12)

    xs = list(warped[:, 0]) + [0, w_a]
    ys = list(warped[:, 1]) + [0, h_a]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    ax.add_patch(
        mpatches.Rectangle(
            (min_x, min_y),
            max_x - min_x,
            max_y - min_y,
            fill=False,
            edgecolor=INK,
            lw=1.6,
            linestyle="--",
        )
    )
    ax.text(min_x, max_y + 8, "画布外接矩形", fontsize=11, color=MUTED)

    ax.set_xlim(min_x - 30, max_x + 30)
    ax.set_ylim(max_y + 40, min_y - 30)  # 图像坐标 y 向下
    ax.set_xlabel("x（左图坐标系）")
    ax.set_ylabel("y")
    save_fig(fig, "ch08_canvas.png")


def plot_result(result, canvas_a, warped_b):
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.6), layout="constrained")
    axes[0].imshow(canvas_a[:, :, ::-1])
    axes[0].set_title("左图放到画布")
    axes[0].axis("off")
    axes[1].imshow(warped_b[:, :, ::-1])
    axes[1].set_title("右图 warp 到画布")
    axes[1].axis("off")
    axes[2].imshow(result[:, :, ::-1])
    axes[2].set_title("叠图结果")
    axes[2].axis("off")
    save_fig(fig, "ch08_result.png")
    save_bgr("ch08_panorama.png", result)


def plot_multi_ref():
    """白话示意：参照帧选中间 vs 选端点。"""
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6), layout="constrained")

    def draw_chain(ax, ref_idx, title):
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 4)
        ax.axis("off")
        ax.set_title(title)
        xs = [1.2, 3.2, 5.2, 7.2, 9.0]
        labels = ["图1", "图2", "图3", "图4", "图5"]
        for i, (x, lab) in enumerate(zip(xs, labels)):
            color = WAVE if i == ref_idx else TEAL
            face = "#fde8df" if i == ref_idx else "#e5f3f1"
            ax.add_patch(
                FancyBboxPatch(
                    (x - 0.55, 1.4),
                    1.1,
                    1.1,
                    boxstyle="round,pad=0.04,rounding_size=0.12",
                    facecolor=face,
                    edgecolor=color,
                    lw=1.6,
                )
            )
            ax.text(x, 1.95, lab, ha="center", va="center", fontsize=11)
            if i == ref_idx:
                ax.text(x, 0.7, "参照", ha="center", color=WAVE, fontsize=11)
        # 箭头：都指向参照，或链式
        if ref_idx == 0:
            for i in range(1, 5):
                ax.annotate(
                    "",
                    xy=(xs[0] + 0.55, 2.5),
                    xytext=(xs[i] - 0.55, 2.5),
                    arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.2),
                )
            ax.text(5, 3.4, "端点作参照：远端图要「连乘」很多个 H", ha="center", fontsize=11, color=MUTED)
        else:
            for i in range(5):
                if i == ref_idx:
                    continue
                ax.annotate(
                    "",
                    xy=(xs[ref_idx], 2.55),
                    xytext=(xs[i], 2.55),
                    arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.2),
                )
            ax.text(5, 3.4, "中间作参照：左右两边路程更短，误差更匀", ha="center", fontsize=11, color=MUTED)

    draw_chain(axes[0], 0, "参照 = 第 1 张（端点）")
    draw_chain(axes[1], 2, "参照 = 第 3 张（中间）")
    save_fig(fig, "ch08_multi_ref.png")


def plot_error_accum():
    """示意：链式连乘 H 时误差像「雪球」。"""
    fig, ax = plt.subplots(figsize=(8.8, 3.8), layout="constrained")
    ax.set_title("误差累积白话：每段小错，连乘后会变大")
    steps = np.arange(1, 8)
    # 理想：零误差；链式：近似随步数放大
    ideal = np.zeros_like(steps, dtype=float)
    chain = 0.4 * steps ** 1.4
    ax.plot(steps, ideal, "o-", color=TEAL, lw=2, label="理想（每段都完美）")
    ax.plot(steps, chain, "s-", color=WAVE, lw=2, label="链式连乘（误差越积越大）")
    ax.set_xlabel("离开参照帧的跳数（第几张邻居）")
    ax.set_ylabel("对齐偏差（示意）")
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, color="#e6dccb")
    ax.text(
        0.98,
        0.08,
        "第 12 篇 Bundle Adjustment 就是想把这些小错一起调顺",
        transform=ax.transAxes,
        ha="right",
        fontsize=11,
        color=MUTED,
    )
    save_fig(fig, "ch08_error_accum.png")


def plot_fail_cases():
    fig, ax = plt.subplots(figsize=(9.4, 4.2), layout="constrained")
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.set_title("拼接失败时先查这几步")

    rows = [
        (5.0, "症状", "先查哪里", "常见对策"),
        (4.0, "特征点极少", "纹理是否太平？模糊？", "换有纹理的区域；降模糊；加特征数"),
        (3.0, "匹配乱成一团", "Lowe 比率是否太松？", "把比率调小（更严）；看连线是否交叉"),
        (2.0, "H 估出但扭曲炸裂", "内点是否太少？重叠太少？", "加大重叠；查 RANSAC 阈值；排除运动物体"),
        (1.0, "叠图有缝/双影", "H 准但融合差？曝光跳？", "第 10–11 篇：接缝与曝光；本篇先保证几何对齐"),
    ]
    for y, a, b, c in rows:
        weight = 700 if y == 5.0 else 400
        ax.text(0.3, y, a, fontsize=12, fontweight=weight, color=WAVE if y == 5.0 else INK)
        ax.text(2.6, y, b, fontsize=12, fontweight=weight)
        ax.text(6.3, y, c, fontsize=12, fontweight=weight, color=MUTED if y != 5.0 else INK)
    ax.axhline(4.55, color="#e3d9c8", lw=1)
    save_fig(fig, "ch08_fail_cases.png")


def plot_steps_board(left, right, result, info):
    """一张「五步进度板」，方便课文对照。"""
    fig, axes = plt.subplots(2, 3, figsize=(11.0, 6.2), layout="constrained")
    axes = axes.ravel()

    axes[0].imshow(left[:, :, ::-1])
    axes[0].set_title(f"① 检测　左 {info['n_kp1']} / 右 {info['n_kp2']} 点")
    # 画一些关键点
    for kp in list(info["kp1"])[:60]:
        axes[0].plot(kp.pt[0], kp.pt[1], ".", color=WAVE, ms=3)
    axes[0].axis("off")

    # 匹配示意：只画左图上的点
    axes[1].imshow(left[:, :, ::-1])
    axes[1].set_title(f"② 匹配　通过比率 {info['n_good']} 对")
    for m in info["good"][:50]:
        pt = info["kp1"][m.queryIdx].pt
        axes[1].plot(pt[0], pt[1], "o", color=TEAL, ms=3, fillstyle="none")
    axes[1].axis("off")

    axes[2].axis("off")
    axes[2].set_title("③ RANSAC 估 H")
    if info["H"] is not None:
        lines = ["  ".join(f"{info['H'][r, c]:8.3f}" for c in range(3)) for r in range(3)]
        axes[2].text(0.05, 0.55, "\n".join(lines), family="monospace", fontsize=11, transform=axes[2].transAxes)
        axes[2].text(
            0.05,
            0.18,
            f"内点 {info['n_inliers']} / 匹配 {info['n_good']}",
            fontsize=12,
            color=MUTED,
            transform=axes[2].transAxes,
        )

    cw, ch = info["canvas"]
    axes[3].set_title(f"④ 画布　{cw}×{ch}")
    axes[3].imshow(info["canvas_a"][:, :, ::-1])
    axes[3].axis("off")

    axes[4].imshow(info["warped_b"][:, :, ::-1])
    axes[4].set_title("⑤a 右图 warp")
    axes[4].axis("off")

    axes[5].imshow(result[:, :, ::-1])
    axes[5].set_title("⑤b 叠图完成")
    axes[5].axis("off")

    save_fig(fig, "ch08_steps.png")


def plot_mine(left, right):
    result, info = stitch_pair(
        left,
        right,
        n_features=N_FEATURES,
        ratio=LOWE_RATIO,
        ransac_thresh=RANSAC_THRESH,
        alpha=BLEND_ALPHA,
    )
    if result is None:
        fig, ax = plt.subplots(figsize=(7, 3), layout="constrained")
        ax.axis("off")
        ax.set_title("你的参数下未能估出 H（匹配/内点不足）")
        ax.text(0.5, 0.5, "试试减小 SHIFT_X 或增大 N_FEATURES", ha="center", transform=ax.transAxes)
        save_fig(fig, "ch08_my_result.png")
        return

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8), layout="constrained")
    axes[0].imshow(cv2.hconcat([left, right])[:, :, ::-1])
    axes[0].set_title("输入左右图（你的参数）")
    axes[0].axis("off")
    axes[1].imshow(result[:, :, ::-1])
    axes[1].set_title(
        f"拼接结果　内点 {info['n_inliers']}/{info['n_good']}　α={BLEND_ALPHA}"
    )
    axes[1].axis("off")
    save_fig(fig, "ch08_my_result.png")
    save_bgr("ch08_my_panorama.png", result)


# ---------------------------------------------------------------------------
# 自检
# ---------------------------------------------------------------------------


def self_check():
    scene = make_scene()
    left, right = crop_pair(scene, shift_x=120, angle_deg=1.5)
    assert left.shape == right.shape and left.ndim == 3

    kp1, kp2, good, _ = detect_and_match(left, right, n_features=800, ratio=0.75)
    assert len(kp1) > 20 and len(kp2) > 20, "特征点过少"
    assert len(good) >= 10, f"匹配过少：{len(good)}"

    H, mask = estimate_homography(kp1, kp2, good, ransac_thresh=5.0)
    assert H is not None and H.shape == (3, 3), "未能估出 H"
    n_in = int(mask.sum())
    assert n_in >= 8, f"内点过少：{n_in}"

    cw, ch, T, _ = compute_canvas(left, right, H)
    assert cw > left.shape[1] and ch >= left.shape[0] - 5
    assert T[0, 2] >= -1e-6 and T[1, 2] >= -1e-6

    result, canvas_a, warped_b = warp_and_blend(left, right, H, cw, ch, T, alpha=0.5)
    assert result.shape[:2] == (ch, cw)
    # 重叠区不应全黑
    assert result.sum() > 0 and canvas_a.sum() > 0 and warped_b.sum() > 0

    print("自检通过：合成重叠图、匹配、RANSAC、画布与叠图均可运行。")


def main():
    use_style()
    self_check()

    plot_pipeline()
    plot_multi_ref()
    plot_error_accum()
    plot_fail_cases()

    scene = make_scene()
    # 课文对照图用固定参数，保证稳定
    left, right = crop_pair(scene, shift_x=140, angle_deg=2.5)
    plot_pair(left, right)

    result, info = stitch_pair(
        left, right, n_features=1200, ratio=0.75, ransac_thresh=4.0, alpha=0.55
    )
    if result is None:
        raise SystemExit("课文示例拼接失败，请检查 OpenCV / 合成图。")

    plot_matches(left, right, info["kp1"], info["kp2"], info["good"], name="ch08_matches.png", title="比率检验后的匹配")
    plot_matches(
        left,
        right,
        info["kp1"],
        info["kp2"],
        info["good"],
        mask=info["mask"],
        name="ch08_inliers.png",
        title="RANSAC 内点",
    )
    plot_canvas_idea(left, info)
    plot_result(result, info["canvas_a"], info["warped_b"])
    plot_steps_board(left, right, result, info)

    # 你的参数：重新生成一对图再拼
    left_m, right_m = crop_pair(scene, shift_x=SHIFT_X, angle_deg=SHIFT_ANGLE_DEG)
    plot_mine(left_m, right_m)

    print("课文对照图已写入 assets/figures/")
    print("你的图：ch08_my_result.png / assets/images/ch08_my_panorama.png")


if __name__ == "__main__":
    main()
