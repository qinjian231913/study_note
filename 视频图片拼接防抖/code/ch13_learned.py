"""第 13 篇：深度学习特征 —— SuperPoint 与 LoFTR（概念 + 合成示意）。

在「视频图片拼接防抖」文件夹里运行：

    MPLCONFIGDIR=/tmp/mpl-cache .venv/bin/python code/ch13_learned.py

本篇**不依赖** torch / 预训练权重。用合成图示意：
  - 可学习检测+描述（SuperPoint 管线）
  - 无检测器的半稠密匹配（LoFTR 管线）
  - 与 ORB / SIFT 的选型对照

顶部「改这里」只影响 ch13_my_*.png。
课文对照图每次按固定参数重画。
"""

from pathlib import Path

SHOW_WINDOW = False

import matplotlib

if not SHOW_WINDOW:
    matplotlib.use("Agg")

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from plot_style import INK, MUTED, TEAL, WAVE, use_style

# ===== 改这里 =====
# 只影响 ch13_my_texture.png 与 ch13_my_matches.png
TEXTURE_LEVEL = 0.35  # 0=几乎无纹理（白墙感），1=纹理很丰富
SHIFT_X = 28  # 右图相对左图的水平位移（像素）
SHIFT_Y = 8
METHOD = "loftr"  # "orb" | "superpoint" | "loftr"
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


# ---------------------------------------------------------------------------
# 合成场景：可调纹理强度
# ---------------------------------------------------------------------------

def make_textured_pair(texture=0.5, shift_x=28, shift_y=8, w=320, h=240):
    """造一对有重叠的「照片」。texture∈[0,1] 控制角点/线条多少。"""
    texture = float(np.clip(texture, 0.0, 1.0))
    big_w, big_h = w + abs(shift_x) + 40, h + abs(shift_y) + 40
    scene = np.full((big_h, big_w, 3), 238, dtype=np.uint8)

    # 底色微噪：弱纹理时几乎看不见
    rng = np.random.default_rng(7)
    noise = rng.normal(0, 6 + 40 * texture, size=scene.shape).astype(np.float32)
    scene = np.clip(scene.astype(np.float32) + noise * texture, 0, 255).astype(np.uint8)

    # 结构线条：纹理越高越密、对比越强
    n_lines = int(2 + 18 * texture)
    contrast = int(30 + 140 * texture)
    for i in range(n_lines):
        x0 = 10 + i * (big_w // max(n_lines, 1))
        cv2.line(scene, (x0, 8), (x0 + int(20 * texture), big_h - 8), (contrast, contrast, contrast), 1)

    n_boxes = int(1 + 10 * texture)
    for i in range(n_boxes):
        x = 20 + (i * 47) % (big_w - 60)
        y = 20 + (i * 31) % (big_h - 60)
        bw, bh = 18 + (i % 5) * 4, 14 + (i % 4) * 5
        shade = int(70 + 80 * texture)
        cv2.rectangle(scene, (x, y), (x + bw, y + bh), (shade, shade - 10, shade + 15), -1)
        cv2.rectangle(scene, (x, y), (x + bw, y + bh), (40, 40, 40), 1)

    # 几个稳定「地标」角：即使弱纹理也保留极少量，方便示意
    landmarks = [(50, 60), (140, 90), (220, 70), (90, 160), (200, 150)]
    for lx, ly in landmarks:
        size = 6 if texture < 0.25 else 10
        cv2.drawMarker(scene, (lx, ly), WAVE_BGR, markerType=cv2.MARKER_CROSS, markerSize=size, thickness=2)

    left = scene[10 : 10 + h, 10 : 10 + w].copy()
    right = scene[10 + shift_y : 10 + shift_y + h, 10 + shift_x : 10 + shift_x + w].copy()
    return left, right, scene


def detect_orb_like(gray, max_corners=60, quality=0.01):
    """用 Shi-Tomasi 近似「手写检测器」：弱纹理时点数会骤降。"""
    pts = cv2.goodFeaturesToTrack(
        gray, maxCorners=max_corners, qualityLevel=quality, minDistance=8, blockSize=5
    )
    if pts is None:
        return np.zeros((0, 2), dtype=np.float32)
    return pts.reshape(-1, 2).astype(np.float32)


def simulate_superpoint_points(gray, texture, max_corners=60):
    """示意 SuperPoint：在角点基础上，弱纹理时仍多保留一些「学出来的」稳定点。"""
    base = detect_orb_like(gray, max_corners=max_corners, quality=0.008)
    h, w = gray.shape
    rng = np.random.default_rng(21)
    # 学出来的点：分布更均匀，弱纹理时也不会掉光
    n_extra = int(8 + 25 * max(0.15, texture))
    xs = rng.uniform(12, w - 12, size=n_extra)
    ys = rng.uniform(12, h - 12, size=n_extra)
    extras = np.stack([xs, ys], axis=1).astype(np.float32)
    if len(base) == 0:
        return extras
    return np.vstack([base, extras])[:max_corners]


def simulate_loftr_matches(h, w, shift_x, shift_y, stride=24, conf_floor=0.35):
    """示意 LoFTR：在规则网格上给出半稠密对应（不依赖检测器）。"""
    pts_a = []
    pts_b = []
    confs = []
    for y in range(stride // 2, h, stride):
        for x in range(stride // 2, w, stride):
            # 置信度：越靠边越低（像真实注意力边沿更糊）
            margin = min(x, y, w - 1 - x, h - 1 - y) / (0.5 * min(w, h))
            conf = float(np.clip(0.45 + 0.5 * margin, 0, 0.98))
            if conf < conf_floor:
                continue
            bx = x - shift_x
            by = y - shift_y
            if 0 <= bx < w and 0 <= by < h:
                pts_a.append((x, y))
                pts_b.append((bx, by))
                confs.append(conf)
    return (
        np.array(pts_a, dtype=np.float32),
        np.array(pts_b, dtype=np.float32),
        np.array(confs, dtype=np.float32),
    )


def match_by_known_shift(pts_a, pts_b, shift_x, shift_y, tol=6.0):
    """用已知位移做「理想匹配」，模拟描述子距离最近邻（仅供示意）。"""
    if len(pts_a) == 0 or len(pts_b) == 0:
        return []
    pairs = []
    used = set()
    for i, (x, y) in enumerate(pts_a):
        expect = np.array([x - shift_x, y - shift_y], dtype=np.float32)
        d = np.linalg.norm(pts_b - expect[None, :], axis=1)
        j = int(np.argmin(d))
        if j in used:
            continue
        if d[j] <= tol:
            pairs.append((i, j))
            used.add(j)
    return pairs


def draw_side_matches(left, right, pts_a, pts_b, pairs, title_a="A", title_b="B"):
    """并排两图并画匹配连线。"""
    gap = 16
    h = max(left.shape[0], right.shape[0])
    canvas = np.full((h, left.shape[1] + gap + right.shape[1], 3), 251, dtype=np.uint8)
    canvas[: left.shape[0], : left.shape[1]] = left
    xoff = left.shape[1] + gap
    canvas[: right.shape[0], xoff : xoff + right.shape[1]] = right
    cv2.putText(canvas, title_a, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, WAVE_BGR, 2, cv2.LINE_AA)
    cv2.putText(canvas, title_b, (xoff + 10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, TEAL_BGR, 2, cv2.LINE_AA)

    for i, j in pairs:
        xa, ya = pts_a[i]
        xb, yb = pts_b[j]
        p1 = (int(xa), int(ya))
        p2 = (int(xb) + xoff, int(yb))
        cv2.circle(canvas, p1, 3, WAVE_BGR, -1, cv2.LINE_AA)
        cv2.circle(canvas, p2, 3, TEAL_BGR, -1, cv2.LINE_AA)
        cv2.line(canvas, p1, p2, (120, 110, 100), 1, cv2.LINE_AA)
    return canvas


# ---------------------------------------------------------------------------
# 课文对照图
# ---------------------------------------------------------------------------

def plot_three_pipelines():
    """经典 / SuperPoint / LoFTR 三条管线对照。"""
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 4.2), layout="constrained")
    titles = [
        "经典：检测 → 描述 → 匹配",
        "SuperPoint：一起学检测+描述",
        "LoFTR：跳过检测，直接匹配",
    ]
    steps_list = [
        ["角点/斑点检测", "手写描述子\n(ORB/SIFT)", "最近邻匹配", "RANSAC 估 H"],
        ["共享 CNN 骨干", "检测头 + 描述头", "仍是稀疏点匹配", "RANSAC 估 H"],
        ["CNN 特征图", "粗匹配(Transformer)", "细匹配半稠密", "可直接喂几何"],
    ]
    colors = ["#f3e6d8", "#fde8df", "#e5f3f1"]

    for ax, title, steps, face in zip(axes, titles, steps_list, colors):
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 12)
        ax.axis("off")
        ax.set_title(title, fontsize=12)
        ax.add_patch(FancyBboxPatch((0.4, 0.4), 9.2, 11.2, boxstyle="round,pad=0.2",
                                   facecolor=face, edgecolor=INK, lw=1.2))
        for i, text in enumerate(steps):
            y = 9.6 - i * 2.5
            ax.add_patch(FancyBboxPatch((1.2, y), 7.6, 1.8, boxstyle="round,pad=0.15",
                                       facecolor="#fffdf8", edgecolor=WAVE if i == 0 else TEAL, lw=1.3))
            ax.text(5, y + 0.9, text, ha="center", va="center", fontsize=11)
            if i < len(steps) - 1:
                ax.annotate("", xy=(5, y - 0.15), xytext=(5, y - 0.55),
                            arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.4))
    save_fig(fig, "ch13_pipelines.png")


def plot_superpoint_schematic():
    fig, ax = plt.subplots(figsize=(9.6, 4.0), layout="constrained")
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 6)
    ax.axis("off")
    ax.set_title("SuperPoint 直觉：一张图进，出来「点在哪」+「每个点的指纹」")

    boxes = [
        (0.4, 2.0, 2.6, 2.2, "输入图像", WAVE),
        (3.6, 2.0, 2.8, 2.2, "共享编码器\n(CNN)", TEAL),
        (7.2, 3.4, 2.8, 1.8, "检测头\n→ 关键图", WAVE),
        (7.2, 1.0, 2.8, 1.8, "描述头\n→ 描述子", TEAL),
        (11.0, 3.4, 2.6, 1.8, "稀疏关键点", WAVE),
        (11.0, 1.0, 2.6, 1.8, "每点一条向量", TEAL),
    ]
    for x, y, w, h, text, color in boxes:
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.12",
                                   facecolor="#fffdf8", edgecolor=color, lw=1.5))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=11)

    arrows = [
        ((3.0, 3.1), (3.6, 3.1)),
        ((6.4, 3.5), (7.2, 4.2)),
        ((6.4, 2.7), (7.2, 1.9)),
        ((10.0, 4.3), (11.0, 4.3)),
        ((10.0, 1.9), (11.0, 1.9)),
    ]
    for (x0, y0), (x1, y1) in arrows:
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="->", mutation_scale=12,
                                    color=MUTED, lw=1.4))

    ax.text(7, 0.35, "训练时：用大量「同一场景、不同视角」的图对，逼网络在同一物理点反复点出、指纹相似",
            ha="center", fontsize=10, color=MUTED)
    save_fig(fig, "ch13_superpoint.png")


def plot_loftr_schematic():
    fig, ax = plt.subplots(figsize=(9.8, 4.2), layout="constrained")
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 6.2)
    ax.axis("off")
    ax.set_title("LoFTR 直觉：不问「先找哪些点」，问「整张特征图哪里对哪里」")

    stages = [
        (0.3, 1.6, 2.4, 3.0, "图 A / 图 B", "各抽一张\n粗特征图"),
        (3.3, 1.6, 3.0, 3.0, "Transformer\n粗匹配", "全局看上下文\n弱纹理也能对上"),
        (7.0, 1.6, 3.0, 3.0, "细匹配", "在局部窗口\n把像素位置抠准"),
        (10.7, 1.6, 2.8, 3.0, "半稠密对应", "很多可靠点对\n不是稀疏十几个"),
    ]
    for i, (x, y, w, h, title, body) in enumerate(stages):
        color = WAVE if i % 2 == 0 else TEAL
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15",
                                   facecolor="#fffdf8", edgecolor=color, lw=1.5))
        ax.text(x + w / 2, y + h - 0.7, title, ha="center", va="center", fontsize=12, fontweight="bold")
        ax.text(x + w / 2, y + 1.1, body, ha="center", va="center", fontsize=10, color=MUTED)
        if i < len(stages) - 1:
            ax.annotate("", xy=(x + w + 0.15, 3.1), xytext=(x + w + 0.55, 3.1),
                        arrowprops=dict(arrowstyle="<-", color=MUTED, lw=1.4))

    ax.text(7, 0.55, "关键：没有「检测器失败 → 后面全完」这条单点故障链",
            ha="center", fontsize=11, color=INK)
    save_fig(fig, "ch13_loftr.png")


def plot_compare_table():
    fig, ax = plt.subplots(figsize=(10.4, 3.8), layout="constrained")
    ax.axis("off")
    ax.set_title("ORB / SIFT / SuperPoint / LoFTR 选型速查（工程直觉，非绝对数值）")
    cols = ["", "ORB", "SIFT", "SuperPoint", "LoFTR"]
    rows = [
        ["典型速度", "CPU 很快", "CPU 中等", "常需 GPU", "更吃 GPU"],
        ["输出形态", "稀疏点", "稀疏点", "稀疏点", "半稠密点对"],
        ["弱纹理/白墙", "很容易挂", "仍偏弱", "好一些", "明显更稳"],
        ["是否必须学权重", "否", "否", "是", "是"],
        ["拼接入门优先？", "是（先练通）", "质量向备选", "有 GPU 再上", "弱纹理救场"],
    ]
    table = ax.table(cellText=rows, colLabels=cols, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10.5)
    table.scale(1.12, 1.55)
    for (r, c), cell in table.get_celld().items():
        if r == 0:
            cell.set_facecolor("#f3e6d8")
            cell.set_text_props(fontweight="bold")
        elif c == 0:
            cell.set_facecolor("#e5f3f1")
    save_fig(fig, "ch13_compare_table.png")


def plot_selection_flow():
    fig, ax = plt.subplots(figsize=(9.6, 5.2), layout="constrained")
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.set_title("经典优先还是学习优先？一条决策链")

    nodes = [
        (4.0, 8.4, 4.0, 1.2, "场景有没有明显角点/纹理？", "#f3e6d8"),
        (0.6, 5.8, 3.6, 1.2, "有 · 要实时/CPU", "#e5f3f1"),
        (4.2, 5.8, 3.6, 1.2, "有 · 要更稳质量", "#e5f3f1"),
        (7.8, 5.8, 3.6, 1.2, "几乎没有（白墙/天空）", "#fde8df"),
        (0.6, 3.4, 3.6, 1.1, "→ ORB（第 6 篇）", "#fffdf8"),
        (4.2, 3.4, 3.6, 1.1, "→ SIFT 或 SuperPoint", "#fffdf8"),
        (7.8, 3.4, 3.6, 1.1, "→ LoFTR 类匹配", "#fffdf8"),
        (2.0, 1.0, 8.0, 1.2, "无论哪种：后面仍要 RANSAC / 单应 / 融合；学习特征只换「找对应」这一环", "#f7f1e8"),
    ]
    for x, y, w, h, text, face in nodes:
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.12",
                                   facecolor=face, edgecolor=INK, lw=1.2))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=11)

    for (x0, y0), (x1, y1) in [
        ((6.0, 8.4), (2.4, 7.0)),
        ((6.0, 8.4), (6.0, 7.0)),
        ((6.0, 8.4), (9.6, 7.0)),
        ((2.4, 5.8), (2.4, 4.5)),
        ((6.0, 5.8), (6.0, 4.5)),
        ((9.6, 5.8), (9.6, 4.5)),
    ]:
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.2))
    save_fig(fig, "ch13_selection.png")


def plot_weak_texture_story():
    """同一弱纹理场景：稀疏检测几乎没点，半稠密仍有许多对应。"""
    left, right, _ = make_textured_pair(texture=0.12, shift_x=26, shift_y=6)
    g_a = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
    g_b = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
    orb_a = detect_orb_like(g_a, max_corners=50, quality=0.02)
    orb_b = detect_orb_like(g_b, max_corners=50, quality=0.02)
    pairs_orb = match_by_known_shift(orb_a, orb_b, 26, 6, tol=5.0)
    canvas_orb = draw_side_matches(left, right, orb_a, orb_b, pairs_orb, "检测器", "检测器")

    la, lb, _ = simulate_loftr_matches(left.shape[0], left.shape[1], 26, 6, stride=28, conf_floor=0.4)
    pairs_l = [(i, i) for i in range(len(la))]
    canvas_l = draw_side_matches(left, right, la, lb, pairs_l, "LoFTR示意", "LoFTR示意")

    fig, axes = plt.subplots(2, 1, figsize=(9.8, 7.0), layout="constrained")
    axes[0].imshow(canvas_orb[:, :, ::-1])
    axes[0].set_title(f"弱纹理 + 稀疏检测：点很少，匹配约 {len(pairs_orb)} 对（容易估不出稳 H）")
    axes[0].axis("off")
    axes[1].imshow(canvas_l[:, :, ::-1])
    axes[1].set_title(f"同一对图 + 半稠密示意：约 {len(pairs_l)} 对网格对应（后面几何更有料）")
    axes[1].axis("off")
    save_fig(fig, "ch13_weak_texture.png")
    save_bgr("ch13_weak_orb_panel.png", canvas_orb)
    save_bgr("ch13_weak_loftr_panel.png", canvas_l)


def plot_method_gallery():
    """三种方法在中等纹理下的示意匹配。"""
    left, right, _ = make_textured_pair(texture=0.45, shift_x=30, shift_y=10)
    g_a = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
    g_b = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)

    orb_a = detect_orb_like(g_a)
    orb_b = detect_orb_like(g_b)
    pairs_o = match_by_known_shift(orb_a, orb_b, 30, 10)
    c0 = draw_side_matches(left, right, orb_a, orb_b, pairs_o, "ORB-like", "ORB-like")

    sp_a = simulate_superpoint_points(g_a, 0.45)
    sp_b = simulate_superpoint_points(g_b, 0.45)
    pairs_s = match_by_known_shift(sp_a, sp_b, 30, 10, tol=7.0)
    c1 = draw_side_matches(left, right, sp_a, sp_b, pairs_s, "SuperPoint示意", "SuperPoint示意")

    la, lb, _ = simulate_loftr_matches(left.shape[0], left.shape[1], 30, 10, stride=26)
    pairs_l = [(i, i) for i in range(len(la))]
    c2 = draw_side_matches(left, right, la, lb, pairs_l, "LoFTR示意", "LoFTR示意")

    fig, axes = plt.subplots(3, 1, figsize=(9.6, 9.2), layout="constrained")
    for ax, img, title in zip(
        axes,
        [c0, c1, c2],
        [
            f"稀疏手写检测：匹配 {len(pairs_o)} 对",
            f"SuperPoint 示意（检测+描述一起学）：匹配 {len(pairs_s)} 对",
            f"LoFTR 示意（半稠密）：匹配 {len(pairs_l)} 对",
        ],
    ):
        ax.imshow(img[:, :, ::-1])
        ax.set_title(title)
        ax.axis("off")
    save_fig(fig, "ch13_method_gallery.png")


def plot_pseudocode_card():
    fig, ax = plt.subplots(figsize=(9.4, 5.0), layout="constrained")
    ax.axis("off")
    ax.set_title("伪代码对照（本机无权重时也能读懂在干什么）")
    ax.text(0.03, 0.92, "经典 / SuperPoint（稀疏路线）", transform=ax.transAxes,
            fontsize=12, fontweight="bold", color=WAVE)
    sparse = (
        "k1, d1 = detect_and_describe(imgA)\n"
        "k2, d2 = detect_and_describe(imgB)\n"
        "matches = nearest_neighbor(d1, d2)\n"
        "H = findHomography(k1, k2, RANSAC)"
    )
    ax.text(0.03, 0.86, sparse, transform=ax.transAxes, va="top", family="monospace", fontsize=11,
            bbox=dict(boxstyle="round,pad=0.45", facecolor="#fffdf8", edgecolor=WAVE, lw=1.2))
    ax.text(0.03, 0.48, "LoFTR（检测器无关 / 半稠密）", transform=ax.transAxes,
            fontsize=12, fontweight="bold", color=TEAL)
    dense = (
        "fA, fB = backbone(imgA), backbone(imgB)\n"
        "coarse = transformer_match(fA, fB)\n"
        "fine   = refine(coarse)\n"
        "H = findHomography(fine.ptsA, fine.ptsB, RANSAC)"
    )
    ax.text(0.03, 0.42, dense, transform=ax.transAxes, va="top", family="monospace", fontsize=11,
            bbox=dict(boxstyle="round,pad=0.45", facecolor="#fffdf8", edgecolor=TEAL, lw=1.2))
    ax.text(0.03, 0.06, "真正落地请用官方权重；本脚本只画直觉，不加载 torch 模型。",
            transform=ax.transAxes, fontsize=10, color=MUTED)
    save_fig(fig, "ch13_pseudocode.png")


def count_points_vs_texture():
    """曲线：纹理 ↓ → 检测点数 ↓；半稠密对数几乎不塌。"""
    textures = np.linspace(0.05, 1.0, 12)
    n_orb, n_sp, n_loftr = [], [], []
    for t in textures:
        left, right, _ = make_textured_pair(texture=float(t), shift_x=24, shift_y=6)
        g = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
        n_orb.append(len(detect_orb_like(g, max_corners=80)))
        n_sp.append(len(simulate_superpoint_points(g, float(t), max_corners=80)))
        la, _, _ = simulate_loftr_matches(left.shape[0], left.shape[1], 24, 6, stride=28)
        n_loftr.append(len(la))

    fig, ax = plt.subplots(figsize=(8.4, 4.2), layout="constrained")
    ax.plot(textures, n_orb, "o-", color=WAVE, lw=2, label="ORB-like 检测点数")
    ax.plot(textures, n_sp, "s-", color="#8a5a2b", lw=2, label="SuperPoint 示意点数")
    ax.plot(textures, n_loftr, "^-", color=TEAL, lw=2, label="LoFTR 示意匹配对数")
    ax.set_xlabel("纹理强度（合成参数）")
    ax.set_ylabel("数量")
    ax.set_title("弱纹理时：稀疏检测先垮，半稠密匹配更扛打（示意）")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    save_fig(fig, "ch13_texture_curve.png")


def plot_mine():
    left, right, _ = make_textured_pair(TEXTURE_LEVEL, SHIFT_X, SHIFT_Y)
    g_a = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
    g_b = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
    method = METHOD.lower().strip()

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6), layout="constrained")
    axes[0].imshow(left[:, :, ::-1])
    axes[0].set_title(f"左图 · 纹理={TEXTURE_LEVEL:.2f}")
    axes[0].axis("off")
    axes[1].imshow(right[:, :, ::-1])
    axes[1].set_title(f"右图 · 位移=({SHIFT_X},{SHIFT_Y})")
    axes[1].axis("off")
    save_fig(fig, "ch13_my_texture.png")

    if method == "orb":
        pa, pb = detect_orb_like(g_a), detect_orb_like(g_b)
        pairs = match_by_known_shift(pa, pb, SHIFT_X, SHIFT_Y)
        label = "ORB-like"
    elif method == "superpoint":
        pa = simulate_superpoint_points(g_a, TEXTURE_LEVEL)
        pb = simulate_superpoint_points(g_b, TEXTURE_LEVEL)
        pairs = match_by_known_shift(pa, pb, SHIFT_X, SHIFT_Y, tol=7.0)
        label = "SuperPoint示意"
    else:
        pa, pb, _ = simulate_loftr_matches(left.shape[0], left.shape[1], SHIFT_X, SHIFT_Y, stride=24)
        pairs = [(i, i) for i in range(len(pa))]
        label = "LoFTR示意"
        method = "loftr"

    canvas = draw_side_matches(left, right, pa, pb, pairs, label, label)
    fig2, ax2 = plt.subplots(figsize=(9.6, 3.8), layout="constrained")
    ax2.imshow(canvas[:, :, ::-1])
    ax2.set_title(f"你的匹配示意 · METHOD={method} · 对数={len(pairs)}")
    ax2.axis("off")
    save_fig(fig2, "ch13_my_matches.png")
    save_bgr("ch13_my_matches_bgr.png", canvas)


def self_check():
    # 1) 弱纹理时 ORB-like 点数应明显少于高纹理
    low_l, _, _ = make_textured_pair(0.08, 20, 5)
    high_l, _, _ = make_textured_pair(0.9, 20, 5)
    n_low = len(detect_orb_like(cv2.cvtColor(low_l, cv2.COLOR_BGR2GRAY)))
    n_high = len(detect_orb_like(cv2.cvtColor(high_l, cv2.COLOR_BGR2GRAY)))
    assert n_high > n_low, (n_low, n_high)

    # 2) LoFTR 示意在弱纹理下仍应给出足够多的点对
    la, lb, conf = simulate_loftr_matches(240, 320, 20, 5, stride=32)
    assert len(la) == len(lb) == len(conf) and len(la) >= 20

    # 3) 已知位移匹配：把 B 点按位移搬回去应能对上
    pts_a = np.array([[40.0, 50.0], [80.0, 90.0]], dtype=np.float32)
    pts_b = pts_a - np.array([20.0, 5.0], dtype=np.float32)
    pairs = match_by_known_shift(pts_a, pts_b, 20, 5, tol=1.0)
    assert len(pairs) == 2

    # 4) METHOD 合法取值
    assert METHOD.lower() in {"orb", "superpoint", "loftr"}

    # 5) 纹理参数裁剪后仍在 [0,1]
    assert 0.0 <= float(np.clip(TEXTURE_LEVEL, 0, 1)) <= 1.0

    print("自检通过：弱纹理点数、半稠密示意、位移匹配、METHOD 均正常。")


def main():
    use_style()
    self_check()
    plot_three_pipelines()
    plot_superpoint_schematic()
    plot_loftr_schematic()
    plot_compare_table()
    plot_selection_flow()
    plot_weak_texture_story()
    plot_method_gallery()
    plot_pseudocode_card()
    count_points_vs_texture()
    plot_mine()
    print("课文对照图已写入 assets/figures/ 与 assets/images/")
    print("你的图：ch13_my_texture.png、ch13_my_matches.png")


if __name__ == "__main__":
    main()
