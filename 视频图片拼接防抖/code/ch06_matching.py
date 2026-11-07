"""第 6 篇：描述子与特征匹配。

在「视频图片拼接防抖」文件夹里运行：

    MPLCONFIGDIR=/tmp/mpl-cache .venv/bin/python code/ch06_matching.py

图会保存到 assets/figures/ 与 assets/images/。
顶部「改这里」只影响 ch06_my_matches.png。
课文对照图每次都按固定参数重画，方便和网页正文对得上。

说明：本环境若有 cv2.SIFT_create 会画 SIFT 对照图；
若没有（旧版 opencv 需 opencv-contrib），脚本会跳过 SIFT 图并打印提示，其余以 ORB 为主。
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

WAVE_BGR = (40, 77, 194)  # #c24d28
TEAL_BGR = (102, 107, 14)  # #0e6b66
INK_BGR = (21, 25, 28)
GOOD_BGR = (60, 160, 90)
BAD_BGR = (40, 77, 194)

# ===== 改这里 =====
# 只影响 ch06_my_matches.png
MY_RATIO = 0.75          # Lowe 比率阈值：越小越严，留下的匹配越少但更干净
MY_MAX_DRAW = 60         # 最多画多少条连线（太多会糊成一团）
MY_NFEATURES = 800       # ORB 最多检测多少个点
MY_CROSS_CHECK = False   # True = 开启互检（交叉检验）；与比率二选一时可先关比率看互检
# ==================

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "assets" / "figures"
IMG = ROOT / "assets" / "images"

HAS_SIFT = hasattr(cv2, "SIFT_create")


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


def make_scene(width=720, height=400):
    """合成一张带独特纹理的「墙」：窗户、砖缝、招牌，方便特征点站稳。"""
    img = np.full((height, width, 3), 236, dtype=np.uint8)

    # 天空
    img[:120] = (228, 214, 196)

    # 墙砖纹理
    for y in range(120, height, 22):
        offset = 12 if ((y - 120) // 22) % 2 else 0
        for x in range(-offset, width, 36):
            cv2.rectangle(img, (x + 1, y + 1), (x + 32, y + 18), (210, 198, 178), -1)
            cv2.rectangle(img, (x + 1, y + 1), (x + 32, y + 18), (160, 145, 130), 1)

    # 三扇窗
    for x0 in (80, 300, 520):
        cv2.rectangle(img, (x0, 150), (x0 + 90, 250), (90, 140, 170), -1)
        cv2.rectangle(img, (x0, 150), (x0 + 90, 250), (40, 40, 40), 2)
        cv2.line(img, (x0 + 45, 150), (x0 + 45, 250), (40, 40, 40), 2)
        cv2.line(img, (x0, 200), (x0 + 90, 200), (40, 40, 40), 2)

    # 招牌：独特字符块，匹配时很好认
    cv2.rectangle(img, (200, 60), (480, 110), WAVE_BGR, -1)
    cv2.putText(img, "CAFE", (250, 98), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 3, cv2.LINE_AA)

    # 门
    cv2.rectangle(img, (330, 270), (400, 390), (70, 55, 45), -1)
    cv2.circle(img, (385, 330), 5, (220, 200, 120), -1)

    # 地面
    img[390:] = (180, 170, 155)
    cv2.line(img, (0, 390), (width, 390), (100, 90, 80), 2)

    # 左侧花坛：增加非重复纹理
    cv2.ellipse(img, (60, 360), (40, 18), 0, 0, 360, TEAL_BGR, -1)
    for i in range(5):
        cv2.circle(img, (40 + i * 12, 340 - (i % 3) * 6), 8, (40, 140, 80), -1)

    return img


def crop_pair(scene, shift_x=90, shift_y=12, crop_w=380, crop_h=300):
    """从同一场景裁两张有重叠的「照片」，右图再轻微旋转，模拟视角差。"""
    h, w = scene.shape[:2]
    x0, y0 = 40, 40
    left = scene[y0 : y0 + crop_h, x0 : x0 + crop_w].copy()

    # 右图裁切位置往右下偏一点
    x1 = x0 + shift_x
    y1 = y0 + shift_y
    right_raw = scene[y1 : y1 + crop_h, x1 : x1 + crop_w].copy()

    # 轻微旋转，逼出一点误匹配
    M = cv2.getRotationMatrix2D((crop_w / 2, crop_h / 2), 4.5, 1.0)
    right = cv2.warpAffine(right_raw, M, (crop_w, crop_h), borderValue=(245, 245, 245))
    return left, right


def draw_side_by_side(img1, img2, matches, kps1, kps2, max_draw=80, color_fn=None):
    """把两图横拼，按 matches 画连线。color_fn(m) -> BGR 或 None 用默认青。"""
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    h = max(h1, h2)
    canvas = np.full((h, w1 + w2, 3), 251, dtype=np.uint8)
    canvas[:h1, :w1] = img1
    canvas[:h2, w1 : w1 + w2] = img2

    draw_list = list(matches)[:max_draw]
    for m in draw_list:
        pt1 = tuple(np.round(kps1[m.queryIdx].pt).astype(int))
        pt2 = tuple(np.round(kps2[m.trainIdx].pt).astype(int))
        pt2 = (pt2[0] + w1, pt2[1])
        color = color_fn(m) if color_fn else TEAL_BGR
        cv2.line(canvas, pt1, pt2, color, 1, cv2.LINE_AA)
        cv2.circle(canvas, pt1, 3, color, -1, cv2.LINE_AA)
        cv2.circle(canvas, pt2, 3, color, -1, cv2.LINE_AA)
    return canvas


def orb_detect_compute(img, nfeatures=1200):
    orb = cv2.ORB_create(nfeatures=nfeatures, scaleFactor=1.2, nlevels=8)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kps, desc = orb.detectAndCompute(gray, None)
    return kps, desc


def sift_detect_compute(img, nfeatures=800):
    if not HAS_SIFT:
        return None, None
    sift = cv2.SIFT_create(nfeatures=nfeatures)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kps, desc = sift.detectAndCompute(gray, None)
    return kps, desc


def bf_knn_orb(desc1, desc2, k=2):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    return bf.knnMatch(desc1, desc2, k=k)


def bf_knn_sift(desc1, desc2, k=2):
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    return bf.knnMatch(desc1, desc2, k=k)


def ratio_filter(knn_matches, ratio=0.75):
    """Lowe 比率检验：只留 best.distance < ratio * second.distance 的对。"""
    good = []
    for pair in knn_matches:
        if len(pair) < 2:
            continue
        m, n = pair[0], pair[1]
        if m.distance < ratio * n.distance:
            good.append(m)
    return good


def cross_check_orb(desc1, desc2):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    return bf.match(desc1, desc2)


def plot_detect_vs_desc():
    fig, ax = plt.subplots(figsize=(9.2, 4.2), layout="constrained")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis("off")
    ax.set_title("检测给「位置」，描述子给「指纹」")

    # 左：检测
    box1 = FancyBboxPatch((0.4, 1.2), 4.0, 3.0, boxstyle="round,pad=0.15",
                          facecolor="#fde8df", edgecolor=INK, lw=1.4)
    ax.add_patch(box1)
    ax.text(2.4, 3.8, "① 特征检测", ha="center", fontsize=14, fontweight="bold", color=WAVE)
    ax.text(2.4, 3.2, "回答：哪里值得认？", ha="center", fontsize=12)
    ax.text(2.4, 2.5, "输出：关键点 (x, y)\n+ 可选尺度 / 方向", ha="center", fontsize=11)
    ax.text(2.4, 1.6, "类比：在人群里指出\n「那个人站在哪」", ha="center", fontsize=11, color=MUTED)

    # 右：描述
    box2 = FancyBboxPatch((5.6, 1.2), 4.0, 3.0, boxstyle="round,pad=0.15",
                          facecolor="#e5f3f1", edgecolor=INK, lw=1.4)
    ax.add_patch(box2)
    ax.text(7.6, 3.8, "② 描述子", ha="center", fontsize=14, fontweight="bold", color=TEAL)
    ax.text(7.6, 3.2, "回答：这里长什么样？", ha="center", fontsize=12)
    ax.text(7.6, 2.5, "输出：一串数字 / 比特\n（指纹向量）", ha="center", fontsize=11)
    ax.text(7.6, 1.6, "类比：记下他的\n身高、发型、衣服颜色", ha="center", fontsize=11, color=MUTED)

    ax.annotate("", xy=(5.5, 2.7), xytext=(4.5, 2.7),
                arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.8))
    ax.text(5.0, 3.0, "再算指纹", ha="center", fontsize=10, color=MUTED)

    save_fig(fig, "ch06_detect_vs_desc.png")


def plot_orb_sift_table():
    fig, ax = plt.subplots(figsize=(9.4, 4.0), layout="constrained")
    ax.axis("off")
    ax.set_title("ORB 与 SIFT：直觉对照（不是完整论文）")

    col_labels = ["维度", "ORB（快、二进制）", "SIFT（经典、更稳）"]
    rows = [
        ["描述子形态", "256 比特（32 字节）", "128 维浮点向量"],
        ["距离度量", "汉明距离（比比特）", "欧氏距离（L2）"],
        ["尺度 / 旋转", "有金字塔 + 方向，够用", "专门为尺度旋转设计，通常更稳"],
        ["速度", "很快，适合实时 / 手机", "更慢，精度场景更常见"],
        ["OpenCV", "cv2.ORB_create", "cv2.SIFT_create（新版主包；旧版在 contrib）"],
        ["本课默认", "主流程用 ORB 演示", "有则对照，无则跳过并注明"],
    ]

    table = ax.table(
        cellText=rows,
        colLabels=col_labels,
        loc="center",
        cellLoc="left",
        colColours=["#efe8dc", "#fde8df", "#e5f3f1"],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.05, 1.55)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#d8cfc0")
        if r == 0:
            cell.set_text_props(fontweight="bold")
    save_fig(fig, "ch06_orb_sift_table.png")


def plot_lowe_idea():
    fig, ax = plt.subplots(figsize=(9.0, 3.8), layout="constrained")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.axis("off")
    ax.set_title("Lowe 比率检验：为什么能压误匹配")

    ax.text(0.4, 3.3, "对左图某一点，在右图找「最近」与「次近」两个候选：", fontsize=12)
    ax.text(0.4, 2.5, "d₁ = 到最近邻的距离　　d₂ = 到次近邻的距离", fontsize=13)
    ax.text(0.4, 1.7, "若 d₁ ≪ d₂（例如 d₁ < 0.75 · d₂）→ 最近邻「明显赢」，更可能是真匹配", fontsize=12, color=TEAL)
    ax.text(0.4, 1.0, "若 d₁ ≈ d₂ → 两个候选差不多像 → 说明「其实认不准」，丢掉更安全", fontsize=12, color=WAVE)
    ax.text(0.4, 0.35, "白话：第一名要大幅领先第二名，才算认出来；模棱两可就弃权。", fontsize=11, color=MUTED)
    save_fig(fig, "ch06_lowe_idea.png")


def plot_keypoints_compare(left):
    """同一张图上标 ORB / SIFT 关键点，体会数量与分布差异。"""
    kps_orb, _ = orb_detect_compute(left, nfeatures=500)
    img_orb = cv2.drawKeypoints(left, kps_orb, None, color=WAVE_BGR, flags=0)

    fig, axes = plt.subplots(1, 2 if HAS_SIFT else 1, figsize=(10.0 if HAS_SIFT else 5.2, 4.2), layout="constrained")
    if not HAS_SIFT:
        axes = [axes]

    axes[0].imshow(img_orb[:, :, ::-1])
    axes[0].set_title(f"ORB 关键点（本例 {len(kps_orb)} 个）")
    axes[0].axis("off")

    if HAS_SIFT:
        kps_sift, _ = sift_detect_compute(left, nfeatures=500)
        img_sift = cv2.drawKeypoints(left, kps_sift, None, color=TEAL_BGR, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        axes[1].imshow(img_sift[:, :, ::-1])
        axes[1].set_title(f"SIFT 关键点（本例 {len(kps_sift)} 个，圆圈示意尺度）")
        axes[1].axis("off")
    else:
        print("提示：当前 OpenCV 无 SIFT_create，已跳过 SIFT 关键点对照。可装 opencv-contrib 或升级主包后再跑。")

    save_fig(fig, "ch06_keypoints_orb_sift.png")


def plot_all_vs_ratio(left, right):
    kps1, d1 = orb_detect_compute(left)
    kps2, d2 = orb_detect_compute(right)
    knn = bf_knn_orb(d1, d2)
    all_best = [p[0] for p in knn if len(p) >= 1]
    good = ratio_filter(knn, ratio=0.75)

    canvas_all = draw_side_by_side(left, right, all_best, kps1, kps2, max_draw=120)
    canvas_good = draw_side_by_side(left, right, good, kps1, kps2, max_draw=80)

    fig, axes = plt.subplots(2, 1, figsize=(10.2, 8.2), layout="constrained")
    axes[0].imshow(canvas_all[:, :, ::-1])
    axes[0].set_title(f"暴力匹配：每个点只连「最近邻」（画了 {min(120, len(all_best))} 条）——线很乱")
    axes[0].axis("off")
    axes[1].imshow(canvas_good[:, :, ::-1])
    axes[1].set_title(f"Lowe 比率 0.75 过滤后（{len(good)} 条中画 {min(80, len(good))} 条）——干净很多")
    axes[1].axis("off")
    save_fig(fig, "ch06_bf_vs_ratio.png")
    save_bgr("ch06_bf_all.png", canvas_all)
    save_bgr("ch06_ratio_clean.png", canvas_good)
    return kps1, kps2, knn, good, d1, d2


def plot_mismatches(left, right, kps1, kps2, knn):
    """专门挑出「比率没过」或「几何上明显离谱」的误匹配样例。"""
    bad = []
    for pair in knn:
        if len(pair) < 2:
            continue
        m, n = pair[0], pair[1]
        # 比率接近 1：典型「模棱两可」误匹配候选
        if m.distance >= 0.85 * n.distance:
            bad.append(m)

    # 再按距离从差到更差排序，取一批画出来
    bad = sorted(bad, key=lambda m: -m.distance)[:40]

    def color_fn(_m):
        return BAD_BGR

    canvas = draw_side_by_side(left, right, bad, kps1, kps2, max_draw=35, color_fn=color_fn)

    # 在画布上加注
    cv2.putText(canvas, "ORANGE = suspicious / wrong-ish matches", (12, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, BAD_BGR, 2, cv2.LINE_AA)

    fig, ax = plt.subplots(figsize=(10.2, 4.4), layout="constrained")
    ax.imshow(canvas[:, :, ::-1])
    ax.set_title("专门展示：比率接近 1 的「认不准」连线——交叉乱飞，就是误匹配长相")
    ax.axis("off")
    save_fig(fig, "ch06_mismatches.png")
    save_bgr("ch06_mismatches.png", canvas)


def plot_cross_check(left, right, kps1, kps2, desc1, desc2, good_ratio):
    """互检 vs 仅比率：两条路都能压误匹配，机制不同。"""
    cross = cross_check_orb(desc1, desc2)
    cross = sorted(cross, key=lambda m: m.distance)[:80]

    canvas_cross = draw_side_by_side(left, right, cross, kps1, kps2, max_draw=60)
    canvas_ratio = draw_side_by_side(left, right, good_ratio, kps1, kps2, max_draw=60)

    fig, axes = plt.subplots(2, 1, figsize=(10.2, 8.0), layout="constrained")
    axes[0].imshow(canvas_ratio[:, :, ::-1])
    axes[0].set_title(f"仅 Lowe 比率（0.75）：{len(good_ratio)} 对")
    axes[0].axis("off")
    axes[1].imshow(canvas_cross[:, :, ::-1])
    axes[1].set_title(f"仅互检 / 交叉检验（crossCheck=True）：{len(cross)} 对（图中画前 60）")
    axes[1].axis("off")
    save_fig(fig, "ch06_cross_check.png")


def plot_sift_matches_if_any(left, right):
    if not HAS_SIFT:
        print("提示：无 SIFT，跳过 ch06_sift_matches.png。新版 opencv-python 通常自带；旧环境可试 opencv-contrib-python。")
        return
    k1, d1 = sift_detect_compute(left)
    k2, d2 = sift_detect_compute(right)
    if d1 is None or d2 is None:
        return
    knn = bf_knn_sift(d1, d2)
    good = ratio_filter(knn, ratio=0.75)
    canvas = draw_side_by_side(left, right, good, k1, k2, max_draw=70)
    fig, ax = plt.subplots(figsize=(10.2, 4.4), layout="constrained")
    ax.imshow(canvas[:, :, ::-1])
    ax.set_title(f"SIFT + BFMatcher(L2) + 比率 0.75（本例 {len(good)} 对，画 {min(70, len(good))} 条）")
    ax.axis("off")
    save_fig(fig, "ch06_sift_matches.png")


def plot_my_matches(left, right):
    kps1, d1 = orb_detect_compute(left, nfeatures=MY_NFEATURES)
    kps2, d2 = orb_detect_compute(right, nfeatures=MY_NFEATURES)
    if MY_CROSS_CHECK:
        matches = cross_check_orb(d1, d2)
        matches = sorted(matches, key=lambda m: m.distance)
        title = f"我的参数：互检开启，画前 {MY_MAX_DRAW} / 共 {len(matches)}"
    else:
        knn = bf_knn_orb(d1, d2)
        matches = ratio_filter(knn, ratio=MY_RATIO)
        title = f"我的参数：比率={MY_RATIO}，共 {len(matches)} 对，画 {min(MY_MAX_DRAW, len(matches))} 条"
    canvas = draw_side_by_side(left, right, matches, kps1, kps2, max_draw=MY_MAX_DRAW)
    fig, ax = plt.subplots(figsize=(10.2, 4.4), layout="constrained")
    ax.imshow(canvas[:, :, ::-1])
    ax.set_title(title)
    ax.axis("off")
    save_fig(fig, "ch06_my_matches.png")
    save_bgr("ch06_my_matches.png", canvas)


def export_playground_json(left, right):
    """导出一组带 d1/d2 的匹配，供网页滑块演示（同时也打印摘要）。"""
    kps1, d1 = orb_detect_compute(left, nfeatures=600)
    kps2, d2 = orb_detect_compute(right, nfeatures=600)
    knn = bf_knn_orb(d1, d2)
    h1, w1 = left.shape[:2]
    records = []
    for pair in knn:
        if len(pair) < 2:
            continue
        m, n = pair[0], pair[1]
        p1 = kps1[m.queryIdx].pt
        p2 = kps2[m.trainIdx].pt
        records.append(
            {
                "x1": round(float(p1[0]), 1),
                "y1": round(float(p1[1]), 1),
                "x2": round(float(p2[0]), 1),
                "y2": round(float(p2[1]), 1),
                "d1": float(m.distance),
                "d2": float(n.distance),
            }
        )
    # 为网页轻量起见，按比率从好到差取前 120
    records.sort(key=lambda r: r["d1"] / max(r["d2"], 1e-6))
    records = records[:120]
    FIG.mkdir(parents=True, exist_ok=True)
    import json

    payload = {"w1": w1, "h1": h1, "w2": right.shape[1], "h2": right.shape[0], "matches": records}
    path = FIG / "ch06_playground_matches.json"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    print(f"已保存 {path}（{len(records)} 条，供交互参考）")

    # 网页滑块直接 <script src=...> 加载，避免 file:// 下 fetch JSON 失败
    js_path = FIG / "ch06_match_data.js"
    js_payload = dict(payload)
    js_payload["matches"] = records[:80]
    js_path.write_text(
        "var MATCH_DATA = " + json.dumps(js_payload, ensure_ascii=False, separators=(",", ":")) + ";\n",
        encoding="utf-8",
    )
    print(f"已保存 {js_path}")

    save_bgr("ch06_pair_left.png", left)
    save_bgr("ch06_pair_right.png", right)
    return payload


def self_check():
    """≥4 项自检：确保核心逻辑没写反。"""
    # 1) ORB 描述子是 32 字节
    img = np.random.randint(0, 255, (120, 160), dtype=np.uint8)
    orb = cv2.ORB_create(nfeatures=50)
    kps, desc = orb.detectAndCompute(img, None)
    assert desc is not None and desc.shape[1] == 32, "ORB 描述子应为 32 字节"

    # 2) 比率过滤：构造假 knn
    class M:
        def __init__(self, d, q=0, t=0):
            self.distance = d
            self.queryIdx = q
            self.trainIdx = t

    fake = [[M(10, 0, 0), M(20, 0, 1)], [M(15, 1, 2), M(16, 1, 3)]]
    kept = ratio_filter(fake, ratio=0.75)
    assert len(kept) == 1 and kept[0].distance == 10, "比率 0.75 应只留 10 < 0.75*20 那一对"
    kept_strict = ratio_filter(fake, ratio=0.5)
    assert len(kept_strict) == 0, "比率 0.5 时 10 不小于 0.5*20，应全丢"

    # 3) 汉明距离：相同描述子距离为 0
    a = np.zeros((1, 32), dtype=np.uint8)
    b = np.zeros((1, 32), dtype=np.uint8)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    ms = bf.match(a, b)
    assert len(ms) == 1 and ms[0].distance == 0, "相同二进制描述子汉明距离应为 0"

    # 4) 互检：单向最近邻未必互为最近
    # 用真实图跑一遍，互检数量应 ≤ 单向最近邻数量
    scene = make_scene()
    left, right = crop_pair(scene)
    k1, d1 = orb_detect_compute(left, nfeatures=400)
    k2, d2 = orb_detect_compute(right, nfeatures=400)
    knn = bf_knn_orb(d1, d2)
    one_way = len([p[0] for p in knn if p])
    cross = len(cross_check_orb(d1, d2))
    assert cross <= one_way, "互检匹配数不应超过单向最近邻数"

    # 5) SIFT 若存在，描述子应为 128 维
    if HAS_SIFT:
        s = cv2.SIFT_create(nfeatures=20)
        _, sd = s.detectAndCompute(img, None)
        if sd is not None:
            assert sd.shape[1] == 128, "SIFT 描述子应为 128 维"

    print("自检通过")


def main():
    use_style()
    print(f"OpenCV {cv2.__version__}；SIFT 可用：{HAS_SIFT}")

    plot_detect_vs_desc()
    plot_orb_sift_table()
    plot_lowe_idea()

    scene = make_scene()
    save_bgr("ch06_scene.png", scene)
    left, right = crop_pair(scene)
    save_bgr("ch06_pair_left.png", left)
    save_bgr("ch06_pair_right.png", right)

    plot_keypoints_compare(left)
    kps1, kps2, knn, good, d1, d2 = plot_all_vs_ratio(left, right)
    plot_mismatches(left, right, kps1, kps2, knn)
    plot_cross_check(left, right, kps1, kps2, d1, d2, good)
    plot_sift_matches_if_any(left, right)
    plot_my_matches(left, right)
    export_playground_json(left, right)

    self_check()
    print("第 6 篇配图全部完成。")


if __name__ == "__main__":
    main()
