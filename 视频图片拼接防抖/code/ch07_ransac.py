"""第 7 篇：RANSAC 与鲁棒估计单应性。

在「视频图片拼接防抖」文件夹里运行：

    MPLCONFIGDIR=/tmp/mpl-cache .venv/bin/python code/ch07_ransac.py

图保存到 assets/figures/。顶部「改这里」只影响 ch07_my_ransac.png。
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
# 只影响 ch07_my_ransac.png
OUTLIER_RATIO = 0.35   # 外点比例，0~0.7 较合适；太高时 RANSAC 更难
RANSAC_THRESH = 3.0    # 重投影误差阈值（像素）；越大内点越多、也可能混进外点
RANSAC_ITERS = 500     # 最大迭代次数；太小可能还没抽到好样本就停了
NOISE_PX = 1.2         # 内点高斯噪声标准差（像素）
# ==================

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "assets" / "figures"


def save_fig(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    path = FIG / name
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"已保存 {path}")


def true_homography():
    """一张温和的透视单应：轻微旋转 + 平移 + 一点透视。"""
    src = np.float32([[0, 0], [200, 0], [200, 150], [0, 150]])
    dst = np.float32([[18, 22], [210, 8], [228, 168], [12, 155]])
    return cv2.getPerspectiveTransform(src, dst)


def apply_h(H, pts):
    """pts: (N,2) → (N,2)。"""
    pts = np.asarray(pts, dtype=np.float64)
    ones = np.ones((len(pts), 1), dtype=np.float64)
    hom = np.hstack([pts, ones])
    mapped = (H @ hom.T).T
    w = mapped[:, 2:3]
    w = np.where(np.abs(w) < 1e-12, 1e-12, w)
    return mapped[:, :2] / w


def reproj_errors(H, src, dst):
    pred = apply_h(H, src)
    return np.linalg.norm(pred - dst, axis=1)


def make_correspondences(
    n_inliers=60,
    outlier_ratio=0.35,
    noise_px=1.2,
    seed=7,
):
    """合成一批带外点的对应点。返回 src, dst, is_inlier(bool)。"""
    rng = np.random.default_rng(seed)
    H = true_homography()

    n_out = int(round(n_inliers * outlier_ratio / max(1e-6, 1.0 - outlier_ratio)))
    # 内点：在图 A 均匀采样，经真 H 映射后再加噪声
    src_in = rng.uniform([10, 10], [190, 140], size=(n_inliers, 2))
    dst_in = apply_h(H, src_in) + rng.normal(0.0, noise_px, size=src_in.shape)

    # 外点：两边各自随机，不服从同一 H
    src_out = rng.uniform([0, 0], [200, 150], size=(n_out, 2))
    dst_out = rng.uniform([0, 0], [240, 180], size=(n_out, 2))

    src = np.vstack([src_in, src_out]).astype(np.float32)
    dst = np.vstack([dst_in, dst_out]).astype(np.float32)
    is_inlier = np.array([True] * n_inliers + [False] * n_out)

    # 打乱顺序，避免「前半段全是内点」的假象
    order = rng.permutation(len(src))
    return src[order], dst[order], is_inlier[order], H


def estimate_h_dlt(src4, dst4):
    """用 4 对点估 H（OpenCV）。失败返回 None。"""
    src4 = np.asarray(src4, dtype=np.float32).reshape(4, 2)
    dst4 = np.asarray(dst4, dtype=np.float32).reshape(4, 2)
    try:
        H = cv2.getPerspectiveTransform(src4, dst4)
    except cv2.error:
        return None
    if H is None or not np.isfinite(H).all():
        return None
    # 退化：行列式接近 0
    if abs(np.linalg.det(H)) < 1e-8:
        return None
    return H


def ransac_homography(src, dst, thresh=3.0, max_iters=500, seed=11):
    """手写一版 RANSAC（教学用），返回 best_H, best_mask, history。"""
    rng = np.random.default_rng(seed)
    n = len(src)
    best_H = None
    best_mask = np.zeros(n, dtype=bool)
    best_count = -1
    history = []  # (iter, inlier_count, is_new_best)

    for it in range(max_iters):
        idx = rng.choice(n, size=4, replace=False)
        H = estimate_h_dlt(src[idx], dst[idx])
        if H is None:
            history.append((it, 0, False))
            continue
        err = reproj_errors(H, src, dst)
        mask = err < thresh
        count = int(mask.sum())
        is_best = count > best_count
        if is_best:
            best_count = count
            best_H = H
            best_mask = mask
        history.append((it, count, is_best))

    # 用全部内点再精炼一次（OpenCV 的 findHomography 也会做类似事）
    if best_H is not None and best_mask.sum() >= 4:
        H_ref, mask_ref = cv2.findHomography(
            src[best_mask],
            dst[best_mask],
            method=0,  # 普通最小二乘
        )
        if H_ref is not None:
            best_H = H_ref
            err = reproj_errors(best_H, src, dst)
            best_mask = err < thresh

    return best_H, best_mask, history


def draw_two_panels(ax_a, ax_b, src, dst, mask=None, title_a="图 A", title_b="图 B"):
    """在左右两幅图上画对应点；mask True=内点(青)，False=外点(橙)。"""
    for ax, title in ((ax_a, title_a), (ax_b, title_b)):
        ax.set_facecolor("#fffdf8")
        ax.set_xlim(-10, 250)
        ax.set_ylim(190, -10)
        ax.set_aspect("equal")
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    if mask is None:
        ax_a.scatter(src[:, 0], src[:, 1], s=22, c=MUTED, zorder=3)
        ax_b.scatter(dst[:, 0], dst[:, 1], s=22, c=MUTED, zorder=3)
        return

    inl = mask.astype(bool)
    out = ~inl
    if out.any():
        ax_a.scatter(src[out, 0], src[out, 1], s=28, c=WAVE, marker="x", linewidths=1.4, zorder=3, label="外点")
        ax_b.scatter(dst[out, 0], dst[out, 1], s=28, c=WAVE, marker="x", linewidths=1.4, zorder=3)
    if inl.any():
        ax_a.scatter(src[inl, 0], src[inl, 1], s=26, c=TEAL, zorder=4, label="内点")
        ax_b.scatter(dst[inl, 0], dst[inl, 1], s=26, c=TEAL, zorder=4)
    ax_a.legend(loc="lower right", fontsize=9, frameon=False)


def plot_outlier_problem():
    """外点如何毁掉「全体最小二乘」估出的 H。"""
    src, dst, is_true, H_true = make_correspondences(
        n_inliers=50, outlier_ratio=0.4, noise_px=1.0, seed=3
    )

    # 全体点直接最小二乘（无 RANSAC）
    H_all, _ = cv2.findHomography(src, dst, method=0)
    # 只用真内点
    H_good, _ = cv2.findHomography(src[is_true], dst[is_true], method=0)

    # 在一组测试格点上看映射偏差
    grid = np.mgrid[20:181:40, 20:131:40].reshape(2, -1).T.astype(np.float64)
    true_map = apply_h(H_true, grid)
    all_map = apply_h(H_all, grid)
    good_map = apply_h(H_good, grid)

    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.8), layout="constrained")

    axes[0].set_title("匹配里混进外点")
    axes[0].set_xlim(-10, 250)
    axes[0].set_ylim(190, -10)
    axes[0].set_aspect("equal")
    # 用连线示意：内点短线、外点乱线——在同一坐标系叠画两边有点挤，改成只画图 B 上的目标点
    axes[0].scatter(dst[is_true, 0], dst[is_true, 1], s=22, c=TEAL, label="好匹配（内点）", zorder=3)
    axes[0].scatter(dst[~is_true, 0], dst[~is_true, 1], s=28, c=WAVE, marker="x", linewidths=1.3, label="误匹配（外点）", zorder=3)
    axes[0].legend(loc="lower right", fontsize=9, frameon=False)
    axes[0].set_xlabel("图 B · x")
    axes[0].set_ylabel("图 B · y")

    axes[1].set_title("全体点硬拟合：H 被带偏")
    axes[1].set_xlim(-10, 250)
    axes[1].set_ylim(190, -10)
    axes[1].set_aspect("equal")
    axes[1].scatter(true_map[:, 0], true_map[:, 1], s=36, facecolors="none", edgecolors=TEAL, linewidths=1.4, label="真 H")
    axes[1].scatter(all_map[:, 0], all_map[:, 1], s=22, c=WAVE, marker="^", label="全点拟合", zorder=3)
    for a, b in zip(true_map, all_map):
        axes[1].plot([a[0], b[0]], [a[1], b[1]], color=MUTED, lw=0.7, alpha=0.7)
    axes[1].legend(loc="lower right", fontsize=9, frameon=False)

    axes[2].set_title("只用内点：接近真 H")
    axes[2].set_xlim(-10, 250)
    axes[2].set_ylim(190, -10)
    axes[2].set_aspect("equal")
    axes[2].scatter(true_map[:, 0], true_map[:, 1], s=36, facecolors="none", edgecolors=TEAL, linewidths=1.4, label="真 H")
    axes[2].scatter(good_map[:, 0], good_map[:, 1], s=22, c=TEAL, marker="o", label="内点拟合", zorder=3)
    for a, b in zip(true_map, good_map):
        axes[2].plot([a[0], b[0]], [a[1], b[1]], color=TEAL, lw=0.7, alpha=0.5)
    axes[2].legend(loc="lower right", fontsize=9, frameon=False)

    err_all = float(np.mean(np.linalg.norm(all_map - true_map, axis=1)))
    err_good = float(np.mean(np.linalg.norm(good_map - true_map, axis=1)))
    fig.suptitle(
        f"外点问题：全点拟合平均偏差 {err_all:.1f}px　|　内点拟合 {err_good:.1f}px",
        fontsize=12,
        color=INK,
    )
    save_fig(fig, "ch07_outliers_problem.png")


def plot_ransac_loop_diagram():
    """白话流程图：抽 4 点 → 估 H → 数内点 → 迭代。"""
    fig, ax = plt.subplots(figsize=(9.6, 4.2), layout="constrained")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4.2)
    ax.axis("off")
    ax.set_title("RANSAC 循环（单应性版本）")

    boxes = [
        (0.4, 2.2, "1. 随机抽\n4 对匹配"),
        (2.7, 2.2, "2. 用这 4 对\n估一张 H"),
        (5.0, 2.2, "3. 用 H 检查\n全部点，数内点"),
        (7.4, 2.2, "4. 记下内点\n最多的那张 H"),
    ]
    for x, y, text in boxes:
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                1.9,
                1.35,
                boxstyle="round,pad=0.08,rounding_size=0.15",
                facecolor="#e5f3f1",
                edgecolor=TEAL,
                lw=1.5,
            )
        )
        ax.text(x + 0.95, y + 0.67, text, ha="center", va="center", fontsize=11)

    for i in range(3):
        x0 = boxes[i][0] + 1.9
        x1 = boxes[i + 1][0]
        ax.annotate(
            "",
            xy=(x1 - 0.05, 2.9),
            xytext=(x0 + 0.05, 2.9),
            arrowprops=dict(arrowstyle="->", color=WAVE, lw=1.6),
        )

    ax.add_patch(
        FancyBboxPatch(
            (2.2, 0.35),
            5.6,
            1.2,
            boxstyle="round,pad=0.08,rounding_size=0.15",
            facecolor="#fde8df",
            edgecolor=WAVE,
            lw=1.4,
        )
    )
    ax.text(
        5.0,
        0.95,
        "重复很多次 → 留下「内点最多」的模型\n最后常再用全部内点精炼一次 H",
        ha="center",
        va="center",
        fontsize=12,
    )
    ax.annotate(
        "",
        xy=(8.35, 2.2),
        xytext=(7.2, 1.55),
        arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.2, connectionstyle="arc3,rad=-0.25"),
    )
    ax.text(8.6, 1.5, "下一轮\n再抽", fontsize=10, color=MUTED, ha="left")
    save_fig(fig, "ch07_ransac_loop.png")


def plot_one_iteration():
    """展示「抽 4 点 → 估 H → 数内点」的一次示意。"""
    src, dst, is_true, _ = make_correspondences(
        n_inliers=55, outlier_ratio=0.35, noise_px=1.0, seed=21
    )
    # 故意选 4 个真内点，演示「好运样本」
    true_idx = np.where(is_true)[0]
    sample = true_idx[:4]
    H = estimate_h_dlt(src[sample], dst[sample])
    err = reproj_errors(H, src, dst)
    thresh = 3.0
    mask = err < thresh

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2), layout="constrained")
    draw_two_panels(axes[0], axes[1], src, dst, mask=mask, title_a="图 A：样本 + 内/外点", title_b="图 B：同一分类")

    # 高亮样本四点
    axes[0].scatter(src[sample, 0], src[sample, 1], s=90, facecolors="none", edgecolors=WAVE, linewidths=2.0, zorder=5)
    axes[1].scatter(dst[sample, 0], dst[sample, 1], s=90, facecolors="none", edgecolors=WAVE, linewidths=2.0, zorder=5)
    for i, (p, q) in enumerate(zip(src[sample], dst[sample])):
        axes[0].text(p[0] + 3, p[1] - 3, f"s{i+1}", color=WAVE, fontsize=9)
        axes[1].text(q[0] + 3, q[1] - 3, f"s{i+1}", color=WAVE, fontsize=9)

    fig.suptitle(
        f"一次好运抽样：橙圈是抽出的 4 对；青点=误差<{thresh}px 的内点（{int(mask.sum())}/{len(mask)}）",
        fontsize=12,
    )
    save_fig(fig, "ch07_sample_fit.png")


def plot_inlier_outlier_vis():
    """findHomography RANSAC 后的内点/外点可视化。"""
    src, dst, is_true, H_true = make_correspondences(
        n_inliers=70, outlier_ratio=0.4, noise_px=1.2, seed=9
    )
    H, mask = cv2.findHomography(
        src,
        dst,
        method=cv2.RANSAC,
        ransacReprojThreshold=3.0,
        maxIters=1000,
        confidence=0.995,
    )
    mask = mask.ravel().astype(bool)

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.3), layout="constrained")
    draw_two_panels(
        axes[0],
        axes[1],
        src,
        dst,
        mask=mask,
        title_a=f"图 A　RANSAC 内点 {mask.sum()}/{len(mask)}",
        title_b="图 B　对应点分类",
    )

    # 在图 B 上画几个真 H 与估 H 的映射对比
    probes = np.array([[40, 40], [160, 40], [160, 120], [40, 120]], dtype=np.float64)
    tmap = apply_h(H_true, probes)
    emap = apply_h(H, probes)
    axes[1].scatter(tmap[:, 0], tmap[:, 1], s=48, facecolors="none", edgecolors=INK, linewidths=1.3, label="真 H 映射", zorder=5)
    axes[1].scatter(emap[:, 0], emap[:, 1], s=28, c=TEAL, marker="D", label="估 H 映射", zorder=5)
    axes[1].legend(loc="lower right", fontsize=9, frameon=False)

    # 统计：RANSAC 内点与真内点的重合
    tp = int((mask & is_true).sum())
    fp = int((mask & ~is_true).sum())
    fig.suptitle(
        f"cv2.findHomography(..., RANSAC)：与真内点重合 {tp}，误收外点 {fp}",
        fontsize=12,
    )
    save_fig(fig, "ch07_inlier_outlier.png")

    # 另存一张「匹配连线」风格，便于课文对照
    fig2, ax = plt.subplots(figsize=(10.2, 3.8), layout="constrained")
    ax.set_xlim(0, 520)
    ax.set_ylim(200, -10)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("把图 A / 图 B 并排放：青线=内点匹配，橙线=外点匹配")

    # 左框 A，右框 B
    ax.add_patch(plt.Rectangle((20, 10), 200, 160, fill=False, edgecolor=INK, lw=1.2))
    ax.add_patch(plt.Rectangle((300, 10), 200, 160, fill=False, edgecolor=INK, lw=1.2))
    ax.text(120, 185, "图 A", ha="center", fontsize=12)
    ax.text(400, 185, "图 B", ha="center", fontsize=12)

    # 只画一部分线以免糊成一团
    show = np.arange(len(src))
    RNG2 = np.random.default_rng(0)
    show = RNG2.choice(show, size=min(40, len(show)), replace=False)
    for i in show:
        x1, y1 = src[i, 0] + 20, src[i, 1] + 10
        x2, y2 = dst[i, 0] * (200 / 240) + 300, dst[i, 1] * (160 / 180) + 10
        color = TEAL if mask[i] else WAVE
        alpha = 0.55 if mask[i] else 0.35
        ax.plot([x1, x2], [y1, y2], color=color, lw=0.9, alpha=alpha)
        ax.scatter([x1], [y1], s=12, c=color, zorder=3)
        ax.scatter([x2], [y2], s=12, c=color, zorder=3)
    save_fig(fig2, "ch07_match_lines.png")


def plot_threshold_effect():
    """不同重投影阈值下的内点数。"""
    src, dst, is_true, _ = make_correspondences(
        n_inliers=70, outlier_ratio=0.35, noise_px=1.5, seed=15
    )
    thresholds = [1.0, 2.0, 3.0, 5.0, 10.0]
    counts = []
    precision = []  # 内点里真内点比例
    recall = []     # 真内点被找回的比例

    for t in thresholds:
        _, mask = cv2.findHomography(
            src, dst, method=cv2.RANSAC, ransacReprojThreshold=t, maxIters=800
        )
        mask = mask.ravel().astype(bool)
        counts.append(int(mask.sum()))
        if mask.sum() == 0:
            precision.append(0.0)
            recall.append(0.0)
        else:
            precision.append(float((mask & is_true).sum() / mask.sum()))
            recall.append(float((mask & is_true).sum() / is_true.sum()))

    fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.8), layout="constrained")
    axes[0].plot(thresholds, counts, "o-", color=TEAL, lw=2)
    axes[0].set_xlabel("ransacReprojThreshold（像素）")
    axes[0].set_ylabel("判为内点的数量")
    axes[0].set_title("阈值越大 → 内点越多（也可能掺假）")
    axes[0].grid(True, color="#e6dccb")

    axes[1].plot(thresholds, precision, "o-", color=TEAL, lw=2, label="精确率：内点里真内点占比")
    axes[1].plot(thresholds, recall, "s--", color=WAVE, lw=2, label="召回率：真内点被找回占比")
    axes[1].set_xlabel("ransacReprojThreshold（像素）")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].set_title("阈值的两难：太松掺假，太紧漏真")
    axes[1].legend(loc="lower right", fontsize=9, frameon=False)
    axes[1].grid(True, color="#e6dccb")
    save_fig(fig, "ch07_threshold.png")


def plot_iterations_effect():
    """迭代次数太少时，可能抽不到好样本。"""
    src, dst, is_true, _ = make_correspondences(
        n_inliers=50, outlier_ratio=0.45, noise_px=1.2, seed=42
    )
    # 用手工 RANSAC 记录「迄今最佳内点数」曲线
    _, _, history = ransac_homography(src, dst, thresh=3.0, max_iters=400, seed=99)
    best_so_far = []
    cur = 0
    for _, count, is_best in history:
        if is_best:
            cur = count
        best_so_far.append(cur)

    fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.8), layout="constrained")
    axes[0].plot(best_so_far, color=TEAL, lw=1.8)
    axes[0].axhline(is_true.sum(), color=WAVE, ls="--", lw=1.2, label=f"真内点数 ≈ {is_true.sum()}")
    axes[0].set_xlabel("迭代次数")
    axes[0].set_ylabel("迄今最多内点数")
    axes[0].set_title("多抽几轮，才更可能撞上好样本")
    axes[0].legend(loc="lower right", fontsize=9, frameon=False)
    axes[0].grid(True, color="#e6dccb")

    # 不同 maxIters 下 OpenCV 结果与真 H 的偏差
    iters_list = [10, 30, 80, 200, 800]
    mean_err = []
    H_true = true_homography()
    grid = np.mgrid[20:181:40, 20:131:40].reshape(2, -1).T.astype(np.float64)
    true_map = apply_h(H_true, grid)
    for it in iters_list:
        # 固定随机：多次取平均，减弱运气
        errs = []
        for s in range(8):
            H, mask = cv2.findHomography(
                src,
                dst,
                method=cv2.RANSAC,
                ransacReprojThreshold=3.0,
                maxIters=it,
                confidence=0.99,
            )
            if H is None:
                errs.append(50.0)
                continue
            pred = apply_h(H, grid)
            errs.append(float(np.mean(np.linalg.norm(pred - true_map, axis=1))))
        mean_err.append(float(np.mean(errs)))

    axes[1].plot(iters_list, mean_err, "o-", color=WAVE, lw=2)
    axes[1].set_xlabel("maxIters")
    axes[1].set_ylabel("格点映射平均误差（px）")
    axes[1].set_title("迭代太少：估计更不稳、误差更大")
    axes[1].grid(True, color="#e6dccb")
    save_fig(fig, "ch07_iterations.png")


def plot_findhomography_api():
    """对照：无 RANSAC vs RANSAC 的 API 结果。"""
    src, dst, is_true, H_true = make_correspondences(
        n_inliers=65, outlier_ratio=0.4, noise_px=1.0, seed=5
    )
    H_ls, _ = cv2.findHomography(src, dst, method=0)
    H_r, mask = cv2.findHomography(
        src, dst, method=cv2.RANSAC, ransacReprojThreshold=3.0, maxIters=1000
    )
    mask = mask.ravel().astype(bool)

    grid = np.mgrid[20:181:45, 20:131:45].reshape(2, -1).T.astype(np.float64)
    tmap = apply_h(H_true, grid)
    lmap = apply_h(H_ls, grid)
    rmap = apply_h(H_r, grid)

    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.7), layout="constrained")
    titles = [
        ("真单应 H*", tmap, TEAL),
        ("method=0 全体拟合", lmap, WAVE),
        ("method=RANSAC", rmap, TEAL),
    ]
    for ax, (title, pts, color) in zip(axes, titles):
        ax.set_xlim(-10, 250)
        ax.set_ylim(190, -10)
        ax.set_aspect("equal")
        ax.set_title(title)
        ax.scatter(tmap[:, 0], tmap[:, 1], s=40, facecolors="none", edgecolors=INK, linewidths=1.1, label="真位置")
        if title.startswith("真"):
            ax.scatter(pts[:, 0], pts[:, 1], s=18, c=color, zorder=3)
        else:
            ax.scatter(pts[:, 0], pts[:, 1], s=22, c=color, marker="^", zorder=3, label="估计")
            for a, b in zip(tmap, pts):
                ax.plot([a[0], b[0]], [a[1], b[1]], color=MUTED, lw=0.6, alpha=0.65)
            ax.legend(loc="lower right", fontsize=8, frameon=False)
        ax.grid(True, color="#e6dccb")

    e_ls = float(np.mean(np.linalg.norm(lmap - tmap, axis=1)))
    e_r = float(np.mean(np.linalg.norm(rmap - tmap, axis=1)))
    fig.suptitle(
        f"findHomography：全体拟合误差 {e_ls:.1f}px　→　RANSAC {e_r:.1f}px（内点 {mask.sum()}）",
        fontsize=12,
    )
    save_fig(fig, "ch07_findHomography.png")


def plot_mine():
    """跟随「改这里」参数的个人实验图。"""
    ratio = float(np.clip(OUTLIER_RATIO, 0.0, 0.7))
    thresh = float(max(0.5, RANSAC_THRESH))
    iters = int(max(10, RANSAC_ITERS))
    noise = float(max(0.0, NOISE_PX))

    src, dst, is_true, H_true = make_correspondences(
        n_inliers=60, outlier_ratio=ratio, noise_px=noise, seed=77
    )
    H, mask = cv2.findHomography(
        src,
        dst,
        method=cv2.RANSAC,
        ransacReprojThreshold=thresh,
        maxIters=iters,
        confidence=0.995,
    )
    if H is None:
        mask = np.zeros(len(src), dtype=bool)
        mean_err = float("nan")
    else:
        mask = mask.ravel().astype(bool)
        grid = np.mgrid[20:181:40, 20:131:40].reshape(2, -1).T.astype(np.float64)
        mean_err = float(np.mean(np.linalg.norm(apply_h(H, grid) - apply_h(H_true, grid), axis=1)))

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2), layout="constrained")
    draw_two_panels(axes[0], axes[1], src, dst, mask=mask if H is not None else None)
    axes[0].set_title("你的参数 · 图 A")
    axes[1].set_title("你的参数 · 图 B")
    fig.suptitle(
        f"外点比例={ratio:.2f}  阈值={thresh:.1f}px  迭代={iters}  噪声={noise:.1f}px\n"
        f"RANSAC 内点 {int(mask.sum())}/{len(mask)}　真内点 {int(is_true.sum())}　"
        f"与真 H 格点误差 ≈ {mean_err:.2f}px",
        fontsize=11,
    )
    save_fig(fig, "ch07_my_ransac.png")


def self_check():
    # 1) 真 H 对自己映射误差应接近 0
    H = true_homography()
    pts = np.array([[10.0, 10.0], [100.0, 50.0], [180.0, 120.0]])
    back = apply_h(H, pts)
    again = apply_h(np.linalg.inv(H), back)
    assert np.allclose(pts, again, atol=1e-6)

    # 2) 4 点应能恢复透视
    src4 = np.float32([[0, 0], [100, 0], [100, 80], [0, 80]])
    dst4 = apply_h(H, src4).astype(np.float32)
    H2 = cv2.getPerspectiveTransform(src4, dst4)
    assert H2.shape == (3, 3)

    # 3) 有外点时 RANSAC 应优于全体拟合（统计意义上）
    src, dst, is_true, H_true = make_correspondences(
        n_inliers=80, outlier_ratio=0.4, noise_px=1.0, seed=1
    )
    H_ls, _ = cv2.findHomography(src, dst, method=0)
    H_r, mask = cv2.findHomography(
        src, dst, method=cv2.RANSAC, ransacReprojThreshold=3.0, maxIters=2000
    )
    assert H_r is not None and mask is not None
    mask = mask.ravel().astype(bool)
    assert mask.sum() >= 4

    grid = np.mgrid[20:181:40, 20:131:40].reshape(2, -1).T.astype(np.float64)
    e_ls = np.mean(np.linalg.norm(apply_h(H_ls, grid) - apply_h(H_true, grid), axis=1))
    e_r = np.mean(np.linalg.norm(apply_h(H_r, grid) - apply_h(H_true, grid), axis=1))
    assert e_r < e_ls, f"RANSAC 应更准: {e_r=} vs {e_ls=}"

    # 4) 手工 RANSAC 至少找到不少内点
    H3, mask3, hist = ransac_homography(src, dst, thresh=3.0, max_iters=300, seed=2)
    assert H3 is not None and mask3.sum() >= 20
    assert len(hist) == 300

    print("自检通过：齐次往返、4 点估 H、RANSAC 优于全体拟合、手写循环可运行。")


def main():
    use_style()
    self_check()
    plot_outlier_problem()
    plot_ransac_loop_diagram()
    plot_one_iteration()
    plot_inlier_outlier_vis()
    plot_threshold_effect()
    plot_iterations_effect()
    plot_findhomography_api()
    plot_mine()
    print("课文对照图已写入 assets/figures/")
    print("你的图：ch07_my_ransac.png")


if __name__ == "__main__":
    main()
