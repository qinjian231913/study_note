"""第 10 篇：接缝寻找与图像融合。

    MPLCONFIGDIR=/tmp/mpl-cache .venv/bin/python code/ch10_blend.py

顶部「改这里」只影响 ch10_my_feather.png。
"""

from pathlib import Path

SHOW_WINDOW = False

import matplotlib

if not SHOW_WINDOW:
    matplotlib.use("Agg")

import cv2
import matplotlib.pyplot as plt
import numpy as np

from plot_style import INK, MUTED, TEAL, WAVE, use_style

# ===== 改这里 =====
# 羽化半宽（像素）。越大，重叠区过渡越宽、越柔；过大会让重影更明显。
FEATHER_HALF_WIDTH = 40
# 两图在重叠区的水平错位（像素）。越大，平均/羽化的「重影」越明显，
# 也越能看出「找缝」避开错位结构的好处。
MISALIGN_PX = 8
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


def make_pair(misalign=8, h=220, w=360, overlap=140):
    """造两张「已对齐到同一画布」的图：左暖右冷，重叠区有竖杆错位与亮度差。

    返回：
      left, right — 同尺寸 BGR，未覆盖处为黑
      mask_l, mask_r — 有效像素掩膜（0/1 float）
      x0, x1 — 重叠区间 [x0, x1)
    """
    left = np.zeros((h, w, 3), dtype=np.uint8)
    right = np.zeros((h, w, 3), dtype=np.uint8)
    x0 = w // 2 - overlap // 2
    x1 = x0 + overlap

    yy, xx = np.mgrid[0:h, 0:w]

    # 左图暖色底（含缓变，避免「死色块」）
    base_l = 200 + 20 * np.sin(yy / 28.0) + 10 * np.sin(xx / 40.0)
    left_f = np.stack(
        [base_l - 35, base_l - 5, base_l + 35],
        axis=2,
    )
    left[:, :x1] = np.clip(left_f[:, :x1], 40, 255).astype(np.uint8)

    # 右图冷色、略暗
    base_r = 175 + 16 * np.sin(yy / 32.0) + 8 * np.sin(xx / 45.0)
    right_f = np.stack(
        [base_r + 40, base_r + 10, base_r - 30],
        axis=2,
    )
    right[:, x0:] = np.clip(right_f[:, x0:], 30, 255).astype(np.uint8)

    # 重叠区：共享一条「低差走廊」+ 两侧故意错位的粗竖杆（制造高代价带）
    # 低差走廊：让右图在走廊附近抄左图颜色，DP 会倾向于从这里穿过
    corridor_x = (x0 + x1) // 2 - 8
    band = 18
    for x in range(max(x0, corridor_x - band), min(x1, corridor_x + band)):
        t = 1.0 - abs(x - corridor_x) / band
        right[:, x] = (
            (1 - 0.85 * t) * right[:, x].astype(np.float32)
            + (0.85 * t) * left[:, x].astype(np.float32)
        ).astype(np.uint8)

    # 竖杆：左图位置 vs 右图错位，重叠区会出现双影
    poles_l = [x0 + 25, x0 + 55, x0 + 95, x0 + 115]
    for i, cx in enumerate(poles_l):
        cv2.rectangle(left, (cx, 15), (cx + 14, h - 15), WAVE_BGR, -1)
        cv2.putText(
            left,
            "AB"[i % 2],
            (cx + 1, 48),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cx2 = cx + misalign
        if x0 <= cx2 < x1 - 14:
            cv2.rectangle(right, (cx2, 15), (cx2 + 14, h - 15), TEAL_BGR, -1)
            cv2.putText(
                right,
                "AB"[i % 2],
                (cx2 + 1, 48),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

    # 斜纹：错位后平均会糊成网状
    for t in range(-h, w, 18):
        cv2.line(left, (t, 0), (t + h // 3, h - 1), (70, 100, 200), 2)
        cv2.line(
            right,
            (t + misalign, 0),
            (t + misalign + h // 3, h - 1),
            (180, 150, 50),
            2,
        )

    # 非覆盖区清零（斜线可能画出界）
    left[:, x1:] = 0
    right[:, :x0] = 0

    mask_l = (left.sum(axis=2) > 0).astype(np.float32)
    mask_r = (right.sum(axis=2) > 0).astype(np.float32)
    return left, right, mask_l, mask_r, x0, x1


def hard_cut(left, right, mask_l, mask_r, x0, x1):
    """硬切：重叠区以中线为界，左半用左图、右半用右图。"""
    out = left.copy()
    mid = (x0 + x1) // 2
    use_r = mask_r.copy()
    use_r[:, :mid] = 0
    m = use_r[:, :, None]
    out = np.where(m > 0, right, out)
    # 仅右图有、左图没有的区域
    only_r = (mask_r > 0) & (mask_l == 0)
    out[only_r] = right[only_r]
    return out.astype(np.uint8), mid


def average_blend(left, right, mask_l, mask_r):
    """重叠区简单平均；单侧区域直接取该侧。"""
    lf = left.astype(np.float32)
    rf = right.astype(np.float32)
    both = (mask_l > 0) & (mask_r > 0)
    only_l = (mask_l > 0) & (mask_r == 0)
    only_r = (mask_r > 0) & (mask_l == 0)
    out = np.zeros_like(lf)
    out[only_l] = lf[only_l]
    out[only_r] = rf[only_r]
    out[both] = 0.5 * (lf[both] + rf[both])
    return np.clip(out, 0, 255).astype(np.uint8)


def feather_weights(mask_l, mask_r, half_width, x0, x1):
    """在重叠区做一维线性羽化：从左到右，左图权重 1→0。"""
    h, w = mask_l.shape
    wl = np.zeros((h, w), dtype=np.float32)
    wr = np.zeros((h, w), dtype=np.float32)
    only_l = (mask_l > 0) & (mask_r == 0)
    only_r = (mask_r > 0) & (mask_l == 0)
    both = (mask_l > 0) & (mask_r > 0)
    wl[only_l] = 1.0
    wr[only_r] = 1.0

    mid = (x0 + x1) / 2.0
    half = max(1.0, float(half_width))
    xs = np.arange(w, dtype=np.float32)
    # 中心附近线性过渡；超出半宽则一侧完全占优
    alpha = np.clip((mid + half - xs) / (2.0 * half), 0.0, 1.0)
    for y in range(h):
        cols = both[y]
        wl[y, cols] = alpha[cols]
        wr[y, cols] = 1.0 - alpha[cols]
    return wl, wr


def feather_blend(left, right, mask_l, mask_r, half_width, x0, x1):
    wl, wr = feather_weights(mask_l, mask_r, half_width, x0, x1)
    out = left.astype(np.float32) * wl[:, :, None] + right.astype(np.float32) * wr[:, :, None]
    covered = (mask_l + mask_r) > 0
    result = np.zeros_like(left)
    result[covered] = np.clip(out[covered], 0, 255).astype(np.uint8)
    return result, wl, wr


def build_gaussian_pyramid(img, levels):
    g = [img.astype(np.float32)]
    for _ in range(1, levels):
        g.append(cv2.pyrDown(g[-1]))
    return g


def build_laplacian_pyramid(img, levels):
    g = build_gaussian_pyramid(img, levels)
    laps = []
    for i in range(levels - 1):
        size = (g[i].shape[1], g[i].shape[0])
        up = cv2.pyrUp(g[i + 1], dstsize=size)
        laps.append(g[i] - up)
    laps.append(g[-1])
    return laps


def collapse_laplacian(laps):
    img = laps[-1]
    for i in range(len(laps) - 2, -1, -1):
        size = (laps[i].shape[1], laps[i].shape[0])
        img = cv2.pyrUp(img, dstsize=size) + laps[i]
    return img


def multiband_blend(left, right, mask_l, mask_r, half_width, x0, x1, levels=4):
    """多频段融合：对每层拉普拉斯用不同「平滑程度」的权重混合。"""
    wl, wr = feather_weights(mask_l, mask_r, half_width, x0, x1)
    # 权重也建高斯金字塔：低频用更糊的权重 → 过渡更宽
    w_pyr = build_gaussian_pyramid(wl, levels)
    l_left = build_laplacian_pyramid(left, levels)
    l_right = build_laplacian_pyramid(right, levels)
    blended = []
    for i in range(levels):
        wi = w_pyr[i]
        # 尺寸对齐（偶发差 1 像素）
        h, w = l_left[i].shape[:2]
        wi = cv2.resize(wi, (w, h), interpolation=cv2.INTER_LINEAR)
        wi = np.clip(wi, 0, 1)
        if l_left[i].ndim == 3:
            wi = wi[:, :, None]
        blended.append(l_left[i] * wi + l_right[i] * (1.0 - wi))
    out = collapse_laplacian(blended)
    covered = (mask_l + mask_r) > 0
    result = np.zeros_like(left)
    result[covered] = np.clip(out[covered], 0, 255).astype(np.uint8)
    return result


def difference_cost(left, right, mask_l, mask_r, x0, x1):
    """重叠区逐像素代价：灰度差越大，越不适合当接缝。"""
    gray_l = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray_r = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY).astype(np.float32)
    both = (mask_l > 0) & (mask_r > 0)
    cost = np.full(gray_l.shape, 1e6, dtype=np.float32)
    cost[both] = np.abs(gray_l[both] - gray_r[both])
    # 只在重叠列上找缝
    cost[:, :x0] = 1e6
    cost[:, x1:] = 1e6
    return cost


def find_seam_dp(cost, x0, x1):
    """动态规划找从上到下的最低代价竖缝（每行一个 x）。"""
    h, w = cost.shape
    dp = cost.copy()
    parent = np.zeros((h, w), dtype=np.int32)
    for y in range(1, h):
        for x in range(x0, x1):
            cands = []
            for dx in (-1, 0, 1):
                px = x + dx
                if x0 <= px < x1:
                    cands.append((dp[y - 1, px], px))
            best_val, best_px = min(cands, key=lambda t: t[0])
            dp[y, x] = cost[y, x] + best_val
            parent[y, x] = best_px
    # 回溯
    x = int(np.argmin(dp[h - 1, x0:x1]) + x0)
    seam = np.zeros(h, dtype=np.int32)
    for y in range(h - 1, -1, -1):
        seam[y] = x
        if y > 0:
            x = int(parent[y, x])
    return seam, dp


def seam_cut(left, right, mask_l, mask_r, seam):
    """按接缝硬切：缝左侧用左图，右侧用右图。"""
    h, w = mask_l.shape
    out = left.copy()
    for y in range(h):
        sx = int(seam[y])
        # 右图从缝右侧开始
        cols = np.arange(w)
        use_r = (cols >= sx) & (mask_r[y] > 0)
        out[y, use_r] = right[y, use_r]
        only_r = (mask_r[y] > 0) & (mask_l[y] == 0)
        out[y, only_r] = right[y, only_r]
    return out


def plot_overlap_setup(left, right, x0, x1):
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.6), layout="constrained")
    axes[0].imshow(left[:, :, ::-1])
    axes[0].axvline(x0, color=WAVE, ls="--", lw=1.2)
    axes[0].axvline(x1, color=WAVE, ls="--", lw=1.2)
    axes[0].set_title("左图（暖色）")
    axes[0].axis("off")
    axes[1].imshow(right[:, :, ::-1])
    axes[1].axvline(x0, color=TEAL, ls="--", lw=1.2)
    axes[1].axvline(x1, color=TEAL, ls="--", lw=1.2)
    axes[1].set_title("右图（冷色，略错位）")
    axes[1].axis("off")
    both = np.zeros_like(left)
    m_l = left.sum(2) > 0
    m_r = right.sum(2) > 0
    both[m_l] = left[m_l]
    # 重叠区半透明叠右图示意
    mix = both.astype(np.float32)
    mix[m_r] = 0.5 * mix[m_r] + 0.5 * right[m_r].astype(np.float32)
    axes[2].imshow(np.clip(mix, 0, 255).astype(np.uint8)[:, :, ::-1])
    axes[2].axvline(x0, color=MUTED, ls="--", lw=1)
    axes[2].axvline(x1, color=MUTED, ls="--", lw=1)
    axes[2].set_title("叠在一起（重叠发糊）")
    axes[2].axis("off")
    fig.suptitle("两张已 warp 到同一画布的图：虚线之间是重叠区", color=INK, fontsize=13)
    save_fig(fig, "ch10_overlap_setup.png")
    save_bgr("ch10_left.png", left)
    save_bgr("ch10_right.png", right)


def plot_three_basic(left, right, mask_l, mask_r, x0, x1, feather_hw):
    hard, mid = hard_cut(left, right, mask_l, mask_r, x0, x1)
    avg = average_blend(left, right, mask_l, mask_r)
    soft, _, _ = feather_blend(left, right, mask_l, mask_r, feather_hw, x0, x1)

    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.8), layout="constrained")
    axes[0].imshow(hard[:, :, ::-1])
    axes[0].axvline(mid, color=WAVE, lw=1.4)
    axes[0].set_title("硬切：一条竖缝")
    axes[0].axis("off")
    axes[1].imshow(avg[:, :, ::-1])
    axes[1].set_title("平均：重叠发糊/重影")
    axes[1].axis("off")
    axes[2].imshow(soft[:, :, ::-1])
    axes[2].set_title(f"羽化：半宽 {feather_hw}px")
    axes[2].axis("off")
    fig.suptitle("三种最基础拼法对照", color=INK, fontsize=13)
    save_fig(fig, "ch10_hard_vs_avg_vs_feather.png")
    return hard, avg, soft


def plot_feather_weights(wl, wr, half_width):
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.4), layout="constrained")
    im0 = axes[0].imshow(wl, cmap="Oranges", vmin=0, vmax=1)
    axes[0].set_title("左图权重 wl")
    axes[0].axis("off")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    im1 = axes[1].imshow(wr, cmap="Greens", vmin=0, vmax=1)
    axes[1].set_title("右图权重 wr")
    axes[1].axis("off")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    # 中行剖面
    row = wl.shape[0] // 2
    axes[2].plot(wl[row], color=WAVE, lw=2, label="wl")
    axes[2].plot(wr[row], color=TEAL, lw=2, label="wr")
    axes[2].set_ylim(-0.05, 1.05)
    axes[2].set_xlabel("x（像素）")
    axes[2].set_ylabel("权重")
    axes[2].set_title(f"中间一行剖面（半宽={half_width}）")
    axes[2].legend(frameon=False)
    axes[2].grid(True, alpha=0.4)
    fig.suptitle("羽化 = 重叠区里左右权重交叉渐变，且 wl + wr = 1", color=INK, fontsize=13)
    save_fig(fig, "ch10_feather_weights.png")


def plot_pyramid_idea():
    fig, ax = plt.subplots(figsize=(9.2, 4.2), layout="constrained")
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.set_title("多频段 / 拉普拉斯金字塔融合（白话）")
    boxes = [
        (0.3, 3.4, 2.8, 1.2, "高频层\n（边缘、文字、竖条）", "#fde8df"),
        (0.3, 1.9, 2.8, 1.2, "中频层\n（较大色块变化）", "#fbf1dc"),
        (0.3, 0.4, 2.8, 1.2, "低频层\n（整体明暗、色调）", "#e5f3f1"),
    ]
    for x, y, w, h, text, fc in boxes:
        ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=fc, edgecolor=INK, lw=1.2))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=11)

    ax.annotate(
        "",
        xy=(4.0, 4.0),
        xytext=(3.3, 4.0),
        arrowprops=dict(arrowstyle="->", color=MUTED),
    )
    ax.annotate(
        "",
        xy=(4.0, 2.5),
        xytext=(3.3, 2.5),
        arrowprops=dict(arrowstyle="->", color=MUTED),
    )
    ax.annotate(
        "",
        xy=(4.0, 1.0),
        xytext=(3.3, 1.0),
        arrowprops=dict(arrowstyle="->", color=MUTED),
    )

    ax.text(4.1, 4.0, "窄过渡：接缝附近很快切到另一侧", fontsize=12, va="center", color=WAVE)
    ax.text(4.1, 2.5, "中等过渡", fontsize=12, va="center", color=MUTED)
    ax.text(4.1, 1.0, "宽过渡：亮度慢慢混过去，不易留色差带", fontsize=12, va="center", color=TEAL)
    ax.text(
        0.3,
        4.85,
        "把图拆成「粗细节 / 细细节」，各层用不同宽度的权重再加回去",
        fontsize=12,
        color=MUTED,
    )
    save_fig(fig, "ch10_pyramid_idea.png")


def plot_seam_and_results(left, right, mask_l, mask_r, x0, x1, feather_hw):
    cost = difference_cost(left, right, mask_l, mask_r, x0, x1)
    seam, _ = find_seam_dp(cost, x0, x1)
    cut = seam_cut(left, right, mask_l, mask_r, seam)
    feathered, _, _ = feather_blend(left, right, mask_l, mask_r, feather_hw, x0, x1)
    multi = multiband_blend(left, right, mask_l, mask_r, feather_hw, x0, x1, levels=4)

    # 代价 + 缝
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.8), layout="constrained")
    show = cost.copy()
    show[show > 1e5] = np.nan
    im = axes[0].imshow(show, cmap="magma")
    axes[0].plot(seam, np.arange(len(seam)), color="cyan", lw=2)
    axes[0].set_title("重叠区代价（亮=两图差大）+ 青色接缝")
    axes[0].axis("off")
    fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)
    axes[1].imshow(cut[:, :, ::-1])
    axes[1].plot(seam, np.arange(len(seam)), color=WAVE, lw=1.5)
    axes[1].set_title("沿接缝硬切的结果")
    axes[1].axis("off")
    fig.suptitle("动态规划找缝：每行选一个 x，使「从上到下总代价」尽量小", color=INK, fontsize=13)
    save_fig(fig, "ch10_seam_cost.png")

    # 三种进阶对照：羽化 / 找缝 / 多频段
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.8), layout="constrained")
    axes[0].imshow(feathered[:, :, ::-1])
    axes[0].set_title("① 线性羽化")
    axes[0].axis("off")
    axes[1].imshow(cut[:, :, ::-1])
    axes[1].set_title("② 低代价接缝硬切")
    axes[1].axis("off")
    axes[2].imshow(multi[:, :, ::-1])
    axes[2].set_title("③ 多频段融合")
    axes[2].axis("off")
    fig.suptitle("三种结果观感对照（同一对合成图）", color=INK, fontsize=13)
    save_fig(fig, "ch10_three_results.png")

    # 多频段单独大图
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8), layout="constrained")
    axes[0].imshow(feathered[:, :, ::-1])
    axes[0].set_title("单层羽化（细节易糊）")
    axes[0].axis("off")
    axes[1].imshow(multi[:, :, ::-1])
    axes[1].set_title("多频段（高频窄混、低频宽混）")
    axes[1].axis("off")
    save_fig(fig, "ch10_multiband_result.png")
    return cut, multi, seam


def plot_mine(left, right, mask_l, mask_r, x0, x1, half_width):
    soft, wl, _ = feather_blend(left, right, mask_l, mask_r, half_width, x0, x1)
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8), layout="constrained")
    axes[0].imshow(soft[:, :, ::-1])
    axes[0].set_title(f"你的羽化结果（半宽={half_width}）")
    axes[0].axis("off")
    row = wl.shape[0] // 2
    axes[1].plot(wl[row], color=WAVE, lw=2)
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].set_title("中间行左图权重")
    axes[1].set_xlabel("x")
    axes[1].grid(True, alpha=0.4)
    save_fig(fig, "ch10_my_feather.png")


def self_check():
    left, right, mask_l, mask_r, x0, x1 = make_pair(misalign=6, h=80, w=120, overlap=40)
    hard, mid = hard_cut(left, right, mask_l, mask_r, x0, x1)
    assert hard.shape == left.shape and x0 < mid < x1

    avg = average_blend(left, right, mask_l, mask_r)
    both = (mask_l > 0) & (mask_r > 0)
    # 重叠区平均应介于两侧之间（抽样）
    ys, xs = np.where(both)
    i = len(ys) // 2
    y, x = int(ys[i]), int(xs[i])
    for c in range(3):
        lo = min(int(left[y, x, c]), int(right[y, x, c]))
        hi = max(int(left[y, x, c]), int(right[y, x, c]))
        assert lo <= int(avg[y, x, c]) <= hi

    wl, wr = feather_weights(mask_l, mask_r, 15, x0, x1)
    assert np.allclose(wl[both] + wr[both], 1.0, atol=1e-5)

    soft, _, _ = feather_blend(left, right, mask_l, mask_r, 15, x0, x1)
    assert soft.dtype == np.uint8

    cost = difference_cost(left, right, mask_l, mask_r, x0, x1)
    seam, _ = find_seam_dp(cost, x0, x1)
    assert len(seam) == left.shape[0]
    assert seam.min() >= x0 and seam.max() < x1
    # 相邻行接缝最多偏 1 列（DP 转移约束）
    assert np.max(np.abs(np.diff(seam.astype(np.int32)))) <= 1

    multi = multiband_blend(left, right, mask_l, mask_r, 15, x0, x1, levels=3)
    assert multi.shape == left.shape

    # 拉普拉斯重建近似原图
    src = left.astype(np.float32)
    laps = build_laplacian_pyramid(src, 3)
    rec = collapse_laplacian(laps)
    err = np.mean(np.abs(rec - src))
    assert err < 2.0, f"拉普拉斯重建误差过大: {err}"

    print("自检通过：硬切/平均/羽化权重/DP 接缝/多频段重建。")


def main():
    use_style()
    self_check()

    left, right, mask_l, mask_r, x0, x1 = make_pair(misalign=MISALIGN_PX)
    plot_overlap_setup(left, right, x0, x1)
    _, _, soft = plot_three_basic(left, right, mask_l, mask_r, x0, x1, feather_hw=35)
    wl, wr = feather_weights(mask_l, mask_r, 35, x0, x1)
    plot_feather_weights(wl, wr, 35)
    plot_pyramid_idea()
    plot_seam_and_results(left, right, mask_l, mask_r, x0, x1, feather_hw=35)
    plot_mine(left, right, mask_l, mask_r, x0, x1, FEATHER_HALF_WIDTH)

    print("课文对照图已写入 assets/figures/")
    print(f"你的图：ch10_my_feather.png（FEATHER_HALF_WIDTH={FEATHER_HALF_WIDTH}）")


if __name__ == "__main__":
    main()
