"""第 25 篇：视频拼接与时空一致性。

多路视频不能逐帧独立套用静态拼接流水线。本脚本用合成双路序列演示：
时间戳同步、帧间 H 平滑、接缝时序稳定，以及「闪烁 / 缝跳舞」对照。

在「视频图片拼接防抖」文件夹里运行：

    MPLCONFIGDIR=/tmp/mpl-cache .venv/bin/python code/ch25_video_stitch.py

顶部「改这里」只影响 ch25_my_*.png。课文对照图每次按固定参数重画。
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
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

from plot_style import INK, MUTED, PAPER, TEAL, WAVE, finish, use_style

# ===== 改这里 =====
# 右相机相对左相机的时间偏移（毫秒）。改大 → 未同步时错配更严重
TIME_OFFSET_MS = 33.0
# 每帧估 H 时叠加的平移噪声（像素标准差）。越大 → 「缝跳舞 / 闪烁」越明显
H_NOISE_STD = 2.8
# 对 H 平移分量做滑动平均的窗口（奇数更直观）。越大越稳，但跟手变慢
H_SMOOTH_WINDOW = 7
# 对接缝横坐标做滑动平均的窗口
SEAM_SMOOTH_WINDOW = 9
# 演示序列帧数（越大图越密，略慢）
N_FRAMES = 48
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
# 合成：双路视频帧 + 真实相对平移 + 噪声观测
# ---------------------------------------------------------------------------


def make_pano_strip(w=520, h=180):
    """一条宽「街景」，左右两路相机各自裁一块重叠视野。"""
    img = np.full((h, w, 3), 236, dtype=np.uint8)
    img[: h // 2] = (232, 214, 188)
    img[h // 2 :] = (208, 198, 182)

    # 楼宇竖条：接缝错位时特别明显
    for x, bw, bh, color in (
        (40, 36, 90, (95, 110, 140)),
        (110, 28, 70, (70, 95, 125)),
        (190, 42, 105, (105, 90, 70)),
        (270, 34, 80, (80, 115, 105)),
        (340, 40, 95, (115, 85, 90)),
        (410, 32, 75, (90, 100, 120)),
        (460, 38, 88, (100, 95, 75)),
    ):
        top = h // 2 - bh
        cv2.rectangle(img, (x, top), (x + bw, h // 2), color, -1)
        cv2.rectangle(img, (x, top), (x + bw, h // 2), (40, 40, 40), 1)

    # 路面虚线
    y = h // 2 + 28
    for x0 in range(10, w - 10, 36):
        cv2.line(img, (x0, y), (x0 + 18, y), (250, 250, 250), 2)

    # 竖向接缝「标尺」线：便于看出缝左右跳
    for x in (200, 260, 320):
        cv2.line(img, (x, 8), (x, h - 8), (160, 145, 130), 1)

    # 一棵树：跟踪用
    cv2.rectangle(img, (248, 95), (256, 140), (55, 75, 40), -1)
    cv2.circle(img, (252, 82), 22, (45, 110, 65), -1)
    return img


def crop_pair(scene, left_x, right_x, view_w, view_h):
    left = scene[:, left_x : left_x + view_w].copy()
    right = scene[:, right_x : right_x + view_w].copy()
    return left, right


def true_tx_sequence(n, base=160.0, amp=6.0):
    """缓慢变化的真实相对平移（像素）：像两路相机一起微微晃。"""
    t = np.arange(n, dtype=np.float64)
    return base + amp * np.sin(2 * np.pi * t / 28.0) + 1.2 * np.sin(2 * np.pi * t / 11.0)


def noisy_tx(true_tx, noise_std, rng):
    return true_tx + rng.normal(0.0, noise_std, size=true_tx.shape)


def moving_average(x, window):
    """简单滑动平均；边界用边缘值填充，长度不变。"""
    w = max(1, int(window))
    if w % 2 == 0:
        w += 1
    pad = w // 2
    xp = np.pad(x.astype(np.float64), (pad, pad), mode="edge")
    kernel = np.ones(w, dtype=np.float64) / w
    return np.convolve(xp, kernel, mode="valid")


def hard_blend(left, right, seam_x):
    """竖直硬接缝：seam_x 左侧取 left，右侧取 right（同尺寸画布）。"""
    h, w = left.shape[:2]
    sx = int(np.clip(seam_x, 1, w - 2))
    out = left.copy()
    out[:, sx:] = right[:, sx:]
    return out, sx


def stitch_with_tx(left, right, tx, canvas_w, seam_x_abs):
    """把右图按水平平移 tx 叠到左图画布，再在绝对坐标 seam_x_abs 切一刀。"""
    h, w = left.shape[:2]
    canvas = np.full((h, canvas_w, 3), 251, dtype=np.uint8)
    canvas[:, :w] = left
    # 右图放到 tx 处（简化：只平移，忽略旋转）
    x0 = int(round(tx))
    x1 = x0 + w
    if x1 <= 0 or x0 >= canvas_w:
        return canvas, int(seam_x_abs)
    src_x0 = max(0, -x0)
    dst_x0 = max(0, x0)
    src_x1 = min(w, canvas_w - x0)
    dst_x1 = dst_x0 + (src_x1 - src_x0)
    # 只在右图覆盖区写入（左图已在）
    canvas[:, dst_x0:dst_x1] = right[:, src_x0:src_x1]
    # 硬接缝：接缝左侧强制保留左图
    sx = int(np.clip(seam_x_abs, 1, canvas_w - 2))
    if sx < w:
        canvas[:, :sx] = left[:, :sx]
    return canvas, sx


# ---------------------------------------------------------------------------
# 时间戳同步
# ---------------------------------------------------------------------------


def make_timestamps(n, fps=30.0, offset_ms=0.0):
    """左路从 0 起；右路整体偏移 offset_ms。"""
    dt = 1000.0 / fps
    left_ts = np.arange(n, dtype=np.float64) * dt
    right_ts = left_ts + float(offset_ms)
    return left_ts, right_ts


def sync_nearest(left_ts, right_ts):
    """对每个左帧，找时间戳最近的右帧索引。"""
    pairs = []
    errs = []
    for i, t in enumerate(left_ts):
        j = int(np.argmin(np.abs(right_ts - t)))
        pairs.append((i, j))
        errs.append(abs(right_ts[j] - t))
    return pairs, np.asarray(errs)


# ---------------------------------------------------------------------------
# 作图
# ---------------------------------------------------------------------------


def plot_independent_fail():
    """图：逐帧独立估 H → 拼缝与标尺线「抖」。"""
    scene = make_pano_strip()
    view_w, view_h = 220, scene.shape[0]
    left_x0, right_x0 = 40, 200
    n = 24
    rng = np.random.default_rng(7)
    true_tx = true_tx_sequence(n, base=float(right_x0 - left_x0), amp=4.0)
    obs_tx = noisy_tx(true_tx, noise_std=3.2, rng=rng)
    smooth_tx = moving_average(obs_tx, 7)

    # 取 5 帧条带：独立 vs 平滑
    idxs = [2, 7, 12, 17, 21]
    strips_raw = []
    strips_sm = []
    canvas_w = 360
    for i in idxs:
        left, right = crop_pair(scene, left_x0, right_x0, view_w, view_h)
        # 轻微晃动：裁切起点抖动，模拟手持
        jx = int(round(1.5 * np.sin(i * 0.7)))
        left2, right2 = crop_pair(
            scene,
            left_x0 + jx,
            right_x0 + jx,
            view_w,
            view_h,
        )
        raw, _ = stitch_with_tx(left2, right2, obs_tx[i], canvas_w, seam_x_abs=view_w - 40)
        sm, _ = stitch_with_tx(left2, right2, smooth_tx[i], canvas_w, seam_x_abs=view_w - 40)
        # 画接缝
        sx = view_w - 40
        cv2.line(raw, (sx, 0), (sx, view_h - 1), WAVE_BGR, 1)
        cv2.line(sm, (sx, 0), (sx, view_h - 1), TEAL_BGR, 1)
        strips_raw.append(raw)
        strips_sm.append(sm)

    fig, axes = plt.subplots(2, 5, figsize=(11.2, 4.4), layout="constrained")
    for c, i in enumerate(idxs):
        axes[0, c].imshow(strips_raw[c][:, :, ::-1])
        axes[0, c].set_title(f"独立 t={i}", fontsize=10)
        axes[0, c].axis("off")
        axes[1, c].imshow(strips_sm[c][:, :, ::-1])
        axes[1, c].set_title(f"平滑 t={i}", fontsize=10)
        axes[1, c].axis("off")
    axes[0, 0].set_ylabel("逐帧独立估 H", fontsize=11)
    axes[1, 0].set_ylabel("帧间平滑后", fontsize=11)
    fig.suptitle("不能逐帧独立拼：独立估 H 时楼边/标尺会左右跳（上）；平滑后稳（下）", fontsize=12)
    save_fig(fig, "ch25_independent_fail.png")

    # OpenCV 对照图：同一帧独立 vs 平滑
    i = idxs[2]
    left, right = crop_pair(scene, left_x0, right_x0, view_w, view_h)
    raw, sx = stitch_with_tx(left, right, obs_tx[i], canvas_w, seam_x_abs=view_w - 40)
    sm, _ = stitch_with_tx(left, right, smooth_tx[i], canvas_w, seam_x_abs=view_w - 40)
    cv2.line(raw, (sx, 0), (sx, view_h - 1), WAVE_BGR, 2)
    cv2.line(sm, (sx, 0), (sx, view_h - 1), TEAL_BGR, 2)
    cv2.putText(raw, "independent H", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, WAVE_BGR, 1, cv2.LINE_AA)
    cv2.putText(sm, "smoothed H", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, TEAL_BGR, 1, cv2.LINE_AA)
    both = np.vstack([raw, sm])
    save_bgr("ch25_frame_compare.png", both)


def plot_timestamp_sync(offset_ms=33.0):
    """时间戳：未对齐错配 vs 最近邻同步。"""
    n = 12
    left_ts, right_ts = make_timestamps(n, fps=30.0, offset_ms=offset_ms)
    # 未同步：错误地一对一按索引（忽略偏移）
    bad_err = np.abs(right_ts - left_ts)
    pairs, good_err = sync_nearest(left_ts, right_ts)

    fig, axes = plt.subplots(2, 1, figsize=(9.6, 5.2), layout="constrained", sharex=False)

    ax = axes[0]
    ax.set_title("两路时间轴：右路整体晚了约一帧（示意）")
    for i, t in enumerate(left_ts):
        ax.plot([t, t], [0.55, 0.85], color=WAVE, lw=2)
        ax.scatter([t], [0.7], color=WAVE, s=36, zorder=3)
    for j, t in enumerate(right_ts):
        ax.plot([t, t], [0.15, 0.45], color=TEAL, lw=2)
        ax.scatter([t], [0.3], color=TEAL, s=36, zorder=3)
    # 画几条错配（按索引）与正确配对
    for i in range(0, n, 3):
        ax.annotate(
            "",
            xy=(right_ts[i], 0.45),
            xytext=(left_ts[i], 0.55),
            arrowprops=dict(arrowstyle="->", color="#b0a090", lw=1.2, ls="--"),
        )
    for i, j in pairs[::3]:
        ax.annotate(
            "",
            xy=(right_ts[j], 0.45),
            xytext=(left_ts[i], 0.55),
            arrowprops=dict(arrowstyle="->", color=TEAL, lw=1.6),
        )
    ax.text(left_ts[0] - 8, 0.72, "左相机", color=WAVE, fontsize=11, va="center")
    ax.text(left_ts[0] - 8, 0.32, "右相机", color=TEAL, fontsize=11, va="center")
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xlabel("时间 (ms)")
    ax.legend(
        handles=[
            mpatches.Patch(color="#b0a090", label="按索引硬配（虚线，错）"),
            mpatches.Patch(color=TEAL, label="最近邻时间戳（实线）"),
        ],
        loc="upper right",
        frameon=False,
    )
    finish(ax)
    ax.axhline(0, visible=False)

    ax2 = axes[1]
    ax2.set_title("每对帧的时间误差 |t_right − t_left|")
    x = np.arange(n)
    ax2.bar(x - 0.18, bad_err, width=0.36, color="#c9b8a4", label="按索引硬配")
    ax2.bar(x + 0.18, good_err, width=0.36, color=TEAL, label="最近邻同步")
    ax2.set_xlabel("左帧序号")
    ax2.set_ylabel("误差 (ms)")
    ax2.legend(frameon=False)
    finish(ax2)
    save_fig(fig, "ch25_timestamp_sync.png")


def plot_h_smooth(noise_std=2.8, window=7, n=48, name="ch25_h_smooth.png", title_extra=""):
    """帧间 H（这里用平移 tx 代表）原始噪声 vs 滑动平均。"""
    rng = np.random.default_rng(21)
    true_tx = true_tx_sequence(n)
    obs = noisy_tx(true_tx, noise_std, rng)
    sm = moving_average(obs, window)

    fig, axes = plt.subplots(2, 1, figsize=(9.6, 5.4), layout="constrained", sharex=True)
    t = np.arange(n)
    ax = axes[0]
    ax.set_title(f"相对平移 tx（代表 H 的平移分量）{title_extra}")
    ax.plot(t, true_tx, color=MUTED, lw=1.4, label="真实缓慢变化")
    ax.plot(t, obs, color=WAVE, lw=1.0, alpha=0.85, label="逐帧独立估计（带噪）")
    ax.plot(t, sm, color=TEAL, lw=2.0, label=f"滑动平均窗口={window}")
    ax.set_ylabel("tx (px)")
    ax.legend(frameon=False, loc="upper right")
    finish(ax)

    ax2 = axes[1]
    ax2.set_title("帧间一阶差分 |Δtx|：越小画面越不「跳」")
    ax2.plot(t[1:], np.abs(np.diff(obs)), color=WAVE, lw=1.0, label="独立估计")
    ax2.plot(t[1:], np.abs(np.diff(sm)), color=TEAL, lw=1.8, label="平滑后")
    ax2.set_xlabel("帧号")
    ax2.set_ylabel("|Δtx| (px)")
    ax2.legend(frameon=False)
    finish(ax2)
    save_fig(fig, name)
    return true_tx, obs, sm


def plot_seam_dance(noise_std=2.8, seam_window=9, n=48, name="ch25_seam_dance.png", title_extra=""):
    """接缝横坐标随时间跳动 vs 时序稳定。"""
    rng = np.random.default_rng(33)
    # 理想接缝应跟重叠中心缓慢走；独立每帧按「代价最小」会抖
    base = 200 + 5 * np.sin(2 * np.pi * np.arange(n) / 30.0)
    # 模拟每帧重新求 seam 的噪声 + 偶发大跳（物体穿过）
    seam_raw = base + rng.normal(0, noise_std * 1.4, size=n)
    jump_at = n // 3
    seam_raw[jump_at : jump_at + 2] += 18  # 「缝跳舞」尖峰
    seam_sm = moving_average(seam_raw, seam_window)

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.0), layout="constrained")

    ax = axes[0]
    ax.set_title(f"接缝横坐标随时间{title_extra}")
    ax.plot(seam_raw, color=WAVE, lw=1.2, label="每帧独立求缝")
    ax.plot(seam_sm, color=TEAL, lw=2.0, label=f"时序平滑窗口={seam_window}")
    ax.plot(base, color=MUTED, lw=1.0, ls="--", label="理想缓慢轨迹")
    ax.set_xlabel("帧号")
    ax.set_ylabel("seam x (px)")
    ax.legend(frameon=False)
    finish(ax)

    # 右侧：把 seam 画在条带上（取若干帧）
    ax2 = axes[1]
    ax2.set_title("「缝在跳舞」示意（竖线位置）")
    h = 12
    for k, idx in enumerate([0, jump_at, jump_at + 1, n // 2, n - 1]):
        y0 = k * (h + 4)
        ax2.add_patch(Rectangle((0, y0), 320, h, facecolor="#f3e6d8", edgecolor="#2a241c", lw=0.6))
        ax2.plot([seam_raw[idx], seam_raw[idx]], [y0, y0 + h], color=WAVE, lw=2.2)
        ax2.plot([seam_sm[idx], seam_sm[idx]], [y0, y0 + h], color=TEAL, lw=2.2)
        ax2.text(-4, y0 + h / 2, f"t={idx}", ha="right", va="center", fontsize=9, color=MUTED)
    ax2.set_xlim(-40, 340)
    ax2.set_ylim(-2, 5 * (h + 4))
    ax2.axis("off")
    ax2.legend(
        handles=[
            mpatches.Patch(color=WAVE, label="独立缝"),
            mpatches.Patch(color=TEAL, label="平滑缝"),
        ],
        loc="lower right",
        frameon=False,
    )
    save_fig(fig, name)
    return seam_raw, seam_sm


def plot_flicker_compare():
    """闪烁：独立 H+独立缝 vs 平滑后，接缝附近亮度差随时间。"""
    scene = make_pano_strip()
    view_w = 220
    left_x0, right_x0 = 40, 200
    n = 40
    rng = np.random.default_rng(5)
    true_tx = true_tx_sequence(n, base=float(right_x0 - left_x0), amp=5.0)
    obs = noisy_tx(true_tx, 3.0, rng)
    sm = moving_average(obs, 7)

    # 理想重叠中心附近一条竖带；度量左/右贡献在接缝处的错位强度
    # 简化：用拼后图在固定列上的灰度相对上一帧的差
    canvas_w = 360
    seam = view_w - 40
    flicker_raw = []
    flicker_sm = []
    prev_raw = prev_sm = None
    for i in range(n):
        left, right = crop_pair(scene, left_x0, right_x0, view_w, scene.shape[0])
        # 曝光微抖：右图整体加减一点，模拟 AE 不一致 → 闪
        gain = 1.0 + 0.04 * np.sin(i * 0.9) + rng.normal(0, 0.01)
        right_g = np.clip(right.astype(np.float32) * gain, 0, 255).astype(np.uint8)
        raw, _ = stitch_with_tx(left, right_g, obs[i], canvas_w, seam)
        smo, _ = stitch_with_tx(left, right_g, sm[i], canvas_w, seam)
        band = slice(seam - 6, seam + 6)
        g_raw = cv2.cvtColor(raw[:, band], cv2.COLOR_BGR2GRAY).astype(np.float32)
        g_sm = cv2.cvtColor(smo[:, band], cv2.COLOR_BGR2GRAY).astype(np.float32)
        if prev_raw is not None:
            flicker_raw.append(float(np.mean(np.abs(g_raw - prev_raw))))
            flicker_sm.append(float(np.mean(np.abs(g_sm - prev_sm))))
        prev_raw, prev_sm = g_raw, g_sm

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.8), layout="constrained")
    ax = axes[0]
    ax.set_title("接缝邻域帧差（越小越不闪）")
    ax.plot(flicker_raw, color=WAVE, lw=1.2, label="独立 H")
    ax.plot(flicker_sm, color=TEAL, lw=1.8, label="平滑 H")
    ax.set_xlabel("帧号")
    ax.set_ylabel("平均 |ΔI|")
    ax.legend(frameon=False)
    finish(ax)

    # 右侧文字示意：闪烁来源
    ax2 = axes[1]
    ax2.axis("off")
    ax2.set_title("闪烁常见来源（白话）")
    lines = [
        "1. 每帧 H 抖 → 同一物体在拼缝两侧来回蹭",
        "2. 每帧重算接缝 → 缝左右跳（缝跳舞）",
        "3. 两路自动曝光不同步 → 接缝亮度一明一暗",
        "4. Rolling Shutter / 运动物体 → 局部对不齐",
        "",
        "最小对策：同步 → 平滑 H → 锁/平滑缝 → 曝光对齐",
    ]
    for i, line in enumerate(lines):
        ax2.text(0.02, 0.92 - i * 0.12, line, fontsize=12, transform=ax2.transAxes, color=INK)
    save_fig(fig, "ch25_flicker_compare.png")


def plot_pipeline():
    """最小可行做法流程图。"""
    fig, ax = plt.subplots(figsize=(10.4, 3.8), layout="constrained")
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4)
    ax.axis("off")
    ax.set_title("视频拼接最小可行流水线（时空一致性版）")

    boxes = [
        (0.3, 1.4, "多路视频\n+ 时间戳"),
        (2.5, 1.4, "时间同步\n最近邻/插值"),
        (4.7, 1.4, "估帧间 H\n(特征/光流)"),
        (6.9, 1.4, "时序平滑 H\n滑动/滤波"),
        (9.1, 1.4, "固定或平滑缝\n+ 融合输出"),
    ]
    for x, y, text in boxes:
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                1.9,
                1.5,
                boxstyle="round,pad=0.08,rounding_size=0.15",
                facecolor="#fde8df" if x < 5 else "#e5f3f1",
                edgecolor="#2a241c",
                lw=1.2,
            )
        )
        ax.text(x + 0.95, y + 0.75, text, ha="center", va="center", fontsize=11)

    for x0 in (2.2, 4.4, 6.6, 8.8):
        ax.add_patch(
            FancyArrowPatch(
                (x0, 2.15),
                (x0 + 0.28, 2.15),
                arrowstyle="->",
                mutation_scale=14,
                color=MUTED,
                lw=1.4,
            )
        )

    ax.text(
        0.3,
        0.55,
        "静态拼接流水线少了「同步」和两处「时序约束」。少它们 → 闪烁、缝跳舞。",
        fontsize=12,
        color=MUTED,
    )
    save_fig(fig, "ch25_pipeline.png")


def plot_difficulty_map():
    """难点地图：哪些已有最小做法，哪些仍难。"""
    fig, ax = plt.subplots(figsize=(9.8, 4.6), layout="constrained")
    ax.axis("off")
    ax.set_title("难点地图（本篇之后仍会反复遇到）")

    col_labels = ["问题", "最小可行做法", "仍难 / 进阶"]
    rows = [
        ["时间不同步", "硬件触发 / 最近邻对齐", "亚帧插值、时钟漂移"],
        ["H 帧间抖", "滑动平均 / 低通", "流形上滤波、与 IMU 融合"],
        ["缝跳舞", "锁缝或平滑缝坐标", "时空 graph-cut、物体感知"],
        ["曝光闪", "增益曲线对齐", "局部色调、HDR 时序"],
        ["Rolling Shutter", "行时间近似校正", "与拼接联合优化"],
        ["大视差 / 近景", "平面假设 + 裁切", "深度/网格、多平面"],
        ["运动物体跨缝", "缝避开高代价区", "跟踪切割、分层合成"],
    ]
    table = ax.table(cellText=rows, colLabels=col_labels, loc="center", cellLoc="left")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.15, 1.55)
    for (r, c), cell in table.get_celld().items():
        if r == 0:
            cell.set_facecolor("#f3e6d8")
            cell.set_text_props(weight="bold")
        elif c == 1:
            cell.set_facecolor("#e5f3f1")
        elif c == 2:
            cell.set_facecolor("#fde8df")
    save_fig(fig, "ch25_difficulty_map.png")


def plot_mine():
    """「改这里」参数驱动的对照图。"""
    plot_h_smooth(
        noise_std=H_NOISE_STD,
        window=H_SMOOTH_WINDOW,
        n=N_FRAMES,
        name="ch25_my_h_smooth.png",
        title_extra="（你的参数）",
    )
    plot_seam_dance(
        noise_std=H_NOISE_STD,
        seam_window=SEAM_SMOOTH_WINDOW,
        n=N_FRAMES,
        name="ch25_my_seam.png",
        title_extra="（你的参数）",
    )
    plot_timestamp_sync_mine()


def plot_timestamp_sync_mine():
    """用 TIME_OFFSET_MS 再画一张同步误差条形图。"""
    n = 16
    left_ts, right_ts = make_timestamps(n, fps=30.0, offset_ms=TIME_OFFSET_MS)
    bad = np.abs(right_ts - left_ts)
    _, good = sync_nearest(left_ts, right_ts)
    fig, ax = plt.subplots(figsize=(8.8, 3.4), layout="constrained")
    x = np.arange(n)
    ax.bar(x - 0.18, bad, width=0.36, color="#c9b8a4", label="按索引硬配")
    ax.bar(x + 0.18, good, width=0.36, color=TEAL, label="最近邻同步")
    ax.set_title(f"你的时间偏移 TIME_OFFSET_MS={TIME_OFFSET_MS:.1f}")
    ax.set_xlabel("左帧序号")
    ax.set_ylabel("误差 (ms)")
    ax.legend(frameon=False)
    finish(ax)
    save_fig(fig, "ch25_my_sync.png")


# ---------------------------------------------------------------------------
# 自检
# ---------------------------------------------------------------------------


def self_check():
    # 1) 时间同步：最近邻误差应明显小于「按索引硬配」
    left_ts, right_ts = make_timestamps(20, fps=30.0, offset_ms=33.0)
    pairs, errs = sync_nearest(left_ts, right_ts)
    assert len(pairs) == 20
    bad = np.abs(right_ts - left_ts)
    assert float(np.mean(errs)) < float(np.mean(bad)) * 0.5

    # 2) 小偏移时，最近邻应把误差压到半帧以内
    left2, right2 = make_timestamps(15, fps=30.0, offset_ms=10.0)
    _, errs2 = sync_nearest(left2, right2)
    half = (1000.0 / 30.0) / 2
    assert float(np.max(errs2)) <= half + 1e-6

    # 3) H 平滑后帧间差分更小
    rng = np.random.default_rng(0)
    true_tx = true_tx_sequence(40)
    obs = noisy_tx(true_tx, 3.0, rng)
    sm = moving_average(obs, 7)
    assert float(np.mean(np.abs(np.diff(sm)))) < float(np.mean(np.abs(np.diff(obs))))

    # 4) 接缝平滑抑制尖峰
    seam = np.ones(30) * 200.0
    seam[10] = 230.0
    seam_sm = moving_average(seam, 9)
    assert abs(seam_sm[10] - 200) < abs(seam[10] - 200)

    # 5) 硬接缝拼图尺寸正确
    a = np.zeros((40, 80, 3), dtype=np.uint8)
    b = np.full((40, 80, 3), 255, dtype=np.uint8)
    out, sx = hard_blend(a, b, 40)
    assert out.shape == (40, 80, 3)
    assert sx == 40
    assert out[0, 39, 0] == 0 and out[0, 40, 0] == 255

    print("自检通过：同步、H 平滑、接缝平滑、硬拼尺寸共 5 项。")


def main():
    use_style()
    self_check()
    plot_independent_fail()
    plot_timestamp_sync(offset_ms=33.0)
    plot_h_smooth()
    plot_seam_dance()
    plot_flicker_compare()
    plot_pipeline()
    plot_difficulty_map()
    plot_mine()
    print("课文对照图已写入 assets/figures/ 与 assets/images/")
    print("你的图：ch25_my_h_smooth.png / ch25_my_seam.png / ch25_my_sync.png")


if __name__ == "__main__":
    main()
