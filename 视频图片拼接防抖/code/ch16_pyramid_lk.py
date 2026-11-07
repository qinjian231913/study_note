"""第 16 篇：金字塔 LK 与特征点跟踪。

运行（在系列根目录）：

    MPLCONFIGDIR=/tmp/mpl-cache .venv/bin/python code/ch16_pyramid_lk.py

顶部「改这里」只影响 ch16_my_track.png。
"""

from pathlib import Path

SHOW_WINDOW = False

import matplotlib

if not SHOW_WINDOW:
    matplotlib.use("Agg")

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from plot_style import INK, MUTED, TEAL, WAVE, use_style

# ===== 改这里 =====
# 合成场景的真实平移（像素）。越大，单层 LK 越容易跟丢。
MY_TRUE_SHIFT = (24, -10)
# 金字塔层数：0=只有原图（单层）；2～3 通常够用
MY_MAX_LEVEL = 3
# LK 窗口边长（奇数更好）。太小怕噪声；太大怕窗口内运动不一致
MY_WIN_SIZE = 21
# 每层最多迭代次数
MY_MAX_ITERS = 30
# 前后向检验阈值（像素）。往返误差大于它就扔掉该点
MY_FB_THRESH = 1.5
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


def make_texture_scene(w=320, h=240, seed=7):
    """带角点的合成街景：方块、竖线、数字，方便跟踪。"""
    rng = np.random.default_rng(seed)
    img = np.full((h, w, 3), 236, dtype=np.uint8)
    # 远景色块
    for _ in range(18):
        x0 = int(rng.integers(0, w - 40))
        y0 = int(rng.integers(0, h - 40))
        ww = int(rng.integers(20, 55))
        hh = int(rng.integers(20, 45))
        color = tuple(int(c) for c in rng.integers(90, 200, size=3))
        cv2.rectangle(img, (x0, y0), (x0 + ww, y0 + hh), color, -1)
        cv2.rectangle(img, (x0, y0), (x0 + ww, y0 + hh), (40, 40, 40), 1)
    # 几条竖向路灯 / 杆
    for x in (40, 110, 180, 250):
        cv2.line(img, (x, 20), (x, h - 20), (60, 55, 50), 2)
        cv2.circle(img, (x, 28), 6, WAVE_BGR, -1)
    # 地面标线
    for x in range(10, w, 36):
        cv2.rectangle(img, (x, h - 28), (x + 18, h - 18), (220, 210, 80), -1)
    # 角落字母，强角点
    cv2.putText(img, "A", (18, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, WAVE_BGR, 2, cv2.LINE_AA)
    cv2.putText(img, "B", (w - 48, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, TEAL_BGR, 2, cv2.LINE_AA)
    cv2.putText(img, "C", (w - 48, h - 18), cv2.FONT_HERSHEY_SIMPLEX, 1.2, WAVE_BGR, 2, cv2.LINE_AA)
    cv2.putText(img, "D", (18, h - 18), cv2.FONT_HERSHEY_SIMPLEX, 1.2, TEAL_BGR, 2, cv2.LINE_AA)
    noise = rng.normal(0, 3.0, img.shape).astype(np.float32)
    img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return img


def shift_frame(img, dx, dy, border=None):
    """按真实平移造下一帧（平移矩阵）。"""
    if border is None:
        border = (251, 251, 251)
    h, w = img.shape[:2]
    m = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LINEAR, borderValue=border)


def to_gray(img):
    if img.ndim == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def pick_corners(gray, max_corners=40):
    pts = cv2.goodFeaturesToTrack(
        gray,
        maxCorners=max_corners,
        qualityLevel=0.01,
        minDistance=12,
        blockSize=7,
    )
    if pts is None:
        return np.zeros((0, 1, 2), dtype=np.float32)
    return pts.astype(np.float32)


def run_lk(prev_gray, next_gray, pts, win_size=21, max_level=3, max_iters=30):
    if pts is None or len(pts) == 0:
        empty = np.zeros((0, 1, 2), dtype=np.float32)
        return empty, np.zeros((0,), dtype=np.uint8), np.zeros((0, 1), dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, int(max_iters), 0.01)
    next_pts, status, err = cv2.calcOpticalFlowPyrLK(
        prev_gray,
        next_gray,
        pts,
        None,
        winSize=(int(win_size), int(win_size)),
        maxLevel=int(max_level),
        criteria=criteria,
        flags=0,
    )
    if next_pts is None:
        next_pts = np.zeros_like(pts)
        status = np.zeros((len(pts), 1), dtype=np.uint8)
        err = np.zeros((len(pts), 1), dtype=np.float32)
    return next_pts, status.reshape(-1), err.reshape(-1)


def forward_backward_ok(prev_gray, next_gray, pts0, pts1, status, fb_thresh, win_size, max_level, max_iters):
    """前后向检验：A→B 再 B→A，往返误差过大则不可靠。"""
    ok = status.astype(bool).copy()
    if not np.any(ok):
        return ok, np.full(len(pts0), np.nan, dtype=np.float32)

    back_pts, st_back, _ = run_lk(
        next_gray, prev_gray, pts1[ok].reshape(-1, 1, 2), win_size, max_level, max_iters
    )
    fb_err = np.full(len(pts0), np.nan, dtype=np.float32)
    idx = np.where(ok)[0]
    for j, i in enumerate(idx):
        if st_back[j] == 0:
            ok[i] = False
            fb_err[i] = 1e9
            continue
        d = np.linalg.norm(pts0[i, 0] - back_pts[j, 0])
        fb_err[i] = d
        if d > fb_thresh:
            ok[i] = False
    return ok, fb_err


def draw_tracks_on(img, pts0, pts1, mask, color_ok=TEAL_BGR, color_bad=WAVE_BGR):
    vis = img.copy()
    for i, good in enumerate(mask):
        p0 = tuple(np.round(pts0[i, 0]).astype(int))
        p1 = tuple(np.round(pts1[i, 0]).astype(int))
        col = color_ok if good else color_bad
        cv2.circle(vis, p0, 3, WAVE_BGR, -1, lineType=cv2.LINE_AA)
        if good or True:
            cv2.arrowedLine(vis, p0, p1, col, 1, tipLength=0.25, line_type=cv2.LINE_AA)
            cv2.circle(vis, p1, 3, col, -1, lineType=cv2.LINE_AA)
    return vis


def plot_large_motion_problem():
    """单层 LK 在大位移下失败，金字塔能跟住。"""
    scene = make_texture_scene()
    dx, dy = 28, -12
    frame1 = shift_frame(scene, dx, dy)
    g0, g1 = to_gray(scene), to_gray(frame1)
    pts = pick_corners(g0, max_corners=30)

    pts_s, st_s, _ = run_lk(g0, g1, pts, win_size=21, max_level=0, max_iters=30)
    pts_p, st_p, _ = run_lk(g0, g1, pts, win_size=21, max_level=3, max_iters=30)

    true = np.array([dx, dy], dtype=np.float32)
    err_s = np.linalg.norm(pts_s[:, 0] - (pts[:, 0] + true), axis=1)
    err_p = np.linalg.norm(pts_p[:, 0] - (pts[:, 0] + true), axis=1)
    ok_s = (st_s == 1) & (err_s < 2.0)
    ok_p = (st_p == 1) & (err_p < 2.0)

    vis_s = draw_tracks_on(frame1, pts, pts_s, ok_s)
    vis_p = draw_tracks_on(frame1, pts, pts_p, ok_p)
    cv2.putText(vis_s, f"single ok={ok_s.sum()}/{len(pts)}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, INK_BGR, 1, cv2.LINE_AA)
    cv2.putText(vis_p, f"pyramid ok={ok_p.sum()}/{len(pts)}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, INK_BGR, 1, cv2.LINE_AA)

    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.8), layout="constrained")
    axes[0].imshow(scene[:, :, ::-1])
    axes[0].set_title(f"帧 0（真实位移 ({dx},{dy}) px）")
    axes[0].axis("off")
    axes[1].imshow(vis_s[:, :, ::-1])
    axes[1].set_title("单层 LK（maxLevel=0）多跟丢")
    axes[1].axis("off")
    axes[2].imshow(vis_p[:, :, ::-1])
    axes[2].set_title("金字塔 LK（maxLevel=3）")
    axes[2].axis("off")
    save_fig(fig, "ch16_large_motion.png")
    save_bgr("ch16_large_motion_single.png", vis_s)
    save_bgr("ch16_large_motion_pyr.png", vis_p)


def plot_pyramid_idea():
    """画出图像金字塔：由粗到细。"""
    scene = make_texture_scene(w=280, h=200)
    gray = to_gray(scene)
    levels = [gray]
    for _ in range(3):
        levels.append(cv2.pyrDown(levels[-1]))

    fig, axes = plt.subplots(1, 4, figsize=(11.0, 3.2), layout="constrained")
    for i, ax in enumerate(axes):
        ax.imshow(levels[i], cmap="gray", vmin=0, vmax=255)
        h, w = levels[i].shape
        ax.set_title(f"L{i}  {w}×{h}")
        ax.axis("off")
        # 在最粗层画一个「大步」箭头示意
        if i == 3:
            ax.annotate(
                "",
                xy=(w * 0.62, h * 0.45),
                xytext=(w * 0.35, h * 0.55),
                arrowprops=dict(arrowstyle="->", color=WAVE, lw=2),
            )
            ax.text(0.05, 0.08, "先估大步", transform=ax.transAxes, color=WAVE, fontsize=10)
        if i == 0:
            ax.text(0.05, 0.08, "最后精炼", transform=ax.transAxes, color=TEAL, fontsize=10)
    save_fig(fig, "ch16_pyramid_levels.png")

    # 流程示意
    fig, ax = plt.subplots(figsize=(9.4, 3.4), layout="constrained")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.axis("off")
    ax.set_title("由粗到细：每下一层，位移大约 ×2 再加本层残差")
    boxes = [
        (0.4, 2.2, "最粗层\n估 (u,v)"),
        (3.0, 2.2, "放大约 ×2\n作初值"),
        (5.6, 2.2, "更细层\n只修残差"),
        (8.2, 2.2, "原图层\n精炼到亚像素"),
    ]
    for x, y, text in boxes:
        ax.add_patch(Rectangle((x, y), 1.6, 1.2, facecolor="#fde8df" if x < 4 else "#e5f3f1", edgecolor=INK, lw=1.2))
        ax.text(x + 0.8, y + 0.6, text, ha="center", va="center", fontsize=11)
    for x0 in (2.1, 4.7, 7.3):
        ax.annotate("", xy=(x0 + 0.7, 2.8), xytext=(x0, 2.8), arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.6))
    ax.text(0.4, 0.7, "白话：先站在山顶看清「大概往哪移」，再一层层走近把脚印踩准。", fontsize=12, color=MUTED)
    save_fig(fig, "ch16_coarse_to_fine.png")


def plot_tracking_pipeline():
    """多帧点跟踪：选点 → 跟踪 → 丢点补点。"""
    scene0 = make_texture_scene(seed=3)
    # 模拟手持轻微晃动 + 缓慢平移
    shifts = [(0, 0), (4, -2), (9, -3), (15, -1), (22, 2), (28, 5)]
    frames = [shift_frame(scene0, dx, dy) for dx, dy in shifts]
    grays = [to_gray(f) for f in frames]

    pts = pick_corners(grays[0], max_corners=25)
    tracks = [pts.copy()]
    alive = np.ones(len(pts), dtype=bool)

    for t in range(1, len(frames)):
        nxt, st, _ = run_lk(grays[t - 1], grays[t], pts, win_size=21, max_level=3, max_iters=30)
        ok, _ = forward_backward_ok(
            grays[t - 1], grays[t], pts, nxt, st, fb_thresh=1.5, win_size=21, max_level=3, max_iters=30
        )
        # 出界也算丢
        h, w = grays[t].shape
        inside = (
            (nxt[:, 0, 0] > 4)
            & (nxt[:, 0, 0] < w - 4)
            & (nxt[:, 0, 1] > 4)
            & (nxt[:, 0, 1] < h - 4)
        )
        ok = ok & inside
        alive = alive & ok
        pts = nxt.copy()
        tracks.append(pts.copy())

        # 丢太多就补点
        if alive.sum() < 12:
            fresh = pick_corners(grays[t], max_corners=20)
            # 简单：用新点替换已死槽位
            dead = np.where(~alive)[0]
            n_add = min(len(dead), len(fresh))
            for k in range(n_add):
                pts[dead[k]] = fresh[k]
                alive[dead[k]] = True
                # 新轨迹从当前帧开始
                for hist in tracks:
                    hist[dead[k]] = fresh[k]

    # 画轨迹到最后一帧
    canvas = frames[-1].copy()
    colors = [(40, 77, 194), (102, 107, 14), (60, 60, 60)]
    for i in range(len(alive)):
        path = [tracks[t][i, 0] for t in range(len(tracks))]
        for t in range(1, len(path)):
            p0 = tuple(np.round(path[t - 1]).astype(int))
            p1 = tuple(np.round(path[t]).astype(int))
            cv2.line(canvas, p0, p1, colors[i % 3], 1, cv2.LINE_AA)
        p_end = tuple(np.round(path[-1]).astype(int))
        cv2.circle(canvas, p_end, 3, TEAL_BGR if alive[i] else WAVE_BGR, -1)

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0), layout="constrained")
    axes[0].imshow(frames[0][:, :, ::-1])
    axes[0].scatter(tracks[0][:, 0, 0], tracks[0][:, 0, 1], s=18, c=WAVE, zorder=3)
    axes[0].set_title("第 0 帧：goodFeatures 选点")
    axes[0].axis("off")
    axes[1].imshow(canvas[:, :, ::-1])
    axes[1].set_title("多帧跟踪轨迹（青=仍活，橙圈=补点/异常）")
    axes[1].axis("off")
    save_fig(fig, "ch16_track_pipeline.png")
    save_bgr("ch16_track_paths.png", canvas)

    # 流水线文字图
    fig, ax = plt.subplots(figsize=(9.6, 2.8), layout="constrained")
    ax.axis("off")
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 3)
    ax.set_title("视频点跟踪流水线")
    steps = ["选角点", "PyrLK\n跟到下一帧", "前后向检验", "丢点 / 出界", "补新点", "交给防抖"]
    for i, s in enumerate(steps):
        x = 0.3 + i * 2.0
        ax.add_patch(Rectangle((x, 0.9), 1.6, 1.3, facecolor="#fffdf8", edgecolor=TEAL if i < 5 else WAVE, lw=1.4))
        ax.text(x + 0.8, 1.55, s, ha="center", va="center", fontsize=11)
        if i < len(steps) - 1:
            ax.annotate("", xy=(x + 1.85, 1.55), xytext=(x + 1.65, 1.55), arrowprops=dict(arrowstyle="->", color=MUTED))
    save_fig(fig, "ch16_pipeline_steps.png")


def plot_forward_backward():
    """前后向检验：可靠点往返重合，不可靠点往返偏大。"""
    scene = make_texture_scene(seed=11)
    # 大位移 + 故意在一侧加局部干扰（模拟遮挡/模糊）
    frame1 = shift_frame(scene, 18, -8)
    h, w = frame1.shape[:2]
    # 右半边糊掉，制造不可靠区
    blur = cv2.GaussianBlur(frame1, (21, 21), 0)
    frame1[:, w // 2 :] = blur[:, w // 2 :]

    g0, g1 = to_gray(scene), to_gray(frame1)
    pts = pick_corners(g0, max_corners=50)
    pts1, st, _ = run_lk(g0, g1, pts, win_size=21, max_level=3, max_iters=30)
    ok_fb, fb_err = forward_backward_ok(
        g0, g1, pts, pts1, st, fb_thresh=1.5, win_size=21, max_level=3, max_iters=30
    )

    vis = frame1.copy()
    for i in range(len(pts)):
        p0 = tuple(np.round(pts[i, 0]).astype(int))
        p1 = tuple(np.round(pts1[i, 0]).astype(int))
        if st[i] == 0:
            continue
        col = TEAL_BGR if ok_fb[i] else WAVE_BGR
        cv2.arrowedLine(vis, p0, p1, col, 1, tipLength=0.3, line_type=cv2.LINE_AA)
        cv2.circle(vis, p1, 4, col, -1, lineType=cv2.LINE_AA)
    cv2.line(vis, (w // 2, 0), (w // 2, h), (80, 80, 80), 1)
    cv2.putText(vis, "sharp", (20, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, TEAL_BGR, 1, cv2.LINE_AA)
    cv2.putText(vis, "blurred", (w // 2 + 12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, WAVE_BGR, 1, cv2.LINE_AA)

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.0), layout="constrained")
    axes[0].imshow(vis[:, :, ::-1])
    axes[0].set_title("青=通过 FB；橙=往返误差大（多在模糊侧）")
    axes[0].axis("off")

    # 误差直方图
    valid = np.isfinite(fb_err) & (fb_err < 1e8)
    axes[1].hist(fb_err[valid], bins=20, color=TEAL, edgecolor=INK, alpha=0.85)
    axes[1].axvline(1.5, color=WAVE, lw=2, label="阈值 1.5 px")
    axes[1].set_xlabel("往返误差（像素）")
    axes[1].set_ylabel("点数")
    axes[1].set_title("前后向误差分布")
    axes[1].legend(frameon=False)
    save_fig(fig, "ch16_forward_backward.png")
    save_bgr("ch16_fb_overlay.png", vis)


def plot_param_compare():
    """对照层数 / 窗口 / 迭代对跟踪误差的影响。"""
    scene = make_texture_scene(seed=5)
    dx, dy = 26, -11
    frame1 = shift_frame(scene, dx, dy)
    g0, g1 = to_gray(scene), to_gray(frame1)
    pts = pick_corners(g0, max_corners=35)
    true = np.array([dx, dy], dtype=np.float32)

    def mean_err(max_level, win, iters):
        nxt, st, _ = run_lk(g0, g1, pts, win_size=win, max_level=max_level, max_iters=iters)
        mask = st == 1
        if mask.sum() == 0:
            return 99.0, 0
        e = np.linalg.norm(nxt[mask, 0] - (pts[mask, 0] + true), axis=1)
        # 也算「跟住」比例：误差 < 2px
        good = (e < 2.0).sum()
        return float(np.median(e)), int(good)

    configs = [
        ("层数 0", 0, 21, 30),
        ("层数 1", 1, 21, 30),
        ("层数 3", 3, 21, 30),
        ("窗 11", 3, 11, 30),
        ("窗 31", 3, 31, 30),
        ("迭代 5", 3, 21, 5),
        ("迭代 30", 3, 21, 30),
    ]
    labels = []
    meds = []
    goods = []
    for name, lv, win, it in configs:
        m, g = mean_err(lv, win, it)
        labels.append(name)
        meds.append(m)
        goods.append(g)

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.0), layout="constrained")
    x = np.arange(len(labels))
    axes[0].bar(x, meds, color=[WAVE if m > 2 else TEAL for m in meds], edgecolor=INK)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=25, ha="right")
    axes[0].set_ylabel("中位误差（px）")
    axes[0].set_title(f"真实位移 ({dx},{dy})：参数对照")
    axes[0].axhline(2.0, color=MUTED, ls="--", lw=1)

    axes[1].bar(x, goods, color=TEAL, edgecolor=INK)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=25, ha="right")
    axes[1].set_ylabel(f"误差<2px 的点数 / {len(pts)}")
    axes[1].set_title("跟住的点越多越好")
    save_fig(fig, "ch16_param_compare.png")


def plot_for_stabilization():
    """为防抖准备：稳定跟踪点构成帧间运动线索。"""
    scene = make_texture_scene(seed=9)
    # 模拟相机抖动：每帧微小随机平移
    rng = np.random.default_rng(0)
    n = 8
    centers = []
    cx, cy = 0.0, 0.0
    frames = []
    for i in range(n):
        cx += float(rng.normal(0, 3.5))
        cy += float(rng.normal(0, 3.0))
        centers.append((cx, cy))
        frames.append(shift_frame(scene, cx, cy))

    grays = [to_gray(f) for f in frames]
    pts = pick_corners(grays[0], max_corners=40)
    # 估计每帧相对第 0 帧的中位平移（用跟踪点）
    est = [(0.0, 0.0)]
    cur = pts.copy()
    for t in range(1, n):
        nxt, st, _ = run_lk(grays[t - 1], grays[t], cur, win_size=21, max_level=3, max_iters=30)
        ok, _ = forward_backward_ok(
            grays[t - 1], grays[t], cur, nxt, st, fb_thresh=1.5, win_size=21, max_level=3, max_iters=30
        )
        if ok.sum() >= 5:
            delta = (nxt[ok, 0] - cur[ok, 0]).mean(axis=0)
        else:
            delta = np.array([0.0, 0.0])
        est.append((est[-1][0] + float(delta[0]), est[-1][1] + float(delta[1])))
        cur = nxt

    true_xy = np.array(centers)
    est_xy = np.array(est)

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.0), layout="constrained")
    # 左：最后一帧上的存活点
    last = frames[-1].copy()
    for p in cur[:, 0]:
        cv2.circle(last, (int(p[0]), int(p[1])), 3, TEAL_BGR, -1, lineType=cv2.LINE_AA)
    axes[0].imshow(last[:, :, ::-1])
    axes[0].set_title("稳定跟踪点（喂给第 19 篇估变换）")
    axes[0].axis("off")

    axes[1].plot(true_xy[:, 0], true_xy[:, 1], "-o", color=WAVE, label="真实取景抖动")
    axes[1].plot(est_xy[:, 0], est_xy[:, 1], "-s", color=TEAL, label="由跟踪点估计")
    axes[1].set_aspect("equal")
    axes[1].set_xlabel("水平位移 px")
    axes[1].set_ylabel("竖直位移 px")
    axes[1].set_title("点跟踪 → 帧间运动线索")
    axes[1].legend(frameon=False)
    axes[1].grid(True, alpha=0.4)
    save_fig(fig, "ch16_for_stabilization.png")


def plot_mine():
    """跟随「改这里」参数，生成个人对照图。"""
    scene = make_texture_scene(seed=16)
    dx, dy = MY_TRUE_SHIFT
    frame1 = shift_frame(scene, dx, dy)
    g0, g1 = to_gray(scene), to_gray(frame1)
    pts = pick_corners(g0, max_corners=40)
    pts1, st, _ = run_lk(
        g0,
        g1,
        pts,
        win_size=MY_WIN_SIZE,
        max_level=MY_MAX_LEVEL,
        max_iters=MY_MAX_ITERS,
    )
    ok, fb_err = forward_backward_ok(
        g0,
        g1,
        pts,
        pts1,
        st,
        fb_thresh=MY_FB_THRESH,
        win_size=MY_WIN_SIZE,
        max_level=MY_MAX_LEVEL,
        max_iters=MY_MAX_ITERS,
    )
    true = np.array([dx, dy], dtype=np.float32)
    err = np.linalg.norm(pts1[:, 0] - (pts[:, 0] + true), axis=1)
    accurate = ok & (err < 2.0)

    vis = draw_tracks_on(frame1, pts, pts1, accurate)
    n_ok = int(accurate.sum())
    med = float(np.median(err[st == 1])) if (st == 1).any() else float("nan")
    title = (
        f"shift={MY_TRUE_SHIFT}  level={MY_MAX_LEVEL}  win={MY_WIN_SIZE}  "
        f"iters={MY_MAX_ITERS}  fb={MY_FB_THRESH}  ok={n_ok}/{len(pts)}  med_err={med:.2f}"
    )

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.0), layout="constrained")
    axes[0].imshow(scene[:, :, ::-1])
    axes[0].scatter(pts[:, 0, 0], pts[:, 0, 1], s=16, c=WAVE)
    axes[0].set_title("帧 0 角点")
    axes[0].axis("off")
    axes[1].imshow(vis[:, :, ::-1])
    axes[1].set_title("你的参数：青=通过 FB 且误差<2")
    axes[1].axis("off")
    fig.suptitle(title, fontsize=11)
    save_fig(fig, "ch16_my_track.png")


def self_check():
    scene = make_texture_scene(w=200, h=150, seed=1)
    # 小位移：单层也应大致能跟上
    f1 = shift_frame(scene, 3, -2)
    g0, g1 = to_gray(scene), to_gray(f1)
    pts = pick_corners(g0, max_corners=20)
    assert len(pts) >= 5, "角点太少，合成场景异常"

    nxt, st, _ = run_lk(g0, g1, pts, win_size=15, max_level=0, max_iters=20)
    assert st.sum() >= 3, "小位移单层 LK 也应跟踪到若干点"
    err = np.linalg.norm(nxt[st == 1, 0] - (pts[st == 1, 0] + np.array([3.0, -2.0])), axis=1)
    assert np.median(err) < 1.5, f"小位移误差过大：{np.median(err)}"

    # 大位移：金字塔应明显好于单层
    f2 = shift_frame(scene, 22, -9)
    g2 = to_gray(f2)
    true = np.array([22.0, -9.0])
    n_s, st_s, _ = run_lk(g0, g2, pts, win_size=21, max_level=0, max_iters=30)
    n_p, st_p, _ = run_lk(g0, g2, pts, win_size=21, max_level=3, max_iters=30)
    e_p = np.linalg.norm(n_p[st_p == 1, 0] - (pts[st_p == 1, 0] + true), axis=1) if st_p.sum() else np.array([99.0])
    good_s = int(((st_s == 1) & (np.linalg.norm(n_s[:, 0] - (pts[:, 0] + true), axis=1) < 2.5)).sum())
    good_p = int(((st_p == 1) & (np.linalg.norm(n_p[:, 0] - (pts[:, 0] + true), axis=1) < 2.5)).sum())
    assert good_p >= good_s, f"金字塔应不少于单层跟住点数：pyr={good_p}, single={good_s}"
    assert np.median(e_p) < 3.0, f"金字塔中位误差应较小：{np.median(e_p)}"

    # 前后向检验能跑
    ok, fb = forward_backward_ok(g0, g2, pts, n_p, st_p, 1.5, 21, 3, 30)
    assert ok.dtype == bool and len(ok) == len(pts)
    assert np.isfinite(fb[st_p == 1]).any()

    # pyrDown 尺寸减半
    small = cv2.pyrDown(g0)
    assert small.shape[0] == (g0.shape[0] + 1) // 2
    assert small.shape[1] == (g0.shape[1] + 1) // 2

    print("自检通过：小位移、大位移金字塔、前后向检验均正常。")


def main():
    use_style()
    self_check()
    plot_large_motion_problem()
    plot_pyramid_idea()
    plot_tracking_pipeline()
    plot_forward_backward()
    plot_param_compare()
    plot_for_stabilization()
    plot_mine()
    print("课文对照图已写入 assets/figures/ 与 assets/images/")
    print("你的图：assets/figures/ch16_my_track.png")


if __name__ == "__main__":
    main()
