"""第 19 篇：二维视频防抖——轨迹平滑。

在「视频图片拼接防抖」文件夹里运行：

    MPLCONFIGDIR=/tmp/mpl-cache .venv/bin/python code/ch19_stabilize_2d.py

流程：估帧间相似变换 → 累积成相机路径 → 滑动窗口平滑
→ 用「原路径相对平滑路径的差」做补偿 warp。

顶部「改这里」只影响 ch19_my_*.png；课文对照图按固定参数重画。
"""

from pathlib import Path

SHOW_WINDOW = False

import matplotlib

if not SHOW_WINDOW:
    matplotlib.use("Agg")

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

from plot_style import MUTED, TEAL, WAVE, finish, use_style

WAVE_BGR = (40, 77, 194)
TEAL_BGR = (102, 107, 14)
INK_BGR = (21, 25, 28)

# ===== 改这里 =====
# 只影响 ch19_my_path.png / ch19_my_strip.png / ch19_my_crop.png
MY_SMOOTH_RADIUS = 7  # 滑动窗口半宽。越大越稳，但过大会「漂」并吃掉视野
MY_SHAKE_AMP = 10.0  # 合成晃动幅度（像素）。改大更抖
MY_PAN_SPEED = 1.6  # 故意运镜的水平速度（像素/帧）。平滑不应把它抹掉
# ==================

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "assets" / "figures"
IMG = ROOT / "assets" / "images"

N_FRAMES = 60
FRAME_W, FRAME_H = 320, 240


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


def make_scene(width=720, height=420):
    """带楼、路、树、网格的合成大场景，方便跟踪与看出晃动。"""
    img = np.full((height, width, 3), 245, dtype=np.uint8)
    img[: height // 2] = (236, 228, 214)
    img[height // 2 :] = (210, 198, 178)

    hills = np.array(
        [[0, 200], [140, 120], [300, 170], [460, 100], [720, 185], [720, 200], [0, 200]],
        np.int32,
    )
    cv2.fillPoly(img, [hills], (180, 168, 150))

    for x, w, h, color in (
        (60, 50, 150, (90, 110, 150)),
        (150, 42, 120, (70, 95, 130)),
        (280, 58, 170, (100, 90, 70)),
        (400, 46, 135, (80, 120, 110)),
        (520, 54, 155, (110, 85, 95)),
        (620, 40, 125, (95, 100, 120)),
    ):
        top = height // 2 - h
        cv2.rectangle(img, (x, top), (x + w, height // 2), color, -1)
        cv2.rectangle(img, (x, top), (x + w, height // 2), (40, 40, 40), 1)
        for wy in range(top + 12, height // 2 - 8, 22):
            for wx in range(x + 8, x + w - 8, 16):
                cv2.rectangle(img, (wx, wy), (wx + 8, wy + 12), (200, 210, 220), -1)

    y = height // 2 + 48
    for x0 in range(20, width - 20, 48):
        cv2.line(img, (x0, y), (x0 + 28, y), (250, 250, 250), 3)

    cv2.rectangle(img, (230, 230), (244, 340), (60, 80, 40), -1)
    cv2.circle(img, (237, 205), 42, (50, 120, 70), -1)
    cv2.rectangle(img, (560, 250), (572, 340), (55, 75, 35), -1)
    cv2.circle(img, (566, 230), 32, (45, 110, 65), -1)

    for gx in range(30, width, 70):
        for gy in range(30, height, 60):
            cv2.circle(img, (gx, gy), 2, (150, 130, 110), -1)

    cv2.putText(img, "CITY", (300, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (90, 80, 70), 2, cv2.LINE_AA)
    return img


def identity_sim():
    return np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)


def sim_matrix(dx, dy, da_rad, scale=1.0):
    """相似变换 2×3：先旋转缩放（绕原点），再平移。"""
    c, s = np.cos(da_rad) * scale, np.sin(da_rad) * scale
    return np.array([[c, -s, dx], [s, c, dy]], dtype=np.float64)


def extract_dx_dy_da(m):
    """从 2×3 相似/近似相似矩阵抽出平移与转角（弧度）。"""
    dx, dy = float(m[0, 2]), float(m[1, 2])
    da = float(np.arctan2(m[1, 0], m[0, 0]))
    return dx, dy, da


def moving_average(arr, radius):
    """两端反射填充的滑动平均。radius=0 时原样返回。"""
    arr = np.asarray(arr, dtype=np.float64)
    if radius <= 0:
        return arr.copy()
    k = 2 * radius + 1
    pad = np.pad(arr, (radius, radius), mode="reflect")
    kernel = np.ones(k, dtype=np.float64) / k
    return np.convolve(pad, kernel, mode="valid")


def synthesize_transforms(n, shake_amp=10.0, pan_speed=1.6, rot_amp_deg=1.2, seed=7):
    """故意运镜（慢平移）+ 高频抖动 + 微旋转，作为「真实」帧间变换。"""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, n - 1)
    dx = pan_speed + shake_amp * 0.55 * np.sin(2 * np.pi * 6.5 * t)
    dx += shake_amp * 0.35 * rng.normal(0, 1, n - 1)
    dy = 0.15 + shake_amp * 0.45 * np.sin(2 * np.pi * 5.0 * t + 0.6)
    dy += shake_amp * 0.30 * rng.normal(0, 1, n - 1)
    da = np.deg2rad(rot_amp_deg) * (
        0.7 * np.sin(2 * np.pi * 4.2 * t) + 0.3 * rng.normal(0, 1, n - 1)
    )
    return dx, dy, da


def accumulate_path(dx, dy, da):
    """把帧间增量累积成相对第 0 帧的路径。"""
    x = np.concatenate([[0.0], np.cumsum(dx)])
    y = np.concatenate([[0.0], np.cumsum(dy)])
    a = np.concatenate([[0.0], np.cumsum(da)])
    return x, y, a


def render_frames(scene, path_x, path_y, path_a, w=FRAME_W, h=FRAME_H):
    """按累积路径从大场景里 warp 出视频帧（模拟手持取景）。"""
    sh, sw = scene.shape[:2]
    cx0, cy0 = sw * 0.28, sh * 0.42
    frames = []
    for x, y, a in zip(path_x, path_y, path_a):
        # warpAffine 的 M：输出坐标 → 输入（场景）坐标
        # 输出帧中心对应场景点 (cx0+x, cy0+y)，并带微旋转
        c, s = np.cos(a), np.sin(a)
        m_out_to_scene = np.array(
            [
                [c, -s, (cx0 + x) - c * (w / 2) + s * (h / 2)],
                [s, c, (cy0 + y) - s * (w / 2) - c * (h / 2)],
            ],
            dtype=np.float64,
        )
        frame = cv2.warpAffine(
            scene,
            m_out_to_scene,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(251, 251, 251),
        )
        frames.append(frame)
    return frames


def estimate_pairwise(frames, use_affine=False):
    """用光流点 + estimateAffinePartial2D / estimateAffine2D 估帧间变换。"""
    prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
    dx_list, dy_list, da_list = [], [], []
    mats = []
    for i in range(1, len(frames)):
        gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
        pts = cv2.goodFeaturesToTrack(
            prev_gray,
            maxCorners=200,
            qualityLevel=0.01,
            minDistance=8,
            blockSize=7,
        )
        if pts is None or len(pts) < 8:
            m = identity_sim()
        else:
            nxt, st, _ = cv2.calcOpticalFlowPyrLK(
                prev_gray,
                gray,
                pts,
                None,
                winSize=(21, 21),
                maxLevel=3,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
            )
            ok = st.reshape(-1) == 1
            p0, p1 = pts[ok], nxt[ok]
            if len(p0) < 6:
                m = identity_sim()
            elif use_affine:
                m, _ = cv2.estimateAffine2D(p0, p1, method=cv2.RANSAC, ransacReprojThreshold=3.0)
                if m is None:
                    m = identity_sim()
            else:
                m, _ = cv2.estimateAffinePartial2D(
                    p0, p1, method=cv2.RANSAC, ransacReprojThreshold=3.0
                )
                if m is None:
                    m = identity_sim()
        dx, dy, da = extract_dx_dy_da(m)
        dx_list.append(dx)
        dy_list.append(dy)
        da_list.append(da)
        mats.append(m)
        prev_gray = gray
    return (
        np.array(dx_list, dtype=np.float64),
        np.array(dy_list, dtype=np.float64),
        np.array(da_list, dtype=np.float64),
        mats,
    )


def stabilize_frames(frames, path_x, path_y, path_a, smooth_x, smooth_y, smooth_a):
    """补偿 = 平滑路径 − 原路径；对每帧做 warpAffine。"""
    out = []
    h, w = frames[0].shape[:2]
    for i, fr in enumerate(frames):
        ddx = smooth_x[i] - path_x[i]
        ddy = smooth_y[i] - path_y[i]
        dda = smooth_a[i] - path_a[i]
        m = sim_matrix(ddx, ddy, dda)
        warped = cv2.warpAffine(
            fr,
            m,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(40, 40, 40),
        )
        out.append(warped)
    return out


def crop_stable(frames, margin):
    """丢掉黑边：四周裁掉 margin 像素，再缩回原尺寸便于对比。"""
    h, w = frames[0].shape[:2]
    m = int(margin)
    if m <= 0 or 2 * m >= min(h, w):
        return [f.copy() for f in frames]
    cropped = []
    for fr in frames:
        piece = fr[m : h - m, m : w - m]
        cropped.append(cv2.resize(piece, (w, h), interpolation=cv2.INTER_LINEAR))
    return cropped


def estimate_needed_margin(path_x, path_y, path_a, smooth_x, smooth_y, smooth_a, w, h):
    """根据补偿量粗估需要裁掉多少边，才能躲开黑边。"""
    corners = np.array(
        [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float64
    )
    max_in = 0.0
    for i in range(len(path_x)):
        ddx = smooth_x[i] - path_x[i]
        ddy = smooth_y[i] - path_y[i]
        dda = smooth_a[i] - path_a[i]
        m = sim_matrix(ddx, ddy, dda)
        ones = np.ones((4, 1))
        pts = np.hstack([corners, ones])
        mapped = (m @ pts.T).T
        for (x, y) in mapped:
            max_in = max(max_in, -x, x - (w - 1), -y, y - (h - 1))
    return int(np.ceil(max(0.0, max_in))) + 2


def make_strip(frames, indices, labels=None):
    gap = 6
    picks = [frames[i].copy() for i in indices]
    for i, fr in enumerate(picks):
        tag = labels[i] if labels else f"t={indices[i]}"
        cv2.putText(fr, tag, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, WAVE_BGR, 2, cv2.LINE_AA)
    h, w = picks[0].shape[:2]
    canvas = np.full((h, len(picks) * w + (len(picks) - 1) * gap, 3), 251, dtype=np.uint8)
    for i, fr in enumerate(picks):
        x0 = i * (w + gap)
        canvas[:, x0 : x0 + w] = fr
    return canvas


def plot_pipeline():
    fig, ax = plt.subplots(figsize=(9.6, 2.8), layout="constrained")
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 3)
    ax.axis("off")
    ax.set_title("二维稳像四步（本篇主线）")
    steps = [
        (0.3, "1 估帧间\n相似/仿射"),
        (3.2, "2 累积成\n相机路径"),
        (6.1, "3 滑动窗口\n平滑路径"),
        (9.0, "4 路径差\n补偿 warp"),
    ]
    for x, text in steps:
        box = FancyBboxPatch(
            (x, 0.7),
            2.4,
            1.6,
            boxstyle="round,pad=0.15,rounding_size=0.2",
            facecolor="#fde8df" if x < 5 else "#e5f3f1",
            edgecolor="#1c1915",
            lw=1.4,
        )
        ax.add_patch(box)
        ax.text(x + 1.2, 1.5, text, ha="center", va="center", fontsize=12)
    for x0 in (2.7, 5.6, 8.5):
        ax.add_patch(
            FancyArrowPatch(
                (x0, 1.5),
                (x0 + 0.45, 1.5),
                arrowstyle="-|>",
                mutation_scale=12,
                color=MUTED,
                lw=1.5,
            )
        )
    save_fig(fig, "ch19_pipeline.png")


def plot_path_compare(path_x, path_y, smooth_x, smooth_y, name="ch19_path_compare.png", title=""):
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.2), layout="constrained")

    ax = axes[0]
    ax.plot(path_x, path_y, color=WAVE, lw=1.6, label="原始累积路径")
    ax.plot(smooth_x, smooth_y, color=TEAL, lw=2.4, label="平滑后路径")
    ax.scatter([path_x[0]], [path_y[0]], color=WAVE, s=40, zorder=3)
    ax.scatter([path_x[-1]], [path_y[-1]], color=TEAL, s=40, zorder=3)
    ax.set_aspect("equal", adjustable="datalim")
    finish(ax)
    ax.set_xlabel("累积平移 x（像素）")
    ax.set_ylabel("累积平移 y（像素）")
    ax.set_title(f"{title}俯视轨迹")
    ax.legend(frameon=False, fontsize=10)

    ax = axes[1]
    t = np.arange(len(path_x))
    ax.plot(t, smooth_x - path_x, color=WAVE, lw=1.6, label="补偿 Δx = Sx−Px")
    ax.plot(t, smooth_y - path_y, color=TEAL, lw=1.6, label="补偿 Δy = Sy−Py")
    ax.axhline(0, color=MUTED, lw=0.8)
    finish(ax)
    ax.set_xlabel("帧号")
    ax.set_ylabel("平滑 − 原路径（像素）")
    ax.set_title("每帧 warp 用的补偿量")
    ax.legend(frameon=False, fontsize=10)
    save_fig(fig, name)


def plot_components(dx, dy, da, path_x, path_y, path_a, smooth_x, smooth_y, smooth_a):
    fig, axes = plt.subplots(3, 1, figsize=(9.2, 7.2), layout="constrained", sharex=True)
    t_inc = np.arange(1, len(dx) + 1)
    t = np.arange(len(path_x))

    axes[0].plot(t_inc, dx, color=WAVE, lw=1.2, alpha=0.7, label="帧间 dx")
    axes[0].plot(t_inc, dy, color=TEAL, lw=1.2, alpha=0.7, label="帧间 dy")
    finish(axes[0])
    axes[0].set_ylabel("增量（像素）")
    axes[0].set_title("步骤 1：帧间相似变换里的平移增量")
    axes[0].legend(frameon=False, fontsize=9, ncol=2)

    axes[1].plot(t, path_x, color=WAVE, lw=1.5, label="累积 x")
    axes[1].plot(t, smooth_x, color=TEAL, lw=2.2, label="平滑 x")
    axes[1].plot(t, path_y, color=WAVE, lw=1.0, ls="--", alpha=0.7, label="累积 y")
    axes[1].plot(t, smooth_y, color=TEAL, lw=1.6, ls="--", alpha=0.9, label="平滑 y")
    finish(axes[1])
    axes[1].set_ylabel("像素")
    axes[1].set_title("步骤 2–3：累积路径 vs 平滑路径")
    axes[1].legend(frameon=False, fontsize=9, ncol=2)

    axes[2].plot(t, np.rad2deg(path_a), color=WAVE, lw=1.5, label="累积转角")
    axes[2].plot(t, np.rad2deg(smooth_a), color=TEAL, lw=2.2, label="平滑转角")
    finish(axes[2])
    axes[2].set_xlabel("帧号")
    axes[2].set_ylabel("度")
    axes[2].set_title("转角同样要平滑，否则画面会来回拧")
    axes[2].legend(frameon=False, fontsize=9)
    save_fig(fig, "ch19_components.png")


def plot_oversmooth(path_x, path_y, path_a):
    """同一原路径，不同平滑半径：过平滑会吞掉运镜、补偿变大。"""
    radii = [3, 15, 35]
    fig, axes = plt.subplots(1, 3, figsize=(10.6, 3.6), layout="constrained")
    t = np.arange(len(path_x))
    for ax, r in zip(axes, radii):
        sx = moving_average(path_x, r)
        sy = moving_average(path_y, r)
        ax.plot(t, path_x, color=WAVE, lw=1.3, label="原 x")
        ax.plot(t, sx, color=TEAL, lw=2.2, label="平滑 x")
        finish(ax)
        ax.set_title(f"半径={r}")
        ax.set_xlabel("帧号")
        if r == 3:
            ax.set_ylabel("累积 x（像素）")
            ax.legend(frameon=False, fontsize=9)
        # 注：半径越大，平滑曲线越「直」，故意爬坡被抹平 → 漂
    fig.suptitle("过平滑：高频抖没了，连慢速运镜也被当成抖动抹掉", fontsize=13)
    save_fig(fig, "ch19_oversmooth.png")

    # 补偿幅度随半径变大 → 视野损失
    fig, ax = plt.subplots(figsize=(8.4, 3.6), layout="constrained")
    radii_dense = list(range(1, 36, 2))
    peaks = []
    for r in radii_dense:
        sx = moving_average(path_x, r)
        sy = moving_average(path_y, r)
        sa = moving_average(path_a, r)
        mag = np.hypot(sx - path_x, sy - path_y)
        peaks.append(mag.max())
    ax.plot(radii_dense, peaks, color=WAVE, lw=2.0)
    ax.scatter([3, 15, 35], [peaks[radii_dense.index(3)], peaks[radii_dense.index(15)], peaks[radii_dense.index(35)]], color=TEAL, zorder=3)
    finish(ax)
    ax.set_xlabel("平滑半径")
    ax.set_ylabel("最大补偿位移（像素）")
    ax.set_title("半径越大，补偿越大 → 黑边越多 → 裁切越狠")
    save_fig(fig, "ch19_crop_cost.png")


def plot_comp_idea():
    fig, ax = plt.subplots(figsize=(8.8, 3.4), layout="constrained")
    t = np.linspace(0, 1, 80)
    raw = 0.2 * t * 12 + 1.8 * np.sin(2 * np.pi * 7 * t)
    smooth = moving_average(raw, 8)
    ax.plot(t, raw, color=WAVE, lw=1.8, label="原路径 P")
    ax.plot(t, smooth, color=TEAL, lw=2.4, label="平滑路径 S")
    i = 35
    ax.annotate(
        "",
        xy=(t[i], smooth[i]),
        xytext=(t[i], raw[i]),
        arrowprops=dict(arrowstyle="<->", color="#1c1915", lw=1.4),
    )
    ax.text(t[i] + 0.02, (raw[i] + smooth[i]) / 2, "补偿 ≈ S−P", fontsize=11, va="center")
    finish(ax)
    ax.set_xlabel("时间")
    ax.set_ylabel("路径标量示意")
    ax.set_title("稳像不是抹掉路径，而是把每帧「拉」向平滑目标")
    ax.legend(frameon=False, fontsize=10)
    save_fig(fig, "ch19_comp_idea.png")


def write_videos(raw_frames, stable_frames):
    IMG.mkdir(parents=True, exist_ok=True)
    for name, frames in (
        ("ch19_shake_demo.mp4", raw_frames),
        ("ch19_stable_demo.mp4", stable_frames),
    ):
        path = IMG / name
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(path), fourcc, 12.0, (FRAME_W, FRAME_H))
        if not writer.isOpened():
            print(f"提示：无法写 {name}，跳过视频（图片已足够）。")
            continue
        for fr in frames:
            writer.write(fr)
        writer.release()
        print(f"已保存 {path}")


def run_pipeline(shake_amp, pan_speed, smooth_radius, seed=7):
    scene = make_scene()
    true_dx, true_dy, true_da = synthesize_transforms(
        N_FRAMES, shake_amp=shake_amp, pan_speed=pan_speed, seed=seed
    )
    true_x, true_y, true_a = accumulate_path(true_dx, true_dy, true_da)
    frames = render_frames(scene, true_x, true_y, true_a)

    est_dx, est_dy, est_da, _ = estimate_pairwise(frames, use_affine=False)
    path_x, path_y, path_a = accumulate_path(est_dx, est_dy, est_da)
    smooth_x = moving_average(path_x, smooth_radius)
    smooth_y = moving_average(path_y, smooth_radius)
    smooth_a = moving_average(path_a, smooth_radius)
    stable = stabilize_frames(frames, path_x, path_y, path_a, smooth_x, smooth_y, smooth_a)
    margin = estimate_needed_margin(
        path_x, path_y, path_a, smooth_x, smooth_y, smooth_a, FRAME_W, FRAME_H
    )
    cropped = crop_stable(stable, min(margin, 40))
    return {
        "frames": frames,
        "stable": stable,
        "cropped": cropped,
        "est_dx": est_dx,
        "est_dy": est_dy,
        "est_da": est_da,
        "path_x": path_x,
        "path_y": path_y,
        "path_a": path_a,
        "smooth_x": smooth_x,
        "smooth_y": smooth_y,
        "smooth_a": smooth_a,
        "margin": margin,
        "true_x": true_x,
        "true_y": true_y,
    }


def plot_before_after(raw_frames, stable_frames, cropped_frames):
    idx = [5, 20, 35, 50]
    raw_strip = make_strip(raw_frames, idx)
    stab_strip = make_strip(stable_frames, idx)
    crop_strip = make_strip(cropped_frames, idx)

    # 竖直拼三行
    gap = 10
    rows = [raw_strip, stab_strip, crop_strip]
    labels = ["晃动原片", "补偿 warp（可见黑边）", "裁切后成片"]
    widths = [r.shape[1] for r in rows]
    max_w = max(widths)
    total_h = sum(r.shape[0] for r in rows) + gap * 2 + 28 * 3
    canvas = np.full((total_h, max_w, 3), 251, dtype=np.uint8)
    y = 0
    for lab, row in zip(labels, rows):
        cv2.putText(canvas, lab, (4, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.65, INK_BGR, 2, cv2.LINE_AA)
        y += 28
        canvas[y : y + row.shape[0], : row.shape[1]] = row
        y += row.shape[0] + gap
    save_bgr("ch19_before_after.png", canvas)

    # 单帧对比放大看楼边
    mid = 30
    pair = np.full((FRAME_H + 36, FRAME_W * 2 + 12, 3), 251, dtype=np.uint8)
    pair[30 : 30 + FRAME_H, 0:FRAME_W] = raw_frames[mid]
    pair[30 : 30 + FRAME_H, FRAME_W + 12 :] = cropped_frames[mid]
    cv2.putText(pair, "before", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, WAVE_BGR, 2, cv2.LINE_AA)
    cv2.putText(
        pair,
        "after + crop",
        (FRAME_W + 20, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        TEAL_BGR,
        2,
        cv2.LINE_AA,
    )
    save_bgr("ch19_frame_pair.png", pair)


def plot_mine(result):
    plot_path_compare(
        result["path_x"],
        result["path_y"],
        result["smooth_x"],
        result["smooth_y"],
        name="ch19_my_path.png",
        title=f"半径={MY_SMOOTH_RADIUS} 幅度={MY_SHAKE_AMP}　",
    )
    idx = [8, 24, 40, 55]
    strip = make_strip(result["cropped"], idx)
    # 在上方标注参数
    banner = np.full((36, strip.shape[1], 3), 251, dtype=np.uint8)
    cv2.putText(
        banner,
        f"radius={MY_SMOOTH_RADIUS}  amp={MY_SHAKE_AMP}  pan={MY_PAN_SPEED}",
        (8, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        INK_BGR,
        2,
        cv2.LINE_AA,
    )
    save_bgr("ch19_my_strip.png", np.vstack([banner, strip]))

    # 视野损失示意：画裁切框
    fr = result["stable"][30].copy()
    m = min(result["margin"], 40)
    cv2.rectangle(fr, (m, m), (FRAME_W - 1 - m, FRAME_H - 1 - m), TEAL_BGR, 2)
    cv2.putText(fr, f"crop margin~{m}px", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, TEAL_BGR, 2, cv2.LINE_AA)
    save_bgr("ch19_my_crop.png", fr)


def self_check():
    # 1. 滑动平均：常数序列不变
    c = np.ones(20)
    assert np.allclose(moving_average(c, 5), c)

    # 2. 累积路径长度
    dx = np.array([1.0, 2.0, 3.0])
    x, y, a = accumulate_path(dx, dx * 0, dx * 0)
    assert len(x) == 4 and abs(x[-1] - 6.0) < 1e-9

    # 3. 相似矩阵抽出的角
    ang = np.deg2rad(8.0)
    m = sim_matrix(3.0, -4.0, ang)
    dx, dy, da = extract_dx_dy_da(m)
    assert abs(dx - 3) < 1e-9 and abs(dy + 4) < 1e-9 and abs(da - ang) < 1e-9

    # 4. 补偿：原路径=平滑时，warp 矩阵应接近单位（平移 0）
    frames = [np.zeros((32, 32, 3), dtype=np.uint8) for _ in range(5)]
    z = np.zeros(5)
    out = stabilize_frames(frames, z, z, z, z, z, z)
    assert len(out) == 5 and out[0].shape == (32, 32, 3)

    # 5. 估变换：静止双帧应接近 0
    img = np.full((80, 100, 3), 200, dtype=np.uint8)
    for i in range(5, 90, 12):
        cv2.circle(img, (i, 40), 4, (30, 30, 30), -1)
    edx, edy, eda, _ = estimate_pairwise([img, img.copy()])
    assert abs(edx[0]) < 0.5 and abs(edy[0]) < 0.5 and abs(eda[0]) < 0.05

    print("自检通过：平滑、累积、相似矩阵、补偿 warp、静止帧估计均正常。")


def main():
    use_style()
    self_check()

    plot_pipeline()
    plot_comp_idea()

    # 课文固定参数：半径 7 压高频、仍跟得上慢速运镜
    lesson = run_pipeline(shake_amp=10.0, pan_speed=1.6, smooth_radius=7, seed=7)
    plot_path_compare(
        lesson["path_x"],
        lesson["path_y"],
        lesson["smooth_x"],
        lesson["smooth_y"],
        name="ch19_path_compare.png",
    )
    plot_components(
        lesson["est_dx"],
        lesson["est_dy"],
        lesson["est_da"],
        lesson["path_x"],
        lesson["path_y"],
        lesson["path_a"],
        lesson["smooth_x"],
        lesson["smooth_y"],
        lesson["smooth_a"],
    )
    plot_oversmooth(lesson["path_x"], lesson["path_y"], lesson["path_a"])
    plot_before_after(lesson["frames"], lesson["stable"], lesson["cropped"])
    write_videos(lesson["frames"], lesson["cropped"])

    # 跟踪点示意
    demo = lesson["frames"][20].copy()
    gray = cv2.cvtColor(demo, cv2.COLOR_BGR2GRAY)
    pts = cv2.goodFeaturesToTrack(gray, 80, 0.01, 8)
    if pts is not None:
        for p in pts.reshape(-1, 2):
            cv2.circle(demo, (int(p[0]), int(p[1])), 3, TEAL_BGR, -1, cv2.LINE_AA)
    save_bgr("ch19_features.png", demo)

    # 「改这里」
    mine = run_pipeline(
        shake_amp=MY_SHAKE_AMP,
        pan_speed=MY_PAN_SPEED,
        smooth_radius=MY_SMOOTH_RADIUS,
        seed=7,
    )
    plot_mine(mine)

    print("课文对照图已写入 assets/figures/ 与 assets/images/")
    print("你的图：ch19_my_path.png、ch19_my_strip.png、ch19_my_crop.png")


if __name__ == "__main__":
    main()
