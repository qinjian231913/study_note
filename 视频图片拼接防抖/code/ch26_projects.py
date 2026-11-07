"""第 26 篇：端到端实战三则（全景 / 二维视觉防抖 / 陀螺辅助 RS）。

在「视频图片拼接防抖」文件夹里运行：

    MPLCONFIGDIR=/tmp/mpl-cache .venv/bin/python code/ch26_projects.py

图会保存到 assets/figures/ 与 assets/images/。
顶部「改这里」只影响 ch26_my_*.png；课文对照图按固定参数重画。
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

from plot_style import INK, MUTED, TEAL, WAVE, finish, use_style

WAVE_BGR = (40, 77, 194)
TEAL_BGR = (102, 107, 14)
INK_BGR = (21, 25, 28)

# ===== 改这里 =====
# 只影响 ch26_my_panorama.png / ch26_my_stabilize.png / ch26_my_gyro.png
PANORAMA_BLEND_WIDTH = 40  # 实战 A：羽化带宽（像素）。越大接缝越软，过大可能发糊
STABILIZE_SMOOTH_WIN = 15  # 实战 B：路径平滑窗口（奇数更好）。越大越稳，过大运镜变「漂」
GYRO_RS_STRENGTH = 1.0  # 实战 C：RS 畸变强度倍数。1.0=默认；改大斜切更夸张，校正量也按比例
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


# ---------------------------------------------------------------------------
# 共用：合成宽街景
# ---------------------------------------------------------------------------


def make_wide_scene(width=960, height=320):
    """横向更宽的合成街景，方便裁多张重叠「照片」和晃动裁切。"""
    img = np.full((height, width, 3), 245, dtype=np.uint8)
    img[: height // 2] = (236, 228, 214)
    img[height // 2 :] = (210, 198, 178)

    hills = np.array(
        [
            [0, 160],
            [160, 90],
            [320, 140],
            [480, 70],
            [640, 130],
            [800, 85],
            [960, 150],
            [960, 160],
            [0, 160],
        ],
        np.int32,
    )
    cv2.fillPoly(img, [hills], (180, 168, 150))

    buildings = (
        (40, 50, 130, (90, 110, 150)),
        (120, 42, 100, (70, 95, 130)),
        (220, 58, 150, (100, 90, 70)),
        (340, 46, 120, (80, 120, 110)),
        (450, 54, 140, (110, 85, 95)),
        (560, 48, 115, (75, 100, 140)),
        (670, 60, 155, (95, 80, 65)),
        (790, 44, 125, (85, 115, 100)),
        (880, 52, 135, (100, 95, 120)),
    )
    for x, w, h, color in buildings:
        top = height // 2 - h
        cv2.rectangle(img, (x, top), (x + w, height // 2), color, -1)
        cv2.rectangle(img, (x, top), (x + w, height // 2), (40, 40, 40), 1)
        # 窗户网格，便于 ORB / 跟踪
        for wy in range(top + 12, height // 2 - 8, 18):
            for wx in range(x + 8, x + w - 8, 14):
                cv2.rectangle(img, (wx, wy), (wx + 6, wy + 8), (200, 210, 220), -1)

    y = height // 2 + 36
    for x0 in range(16, width - 16, 48):
        cv2.line(img, (x0, y), (x0 + 26, y), (250, 250, 250), 3)

    # 几棵树
    for tx in (200, 500, 720):
        cv2.rectangle(img, (tx, 200), (tx + 12, 290), (60, 80, 40), -1)
        cv2.circle(img, (tx + 6, 175), 34, (50, 120, 70), -1)

    # 竖向标杆：对 RS 斜切特别敏感
    for px in (90, 300, 610, 850):
        cv2.line(img, (px, 40), (px, height - 20), (30, 30, 30), 2)

    for gx in range(30, width, 60):
        for gy in range(30, height, 55):
            cv2.circle(img, (gx, gy), 2, (160, 140, 120), -1)

    return img


def crop_view(scene, left, top, view_w, view_h):
    h, w = scene.shape[:2]
    left = int(np.clip(left, 0, max(0, w - view_w)))
    top = int(np.clip(top, 0, max(0, h - view_h)))
    return scene[top : top + view_h, left : left + view_w].copy(), left, top


# ---------------------------------------------------------------------------
# 实战 A：多图全景（平面单应 + 羽化；柱面示意）
# ---------------------------------------------------------------------------


def make_panorama_views(scene, view_w=320, view_h=240, tops=(40, 42, 38)):
    """从宽场景裁三张有重叠的「照片」，略带竖直抖动模拟手持。"""
    starts = (20, 280, 540)
    views = []
    metas = []
    for left0, top0 in zip(starts, tops):
        view, left, top = crop_view(scene, left0, top0, view_w, view_h)
        views.append(view)
        metas.append((left, top))
    return views, metas


def estimate_homography(img_a, img_b, ratio=0.75):
    """ORB + Lowe 比率 + RANSAC 估 H，把 B 变到 A 坐标系。"""
    orb = cv2.ORB_create(nfeatures=1200)
    kp1, des1 = orb.detectAndCompute(img_a, None)
    kp2, des2 = orb.detectAndCompute(img_b, None)
    if des1 is None or des2 is None or len(kp1) < 8 or len(kp2) < 8:
        return None, 0, 0
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    knn = bf.knnMatch(des2, des1, k=2)
    good = []
    for pair in knn:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < ratio * n.distance:
            good.append(m)
    if len(good) < 8:
        return None, len(good), 0
    src = np.float32([kp2[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kp1[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 4.0)
    inliers = int(mask.sum()) if mask is not None else 0
    return H, len(good), inliers


def warp_to_canvas(img, H, canvas_size, offset=(0, 0)):
    """把图经 H 投到大画布；offset 是额外平移（写进 H）。"""
    ox, oy = offset
    T = np.array([[1, 0, ox], [0, 1, oy], [0, 0, 1]], dtype=np.float64)
    M = T @ H
    return cv2.warpPerspective(img, M, canvas_size)


def feather_blend(base, overlay, blend_width=40):
    """按非零掩码做水平方向羽化融合（简化版多图叠图）。"""
    out = base.copy().astype(np.float32)
    ov = overlay.astype(np.float32)
    mask = (overlay.sum(axis=2) > 0).astype(np.uint8)
    if mask.sum() == 0:
        return base
    # 距离变换：离掩码边缘越远权重大
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
    alpha = np.clip(dist / max(blend_width, 1), 0, 1)
    # 已有内容处按 alpha 混；空白处直接贴
    base_mask = (base.sum(axis=2) > 0).astype(np.float32)
    a = alpha * mask.astype(np.float32)
    # 重叠区：两边都有内容时用羽化；只一边有时用有内容的
    both = (base_mask > 0) & (mask > 0)
    only_ov = (base_mask == 0) & (mask > 0)
    for c in range(3):
        ch = out[:, :, c]
        och = ov[:, :, c]
        ch[both] = (1 - a[both]) * ch[both] + a[both] * och[both]
        ch[only_ov] = och[only_ov]
        out[:, :, c] = ch
    return np.clip(out, 0, 255).astype(np.uint8)


def stitch_three(views, blend_width=40, canvas_pad=80):
    """三图链式拼接：以中图为锚，左图 / 右图分别估 H。"""
    left, mid, right = views
    h, w = mid.shape[:2]
    H_ml, n_l, in_l = estimate_homography(mid, left)
    H_mr, n_r, in_r = estimate_homography(mid, right)
    if H_ml is None or H_mr is None:
        raise SystemExit(
            f"全景匹配失败：左 good/inliers={n_l}/{in_l}，右={n_r}/{in_r}。"
            "请确认场景纹理足够，或降低 Lowe 比率。"
        )

    # 估计画布范围：把四角经 H 投出去
    corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    all_pts = [corners]
    for H in (H_ml, H_mr, np.eye(3)):
        pts = cv2.perspectiveTransform(corners, H.astype(np.float64))
        all_pts.append(pts)
    pts_cat = np.concatenate(all_pts, axis=0).reshape(-1, 2)
    min_xy = pts_cat.min(axis=0) - canvas_pad
    max_xy = pts_cat.max(axis=0) + canvas_pad
    ox, oy = -min_xy[0], -min_xy[1]
    cw = int(np.ceil(max_xy[0] - min_xy[0]))
    ch = int(np.ceil(max_xy[1] - min_xy[1]))
    canvas_size = (cw, ch)

    mid_w = warp_to_canvas(mid, np.eye(3), canvas_size, (ox, oy))
    left_w = warp_to_canvas(left, H_ml, canvas_size, (ox, oy))
    right_w = warp_to_canvas(right, H_mr, canvas_size, (ox, oy))

    hard = mid_w.copy()
    for piece in (left_w, right_w):
        m = piece.sum(axis=2) > 0
        hard[m] = piece[m]

    soft = mid_w.copy()
    soft = feather_blend(soft, left_w, blend_width)
    soft = feather_blend(soft, right_w, blend_width)

    info = {
        "left_matches": (n_l, in_l),
        "right_matches": (n_r, in_r),
        "canvas": canvas_size,
    }
    return hard, soft, info


def cylindrical_warp(img, f=280):
    """把平面图卷到柱面再展开（简单针孔模型）。f 越大弯曲越弱。"""
    h, w = img.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    out = np.zeros_like(img)
    xs = np.arange(w)
    ys = np.arange(h)
    map_x = np.zeros((h, w), np.float32)
    map_y = np.zeros((h, w), np.float32)
    for y in ys:
        for x in xs:
            theta = (x - cx) / f
            h_cyl = (y - cy) / f
            # 柱面点反投到平面
            map_x[y, x] = f * np.tan(theta) + cx
            map_y[y, x] = f * h_cyl / np.cos(theta) + cy
    return cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)


def plot_panorama_pipeline():
    fig, ax = plt.subplots(figsize=(10.2, 3.2), layout="constrained")
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 3.2)
    ax.axis("off")
    steps = [
        (0.3, "多图输入"),
        (2.5, "特征匹配"),
        (4.7, "估 H / 柱面"),
        (6.9, "投到画布"),
        (9.1, "羽化融合"),
    ]
    for x, label in steps:
        box = FancyBboxPatch(
            (x, 1.0),
            1.8,
            1.2,
            boxstyle="round,pad=0.08,rounding_size=0.15",
            facecolor="#fde8df" if x < 5 else "#e5f3f1",
            edgecolor=WAVE if x < 5 else TEAL,
            lw=1.6,
        )
        ax.add_patch(box)
        ax.text(x + 0.9, 1.6, label, ha="center", va="center", fontsize=11, color=INK)
    for x0 in (2.1, 4.3, 6.5, 8.7):
        ax.annotate(
            "",
            xy=(x0 + 0.35, 1.6),
            xytext=(x0 - 0.05, 1.6),
            arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.5),
        )
    ax.set_title("实战 A 流水线（对应第 5–11 篇积木）")
    save_fig(fig, "ch26_panorama_pipeline.png")


def make_panorama_figures(blend_width=40):
    scene = make_wide_scene()
    views, metas = make_panorama_views(scene)

    # 输入条带
    labeled = []
    for i, (v, (L, T)) in enumerate(zip(views, metas)):
        vis = v.copy()
        cv2.putText(
            vis,
            f"view {i + 1}  L={L}",
            (8, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            WAVE_BGR if i != 1 else TEAL_BGR,
            2,
            cv2.LINE_AA,
        )
        labeled.append(vis)
    gap = 8
    h, w = labeled[0].shape[:2]
    strip = np.full((h, 3 * w + 2 * gap, 3), 251, dtype=np.uint8)
    for i, fr in enumerate(labeled):
        x0 = i * (w + gap)
        strip[:, x0 : x0 + w] = fr
    save_bgr("ch26_panorama_inputs.png", strip)

    hard, soft, info = stitch_three(views, blend_width=blend_width)

    fig, axes = plt.subplots(2, 1, figsize=(10.5, 6.2), layout="constrained")
    axes[0].imshow(hard[:, :, ::-1])
    axes[0].set_title(
        f"硬叠（无羽化）　左内点 {info['left_matches'][1]} / 右内点 {info['right_matches'][1]}"
    )
    axes[0].axis("off")
    axes[1].imshow(soft[:, :, ::-1])
    axes[1].set_title(f"羽化融合 blend_width={blend_width}")
    axes[1].axis("off")
    save_fig(fig, "ch26_panorama_result.png")

    # 柱面示意：单张中图卷一下
    cyl = cylindrical_warp(views[1], f=260)
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6), layout="constrained")
    axes[0].imshow(views[1][:, :, ::-1])
    axes[0].set_title("平面取景（中图）")
    axes[0].axis("off")
    axes[1].imshow(cyl[:, :, ::-1])
    axes[1].set_title("柱面展开示意（f=260）")
    axes[1].axis("off")
    save_fig(fig, "ch26_panorama_cylindrical.png")

    return soft, info


# ---------------------------------------------------------------------------
# 实战 B：二维视觉防抖
# ---------------------------------------------------------------------------


def shaky_crop_path(n=60, amplitude=14.0, seed=7):
    """故意运镜（缓慢平移）+ 高频抖动。"""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, n)
    ideal_x = 80 + 220 * t
    ideal_y = 40 + 12 * np.sin(np.pi * t)
    jitter_x = amplitude * (
        0.55 * np.sin(2 * np.pi * 6.5 * t) + 0.25 * rng.normal(0, 1, n)
    )
    jitter_y = amplitude * (
        0.60 * np.sin(2 * np.pi * 5.2 * t + 0.5) + 0.25 * rng.normal(0, 1, n)
    )
    raw_x = ideal_x + jitter_x
    raw_y = ideal_y + jitter_y
    return t, raw_x, raw_y, ideal_x, ideal_y


def extract_frames(scene, raw_x, raw_y, view_w=320, view_h=240):
    frames = []
    for x, y in zip(raw_x, raw_y):
        fr, _, _ = crop_view(scene, x, y, view_w, view_h)
        frames.append(fr)
    return frames


def estimate_frame_translations(frames):
    """用稀疏 LK 估帧间平移（二维视觉 EIS 最小可行）。"""
    prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
    pts0 = cv2.goodFeaturesToTrack(
        prev_gray, maxCorners=200, qualityLevel=0.01, minDistance=8, blockSize=7
    )
    dxs = [0.0]
    dys = [0.0]
    lk = dict(
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )
    for i in range(1, len(frames)):
        gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
        if pts0 is None or len(pts0) < 8:
            pts0 = cv2.goodFeaturesToTrack(
                prev_gray, maxCorners=200, qualityLevel=0.01, minDistance=8, blockSize=7
            )
        if pts0 is None:
            dxs.append(0.0)
            dys.append(0.0)
            prev_gray = gray
            continue
        pts1, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, pts0, None, **lk)
        if pts1 is None or st is None:
            dxs.append(0.0)
            dys.append(0.0)
            prev_gray = gray
            continue
        good0 = pts0[st.ravel() == 1]
        good1 = pts1[st.ravel() == 1]
        if len(good0) < 6:
            dxs.append(0.0)
            dys.append(0.0)
        else:
            d = good1.reshape(-1, 2) - good0.reshape(-1, 2)
            # 中位数抗外点
            dxs.append(float(np.median(d[:, 0])))
            dys.append(float(np.median(d[:, 1])))
        prev_gray = gray
        pts0 = good1.reshape(-1, 1, 2) if len(good1) >= 8 else None
    return np.array(dxs), np.array(dys)


def accumulate_path(dxs, dys):
    """帧间位移累积成路径（相对第 0 帧）。"""
    path_x = np.cumsum(dxs)
    path_y = np.cumsum(dys)
    return path_x, path_y


def moving_average(arr, win):
    win = max(3, int(win))
    if win % 2 == 0:
        win += 1
    pad = win // 2
    kernel = np.ones(win, dtype=np.float64) / win
    padded = np.pad(arr, (pad, pad), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def stabilize_frames(frames, path_x, path_y, smooth_x, smooth_y, crop_margin=24):
    """用「原路径 − 平滑路径」的差反向平移，再裁掉黑边。"""
    h, w = frames[0].shape[:2]
    out_w, out_h = w - 2 * crop_margin, h - 2 * crop_margin
    stab = []
    for i, fr in enumerate(frames):
        # 补偿量：把当前帧拉回平滑路径
        cx = smooth_x[i] - path_x[i]
        cy = smooth_y[i] - path_y[i]
        M = np.float32([[1, 0, cx], [0, 1, cy]])
        warped = cv2.warpAffine(fr, M, (w, h), borderMode=cv2.BORDER_CONSTANT)
        cropped = warped[crop_margin : crop_margin + out_h, crop_margin : crop_margin + out_w]
        stab.append(cropped)
    return stab


def make_stabilize_figures(smooth_win=15, amplitude=14.0):
    scene = make_wide_scene()
    t, raw_x, raw_y, ideal_x, ideal_y = shaky_crop_path(n=60, amplitude=amplitude)
    frames = extract_frames(scene, raw_x, raw_y)
    dxs, dys = estimate_frame_translations(frames)
    path_x, path_y = accumulate_path(dxs, dys)
    # 跟踪估出的路径是「画面内容相对取景」的位移；为了可视化，叠加到裁切中心轨迹上
    # 更直观：直接平滑裁切中心的 raw 路径（真值演示）+ 跟踪路径对照
    track_cx = raw_x[0] + path_x
    track_cy = raw_y[0] + path_y
    smooth_tx = moving_average(track_cx, smooth_win)
    smooth_ty = moving_average(track_cy, smooth_win)

    # 真正用于补偿的：平滑「估出的路径」
    smooth_px = moving_average(path_x, smooth_win)
    smooth_py = moving_average(path_y, smooth_win)
    stab = stabilize_frames(frames, path_x, path_y, smooth_px, smooth_py, crop_margin=28)

    fig, axes = plt.subplots(2, 1, figsize=(9.8, 6.0), layout="constrained", sharex=True)
    axes[0].plot(t, raw_x, color=WAVE, lw=1.6, label="裁切中心 x（含抖动）")
    axes[0].plot(t, ideal_x, color=TEAL, lw=2.0, label="故意运镜 x")
    axes[0].plot(t, track_cx, color=MUTED, lw=1.2, ls="--", label="跟踪累积 x")
    axes[0].plot(t, smooth_tx, color=INK, lw=1.8, label=f"平滑后 x（win={smooth_win}）")
    finish(axes[0])
    axes[0].set_ylabel("水平位置")
    axes[0].legend(frameon=False, fontsize=9, ncol=2)
    axes[0].set_title("实战 B：路径估计与平滑")

    axes[1].plot(t, raw_y, color=WAVE, lw=1.6, label="裁切中心 y")
    axes[1].plot(t, ideal_y, color=TEAL, lw=2.0, label="故意运镜 y")
    axes[1].plot(t, track_cy, color=MUTED, lw=1.2, ls="--", label="跟踪累积 y")
    axes[1].plot(t, smooth_ty, color=INK, lw=1.8, label="平滑后 y")
    finish(axes[1])
    axes[1].set_xlabel("归一化时间")
    axes[1].set_ylabel("竖直位置")
    axes[1].legend(frameon=False, fontsize=9, ncol=2)
    save_fig(fig, "ch26_stabilize_path.png")

    # 前后对比条带：取 5 帧
    idxs = [0, 12, 24, 36, 48]
    before = []
    after = []
    for i in idxs:
        b = frames[i].copy()
        a = stab[i].copy()
        # 把 after 缩放到和 before 同高便于并排（裁切后略小）
        a2 = cv2.resize(a, (b.shape[1], b.shape[0]))
        cv2.putText(b, f"raw t={i}", (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, WAVE_BGR, 2)
        cv2.putText(a2, f"stab t={i}", (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, TEAL_BGR, 2)
        before.append(b)
        after.append(a2)

    def pack_row(parts, gap=6):
        h, w = parts[0].shape[:2]
        canvas = np.full((h, len(parts) * w + (len(parts) - 1) * gap, 3), 251, dtype=np.uint8)
        for i, p in enumerate(parts):
            x0 = i * (w + gap)
            canvas[:, x0 : x0 + w] = p
        return canvas

    row_b = pack_row(before)
    row_a = pack_row(after)
    compare = np.full(
        (row_b.shape[0] + row_a.shape[0] + 20, max(row_b.shape[1], row_a.shape[1]), 3),
        251,
        dtype=np.uint8,
    )
    compare[: row_b.shape[0], : row_b.shape[1]] = row_b
    compare[row_b.shape[0] + 20 :, : row_a.shape[1]] = row_a
    save_bgr("ch26_stabilize_compare.png", compare)

    # 抖动能量：补偿前后帧间位移的 RMS
    def frame_rms(seq):
        diffs = []
        for i in range(1, len(seq)):
            g0 = cv2.cvtColor(seq[i - 1], cv2.COLOR_BGR2GRAY).astype(np.float32)
            g1 = cv2.cvtColor(seq[i], cv2.COLOR_BGR2GRAY).astype(np.float32)
            # 用中心块平均绝对差近似「跳变」
            h, w = g0.shape
            c0 = g0[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4]
            c1 = g1[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4]
            diffs.append(float(np.mean(np.abs(c1 - c0))))
        return float(np.mean(diffs))

    rms_before = frame_rms(frames)
    # stab 尺寸不同，统一缩放到同一尺寸再比
    stab_full = [cv2.resize(s, (frames[0].shape[1], frames[0].shape[0])) for s in stab]
    rms_after = frame_rms(stab_full)

    fig, ax = plt.subplots(figsize=(6.4, 3.4), layout="constrained")
    ax.bar([0, 1], [rms_before, rms_after], color=[WAVE, TEAL], width=0.55)
    ax.set_xticks([0, 1], ["防抖前", "防抖后"])
    ax.set_ylabel("帧间中心块平均绝对差")
    ax.set_title(f"预期效果：跳变能量下降　{rms_before:.1f} → {rms_after:.1f}")
    finish(ax)
    ax.set_xlim(-0.6, 1.6)
    save_fig(fig, "ch26_stabilize_metric.png")

    return {
        "rms_before": rms_before,
        "rms_after": rms_after,
        "path_x": path_x,
        "smooth_px": smooth_px,
        "frames": frames,
        "stab": stab,
    }


# ---------------------------------------------------------------------------
# 实战 C：合成 IMU + Rolling Shutter 校正
# ---------------------------------------------------------------------------


def make_pole_scene(width=360, height=280):
    """竖杆 + 窗户网格：RS 旋转时竖线变斜最明显。"""
    img = np.full((height, width, 3), 240, dtype=np.uint8)
    img[:] = (230, 222, 210)
    cv2.rectangle(img, (0, 0), (width, height // 2), (210, 200, 188), -1)
    for x in (60, 120, 180, 240, 300):
        cv2.line(img, (x, 20), (x, height - 20), (35, 35, 35), 3)
    for y in range(40, height - 20, 28):
        cv2.line(img, (20, y), (width - 20, y), (120, 110, 100), 1)
    cv2.putText(img, "GS", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, TEAL_BGR, 2)
    return img


def apply_rs_yaw(img, omega_yaw, readout_time=0.030, fx=320.0):
    """全局快门图 → 卷帘快门：绕竖直轴角速度导致每行水平错位。

    白话：第 y 行比第 0 行晚曝光 y/H * readout_time，期间相机又转了一点，
    同一竖线在画面里被「拧」成斜线。omega_yaw 单位 rad/s。
    """
    h, w = img.shape[:2]
    map_x = np.zeros((h, w), np.float32)
    map_y = np.zeros((h, w), np.float32)
    cx = w / 2.0
    for y in range(h):
        t_row = (y / max(h - 1, 1)) * readout_time
        # 小角度：像素水平位移 ≈ fx * tan(omega * t) ≈ fx * omega * t
        shift = fx * np.tan(omega_yaw * t_row)
        map_x[y, :] = np.arange(w, dtype=np.float32) - shift
        map_y[y, :] = y
    return cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)


def correct_rs_yaw(img, omega_yaw, readout_time=0.030, fx=320.0):
    """已知角速度时，对 RS 图做反向行位移（陀螺辅助 RS 校正最小版）。"""
    return apply_rs_yaw(img, -omega_yaw, readout_time=readout_time, fx=fx)


def synth_imu_omega(n=120, fps_imu=200.0, seed=3):
    """合成一段角速度日志：低频姿态变化 + 高频手抖。"""
    rng = np.random.default_rng(seed)
    t = np.arange(n) / fps_imu
    # yaw 为主；pitch/roll 较小
    yaw = 0.8 * np.sin(2 * np.pi * 1.5 * t) + 0.25 * np.sin(2 * np.pi * 7 * t)
    yaw += 0.05 * rng.normal(0, 1, n)
    pitch = 0.15 * np.sin(2 * np.pi * 2.2 * t + 0.3)
    roll = 0.12 * np.sin(2 * np.pi * 3.1 * t + 1.0)
    return t, yaw, pitch, roll


def gyro_eis_compensation(omega_seq, dt, smooth_win=9):
    """对角速度积分得姿态，再平滑，差分为补偿旋转（仅演示 yaw→水平位移量）。"""
    theta = np.cumsum(omega_seq) * dt
    smooth = moving_average(theta, smooth_win)
    # 补偿角 = 平滑 − 原始（把高频抖掉）
    return theta, smooth, smooth - theta


def make_gyro_figures(rs_strength=1.0):
    gs = make_pole_scene()
    # 选一个瞬时 yaw 角速度（取 IMU 峰值附近）
    t_imu, yaw, pitch, roll = synth_imu_omega()
    omega0 = float(yaw[np.argmax(np.abs(yaw))]) * rs_strength
    readout = 0.030
    fx = 320.0

    rs = apply_rs_yaw(gs, omega0, readout_time=readout, fx=fx)
    fixed = correct_rs_yaw(rs, omega0, readout_time=readout, fx=fx)

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.5), layout="constrained")
    titles = [
        "全局快门（真值）",
        f"RS 斜切 ω={omega0:.2f} rad/s",
        "陀螺反向校正后",
    ]
    for ax, im, title in zip(axes, (gs, rs, fixed), titles):
        ax.imshow(im[:, :, ::-1])
        ax.set_title(title, fontsize=11)
        ax.axis("off")
    save_fig(fig, "ch26_gyro_rs_demo.png")

    fig, axes = plt.subplots(2, 1, figsize=(9.6, 5.4), layout="constrained", sharex=True)
    axes[0].plot(t_imu, yaw, color=WAVE, lw=1.5, label="yaw")
    axes[0].plot(t_imu, pitch, color=TEAL, lw=1.2, label="pitch")
    axes[0].plot(t_imu, roll, color=MUTED, lw=1.2, label="roll")
    finish(axes[0])
    axes[0].set_ylabel("角速度 (rad/s)")
    axes[0].legend(frameon=False, fontsize=9, ncol=3)
    axes[0].set_title("合成 IMU 日志（演示用，非真实手机数据）")

    dt = t_imu[1] - t_imu[0]
    theta, smooth, comp = gyro_eis_compensation(yaw * rs_strength, dt, smooth_win=11)
    axes[1].plot(t_imu, theta, color=WAVE, lw=1.5, label="积分姿态 yaw")
    axes[1].plot(t_imu, smooth, color=TEAL, lw=2.0, label="平滑目标姿态")
    axes[1].plot(t_imu, comp, color=MUTED, lw=1.2, ls="--", label="补偿角")
    finish(axes[1])
    axes[1].set_xlabel("时间 (s)")
    axes[1].set_ylabel("角度 (rad)")
    axes[1].legend(frameon=False, fontsize=9, ncol=3)
    save_fig(fig, "ch26_gyro_curves.png")

    # 竖线斜率度量：看校正是否把杆掰直
    def pole_tilt(gray_img, x0=180):
        """在 x0 附近找每行最暗像素，拟合斜率。"""
        h, w = gray_img.shape
        xs = []
        ys = []
        for y in range(30, h - 30):
            row = gray_img[y, max(0, x0 - 40) : min(w, x0 + 40)]
            if row.size == 0:
                continue
            xs.append(max(0, x0 - 40) + int(np.argmin(row)))
            ys.append(y)
        if len(xs) < 10:
            return 0.0
        coef = np.polyfit(ys, xs, 1)
        return float(coef[0])  # dx/dy，0 表示竖直

    g_gs = cv2.cvtColor(gs, cv2.COLOR_BGR2GRAY)
    g_rs = cv2.cvtColor(rs, cv2.COLOR_BGR2GRAY)
    g_fix = cv2.cvtColor(fixed, cv2.COLOR_BGR2GRAY)
    tilt = {
        "gs": pole_tilt(g_gs),
        "rs": pole_tilt(g_rs),
        "fixed": pole_tilt(g_fix),
        "omega": omega0,
    }
    return tilt


# ---------------------------------------------------------------------------
# 全套概念收束图
# ---------------------------------------------------------------------------


def plot_concept_map():
    fig, ax = plt.subplots(figsize=(11.2, 7.2), layout="constrained")
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.set_title("全套概念收束：四条主线如何接到三则实战", fontsize=14, pad=8)

    tracks = [
        (0.4, 8.2, 6.2, 1.4, "几何与配准  第 0–7 篇", "坐标 · 仿射/透视 · 标定 · 特征 · RANSAC", WAVE),
        (0.4, 6.3, 6.2, 1.4, "图片拼接  第 8–13 篇", "流水线 · 柱面/球面 · 融合 · 曝光 · BA", WAVE),
        (0.4, 4.4, 6.2, 1.4, "光流重点  第 14–18 篇", "亮度守恒 · LK · 金字塔 · 稠密 · 评估", TEAL),
        (0.4, 2.5, 6.2, 1.4, "防抖 / RS / 陀螺  第 19–25 篇", "路径平滑 · 裁切 · RS · IMU · 陀螺 EIS · 3D/视频拼", TEAL),
    ]
    for x, y, w, h, title, sub, color in tracks:
        box = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.06,rounding_size=0.12",
            facecolor="#fffdf8",
            edgecolor=color,
            lw=1.8,
        )
        ax.add_patch(box)
        ax.text(x + 0.25, y + h - 0.45, title, fontsize=12, color=color, fontweight="bold")
        ax.text(x + 0.25, y + 0.35, sub, fontsize=10, color=MUTED)

    projects = [
        (8.0, 7.4, 5.4, 1.8, "实战 A  多图全景", "特征→H→画布→羽化\n（可加柱面 / BA）"),
        (8.0, 5.0, 5.4, 1.8, "实战 B  二维视觉防抖", "跟踪→累积路径→平滑→补偿裁切"),
        (8.0, 2.6, 5.4, 1.8, "实战 C  陀螺 / RS", "合成 IMU→姿态→行级校正 / EIS"),
    ]
    for x, y, w, h, title, sub in projects:
        box = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.06,rounding_size=0.12",
            facecolor="#e5f3f1",
            edgecolor=TEAL,
            lw=1.8,
        )
        ax.add_patch(box)
        ax.text(x + 0.25, y + h - 0.5, title, fontsize=12, color=TEAL, fontweight="bold")
        ax.text(x + 0.25, y + 0.35, sub, fontsize=10, color=INK)

    # 箭头：主线 → 实战
    arrows = [
        ((6.6, 8.9), (8.0, 8.3)),
        ((6.6, 7.0), (8.0, 8.0)),
        ((6.6, 5.1), (8.0, 5.9)),
        ((6.6, 3.2), (8.0, 5.6)),
        ((6.6, 3.2), (8.0, 3.5)),
    ]
    for (x0, y0), (x1, y1) in arrows:
        ax.annotate(
            "",
            xy=(x1, y1),
            xytext=(x0, y0),
            arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.3, connectionstyle="arc3,rad=0.05"),
        )

    ax.text(
        7.0,
        0.7,
        "记住分工：拼接拼「空间视野」；防抖稳「时间观感」；陀螺补「视觉看不清/RS 行时间差」。",
        ha="center",
        fontsize=11,
        color=INK,
    )
    save_fig(fig, "ch26_concept_map.png")


def plot_error_checklist():
    """报错自检清单可视化（方便网页引用）。"""
    rows = [
        ("全景：匹配点太少", "纹理不足 / 重叠太少 / 模糊", "加大重叠；换 ORB→SIFT；查曝光"),
        ("全景：重影、接缝硬", "H 外点多 / 未羽化 / 视差大", "看内点数；加 blend；近景少拼"),
        ("防抖：越稳越「漂」", "平滑窗口过大", "减小 STABILIZE_SMOOTH_WIN"),
        ("防抖：仍抖、黑边多", "跟踪丢点 / 裁切不够", "提高特征数；加大 crop_margin"),
        ("陀螺：校正反了", "轴符号 / 延迟反了", "对 omega 取负；检查时间戳对齐"),
        ("陀螺：越校越斜", "读出时间/焦距不准", "调 readout_time 与 fx"),
        ("通用：中文方框", "缺字体", "看终端「作图字体」提示"),
        ("通用：No module cv2", "未装依赖 / 未激活 venv", "pip install -r code/requirements.txt"),
    ]
    fig, ax = plt.subplots(figsize=(11.0, 5.8), layout="constrained")
    ax.axis("off")
    ax.set_title("报错与自检清单（先对号入座再改参数）", fontsize=13)
    col_x = [0.02, 0.28, 0.55]
    headers = ["现象", "常见原因", "先试什么"]
    y0 = 0.92
    for i, h in enumerate(headers):
        ax.text(col_x[i], y0, h, transform=ax.transAxes, fontsize=11, color=WAVE, fontweight="bold")
    for r, (a, b, c) in enumerate(rows):
        y = y0 - 0.09 * (r + 1)
        bg = "#fde8df" if r % 2 == 0 else "#e5f3f1"
        ax.add_patch(
            FancyBboxPatch(
                (0.01, y - 0.03),
                0.98,
                0.075,
                transform=ax.transAxes,
                boxstyle="round,pad=0.01,rounding_size=0.02",
                facecolor=bg,
                edgecolor="none",
            )
        )
        ax.text(col_x[0], y, a, transform=ax.transAxes, fontsize=10, color=INK, va="center")
        ax.text(col_x[1], y, b, transform=ax.transAxes, fontsize=10, color=MUTED, va="center")
        ax.text(col_x[2], y, c, transform=ax.transAxes, fontsize=10, color=TEAL, va="center")
    save_fig(fig, "ch26_error_checklist.png")


# ---------------------------------------------------------------------------
# 「改这里」输出
# ---------------------------------------------------------------------------


def plot_mine():
    """只写 ch26_my_*.png，不覆盖课文对照图。"""
    scene = make_wide_scene()
    views, _ = make_panorama_views(scene)
    _, soft, info = stitch_three(views, blend_width=PANORAMA_BLEND_WIDTH)
    fig, ax = plt.subplots(figsize=(10.0, 3.4), layout="constrained")
    ax.imshow(soft[:, :, ::-1])
    ax.set_title(
        f"你的全景：blend_width={PANORAMA_BLEND_WIDTH}　"
        f"内点 L/R={info['left_matches'][1]}/{info['right_matches'][1]}"
    )
    ax.axis("off")
    save_fig(fig, "ch26_my_panorama.png")

    t, raw_x, raw_y, _, _ = shaky_crop_path(n=60, amplitude=14.0)
    frames = extract_frames(scene, raw_x, raw_y)
    dxs, dys = estimate_frame_translations(frames)
    path_x, path_y = accumulate_path(dxs, dys)
    smooth_px = moving_average(path_x, STABILIZE_SMOOTH_WIN)
    smooth_py = moving_average(path_y, STABILIZE_SMOOTH_WIN)
    stab = stabilize_frames(frames, path_x, path_y, smooth_px, smooth_py, crop_margin=28)

    def frame_rms(seq):
        diffs = []
        for i in range(1, len(seq)):
            g0 = cv2.cvtColor(seq[i - 1], cv2.COLOR_BGR2GRAY).astype(np.float32)
            g1 = cv2.cvtColor(seq[i], cv2.COLOR_BGR2GRAY).astype(np.float32)
            h, w = g0.shape
            c0 = g0[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4]
            c1 = g1[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4]
            diffs.append(float(np.mean(np.abs(c1 - c0))))
        return float(np.mean(diffs))

    stab_full = [cv2.resize(s, (frames[0].shape[1], frames[0].shape[0])) for s in stab]
    rms_b, rms_a = frame_rms(frames), frame_rms(stab_full)

    fig, ax = plt.subplots(figsize=(9.2, 3.4), layout="constrained")
    ax.plot(path_x, color=WAVE, lw=1.5, label="估出的路径 x")
    ax.plot(smooth_px, color=TEAL, lw=2.0, label=f"平滑 win={STABILIZE_SMOOTH_WIN}")
    finish(ax)
    ax.set_title(f"你的防抖平滑　帧间跳变 {rms_b:.1f}→{rms_a:.1f}")
    ax.legend(frameon=False, fontsize=10)
    save_fig(fig, "ch26_my_stabilize.png")

    gs = make_pole_scene()
    _, yaw, _, _ = synth_imu_omega()
    omega0 = float(yaw[np.argmax(np.abs(yaw))]) * GYRO_RS_STRENGTH
    rs = apply_rs_yaw(gs, omega0)
    fixed = correct_rs_yaw(rs, omega0)

    def pole_tilt(gray_img, x0=180):
        h, w = gray_img.shape
        xs, ys = [], []
        for y in range(30, h - 30):
            row = gray_img[y, max(0, x0 - 40) : min(w, x0 + 40)]
            if row.size == 0:
                continue
            xs.append(max(0, x0 - 40) + int(np.argmin(row)))
            ys.append(y)
        if len(xs) < 10:
            return 0.0
        return float(np.polyfit(ys, xs, 1)[0])

    tilt_rs = pole_tilt(cv2.cvtColor(rs, cv2.COLOR_BGR2GRAY))
    tilt_fx = pole_tilt(cv2.cvtColor(fixed, cv2.COLOR_BGR2GRAY))
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.4), layout="constrained")
    axes[0].imshow(rs[:, :, ::-1])
    axes[0].set_title(f"你的 RS（strength={GYRO_RS_STRENGTH}）")
    axes[0].axis("off")
    axes[1].imshow(fixed[:, :, ::-1])
    axes[1].set_title(f"校正后　斜率 {tilt_rs:.3f}→{tilt_fx:.3f}")
    axes[1].axis("off")
    save_fig(fig, "ch26_my_gyro.png")


# ---------------------------------------------------------------------------
# 自检
# ---------------------------------------------------------------------------


def self_check():
    scene = make_wide_scene()
    assert scene.shape == (320, 960, 3), scene.shape

    views, _ = make_panorama_views(scene)
    assert len(views) == 3
    hard, soft, info = stitch_three(views, blend_width=40)
    assert soft.shape[0] > 200 and soft.shape[1] > 500, soft.shape
    assert info["left_matches"][1] >= 8 and info["right_matches"][1] >= 8, info

    t, raw_x, raw_y, _, _ = shaky_crop_path(n=30, amplitude=10.0)
    frames = extract_frames(scene, raw_x, raw_y, view_w=240, view_h=180)
    dxs, dys = estimate_frame_translations(frames)
    path_x, path_y = accumulate_path(dxs, dys)
    sm = moving_average(path_x, 9)
    assert len(sm) == len(path_x)
    stab = stabilize_frames(frames, path_x, path_y, sm, moving_average(path_y, 9), crop_margin=16)
    assert len(stab) == len(frames)

    gs = make_pole_scene()
    rs = apply_rs_yaw(gs, 0.9)
    fixed = correct_rs_yaw(rs, 0.9)
    assert rs.shape == gs.shape == fixed.shape
    # 校正后应更接近原图（中心竖带）
    def center_mae(a, b):
        ga = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY).astype(np.float32)
        gb = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY).astype(np.float32)
        h, w = ga.shape
        return float(np.mean(np.abs(ga[:, w // 3 : 2 * w // 3] - gb[:, w // 3 : 2 * w // 3])))

    assert center_mae(gs, fixed) < center_mae(gs, rs) * 0.85 + 5.0

    t_imu, yaw, pitch, roll = synth_imu_omega(n=50)
    assert len(t_imu) == 50 and yaw.std() > 0

    print("自检通过：全景可拼、路径可平滑、合成 IMU+RS 可校正。")


def main():
    use_style()
    self_check()
    plot_concept_map()
    plot_error_checklist()
    plot_panorama_pipeline()
    make_panorama_figures(blend_width=40)
    make_stabilize_figures(smooth_win=15, amplitude=14.0)
    tilt = make_gyro_figures(rs_strength=1.0)
    print(
        f"RS 斜率示意：GS={tilt['gs']:.4f}，畸变={tilt['rs']:.4f}，校正后={tilt['fixed']:.4f}"
    )
    plot_mine()
    print("课文对照图已写入 assets/figures/ 与 assets/images/")
    print("你的图：ch26_my_panorama.png、ch26_my_stabilize.png、ch26_my_gyro.png")


if __name__ == "__main__":
    main()
