"""第 18 篇：光流可视化、评估与常见坑。

在「视频图片拼接防抖」文件夹里运行：

    MPLCONFIGDIR=/tmp/mpl-cache .venv/bin/python code/ch18_flow_eval.py

图会保存到 assets/figures/。顶部「改这里」只影响 ch18_my_*.png。
课文对照图每次都按固定参数重画，方便和网页正文对得上。
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
# 只影响 ch18_my_arrow_wheel.png 与 ch18_my_epe.png
MY_FLOW_U = 8.0  # 水平光流（像素，右为正）
MY_FLOW_V = -4.0  # 竖直光流（像素，下为正，与图像坐标一致）
MY_NOISE_STD = 1.5  # 估计光流相对真值的噪声标准差（像素）
# ==================

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "assets" / "figures"

WAVE_BGR = (40, 77, 194)


def save_fig(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    path = FIG / name
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"已保存 {path}")


# ---------------------------------------------------------------------------
# Middlebury 风格色轮：色相=方向，亮度/饱和度=幅值
# ---------------------------------------------------------------------------


def make_colorwheel(ncols=55):
    """经典 Middlebury color wheel（RGB，uint8），形状 (ncols, 3)。"""
    ry, yg, gc, cb, bm, mr = 15, 6, 4, 11, 13, 6
    ncols_expected = ry + yg + gc + cb + bm + mr
    assert ncols_expected == ncols
    colorwheel = np.zeros((ncols, 3), dtype=np.float32)
    col = 0
    # RY
    colorwheel[0:ry, 0] = 255
    colorwheel[0:ry, 1] = np.floor(255 * np.arange(ry) / ry)
    col += ry
    # YG
    colorwheel[col : col + yg, 0] = 255 - np.floor(255 * np.arange(yg) / yg)
    colorwheel[col : col + yg, 1] = 255
    col += yg
    # GC
    colorwheel[col : col + gc, 1] = 255
    colorwheel[col : col + gc, 2] = np.floor(255 * np.arange(gc) / gc)
    col += gc
    # CB
    colorwheel[col : col + cb, 1] = 255 - np.floor(255 * np.arange(cb) / cb)
    colorwheel[col : col + cb, 2] = 255
    col += cb
    # BM
    colorwheel[col : col + bm, 2] = 255
    colorwheel[col : col + bm, 0] = np.floor(255 * np.arange(bm) / bm)
    col += bm
    # MR
    colorwheel[col : col + mr, 2] = 255 - np.floor(255 * np.arange(mr) / mr)
    colorwheel[col : col + mr, 0] = 255
    return colorwheel.astype(np.uint8)


COLORWHEEL = make_colorwheel()


def flow_to_color(u, v, max_flow=None):
    """把 (u,v) 光流场染成 Middlebury RGB 图。u/v 可为标量或同形状数组。"""
    u = np.asarray(u, dtype=np.float32)
    v = np.asarray(v, dtype=np.float32)
    if u.ndim == 0:
        u = u.reshape(1, 1)
        v = v.reshape(1, 1)
        squeeze = True
    else:
        squeeze = False

    mag = np.sqrt(u * u + v * v)
    if max_flow is None:
        max_flow = float(np.percentile(mag, 99)) if mag.size else 1.0
        max_flow = max(max_flow, 1e-3)
    mag_n = np.clip(mag / max_flow, 0, 1)

    # Middlebury 惯例：arctan2(-v, -u)，向右 (+u) 落在色轮红端
    ang = np.arctan2(-v, -u) / np.pi  # [-1, 1]
    ang = (ang + 1.0) / 2.0  # [0, 1]

    ncols = COLORWHEEL.shape[0]
    fk = ang * (ncols - 1)
    k0 = np.floor(fk).astype(np.int32)
    k1 = (k0 + 1) % ncols
    f = fk - k0

    col0 = COLORWHEEL[k0] / 255.0
    col1 = COLORWHEEL[k1] / 255.0
    col = (1 - f)[..., None] * col0 + f[..., None] * col1

    # 幅值越大颜色越「实」；零运动偏白
    col = 1.0 - mag_n[..., None] * (1.0 - col)
    rgb = (col * 255).astype(np.uint8)
    if squeeze:
        return rgb[0, 0]
    return rgb


def colorwheel_disk(size=280, max_r=1.0):
    """画一张色轮图例：中心零运动，半径表示幅值方向。"""
    yy, xx = np.mgrid[-1 : 1 : complex(size), -1 : 1 : complex(size)]
    rr = np.sqrt(xx * xx + yy * yy)
    # 图像坐标：+x 右，+y 下；色轮上「上」应对应负 v
    u = xx * max_r
    v = yy * max_r
    rgb = flow_to_color(u, v, max_flow=max_r)
    mask = rr > 1.0
    rgb[mask] = 251
    return rgb


def draw_flow_arrows(base_rgb, u, v, step=16, scale=1.0, color_rgb=(194, 77, 40)):
    h, w = base_rgb.shape[:2]
    out_bgr = base_rgb[:, :, ::-1].copy()
    bgr = (int(color_rgb[2]), int(color_rgb[1]), int(color_rgb[0]))
    uu = np.asarray(u, dtype=np.float32)
    vv = np.asarray(v, dtype=np.float32)
    for y in range(step // 2, h, step):
        for x in range(step // 2, w, step):
            dx = float(uu[y, x] if uu.ndim == 2 else uu)
            dy = float(vv[y, x] if vv.ndim == 2 else vv)
            x2 = int(round(x + dx * scale))
            y2 = int(round(y + dy * scale))
            cv2.arrowedLine(out_bgr, (x, y), (x2, y2), bgr, 1, tipLength=0.35, line_type=cv2.LINE_AA)
    return out_bgr[:, :, ::-1]


def make_uniform_flow(h, w, u, v):
    fu = np.full((h, w), float(u), dtype=np.float32)
    fv = np.full((h, w), float(v), dtype=np.float32)
    return fu, fv


def make_swirl_flow(h, w, strength=6.0):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    dx, dy = xx - cx, yy - cy
    r = np.sqrt(dx * dx + dy * dy) + 1e-6
    # 切向：(-dy, dx) / r * strength * (r/max)
    scale = strength * (r / r.max())
    u = -dy / r * scale
    v = dx / r * scale
    return u.astype(np.float32), v.astype(np.float32)


# ---------------------------------------------------------------------------
# 合成场景与失败模式
# ---------------------------------------------------------------------------


def make_textured_scene(w=240, h=160, seed=0):
    """带几何图案的合成场景，方便看出运动。"""
    rng = np.random.default_rng(seed)
    img = np.full((h, w, 3), 236, dtype=np.uint8)
    img[: h // 2] = (228, 220, 208)
    img[h // 2 :] = (200, 188, 170)
    for x, bw, bh, color in (
        (30, 28, 70, (90, 110, 150)),
        (90, 22, 55, (70, 95, 130)),
        (150, 34, 80, (100, 90, 70)),
        (190, 26, 60, (80, 120, 110)),
    ):
        top = h // 2 - bh
        cv2.rectangle(img, (x, top), (x + bw, h // 2), color, -1)
    y = h // 2 + 28
    for x0 in range(10, w - 10, 36):
        cv2.line(img, (x0, y), (x0 + 20, y), (250, 250, 250), 2)
    cv2.circle(img, (120, 55), 18, (50, 120, 70), -1)
    noise = rng.integers(0, 12, (h, w, 1), dtype=np.uint8)
    img = cv2.add(img, np.repeat(noise, 3, axis=2))
    return img


def warp_by_flow(img, u, v):
    """按光流把 img「搬」到下一帧：dst(x+u,y+v) ≈ src(x,y) → 用反向采样。"""
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    map_x = (xx - u).astype(np.float32)
    map_y = (yy - v).astype(np.float32)
    return cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def farneback(gray1, gray2, levels=3, winsize=15):
    return cv2.calcOpticalFlowFarneback(
        gray1,
        gray2,
        None,
        pyr_scale=0.5,
        levels=levels,
        winsize=winsize,
        iterations=3,
        poly_n=5,
        poly_sigma=1.1,
        flags=0,
    )


def endpoint_error(u_est, v_est, u_gt, v_gt):
    return np.sqrt((u_est - u_gt) ** 2 + (v_est - v_gt) ** 2)


def mean_epe(u_est, v_est, u_gt, v_gt, mask=None):
    e = endpoint_error(u_est, v_est, u_gt, v_gt)
    if mask is not None:
        e = e[mask]
    return float(np.mean(e)) if e.size else 0.0


# ---------------------------------------------------------------------------
# 课文图
# ---------------------------------------------------------------------------


def plot_colorwheel_legend():
    disk = colorwheel_disk(300)
    fig, ax = plt.subplots(figsize=(5.2, 5.0), layout="constrained")
    ax.imshow(disk)
    ax.set_title("Middlebury 风格色轮（色相=方向，浅白=几乎不动）")
    ax.axis("off")
    # 方向标注
    cx = cy = 150
    ax.annotate("右 +u", xy=(260, cy), xytext=(210, cy - 8), color=INK, fontsize=11)
    ax.annotate("左 −u", xy=(40, cy), xytext=(55, cy - 8), color=INK, fontsize=11)
    ax.annotate("上 −v", xy=(cx, 40), xytext=(cx - 18, 55), color=INK, fontsize=11)
    ax.annotate("下 +v", xy=(cx, 260), xytext=(cx - 18, 240), color=INK, fontsize=11)
    ax.text(0.5, -0.02, "圆心附近偏白：幅值≈0；靠外缘颜色更实：幅值更大", ha="center", transform=ax.transAxes, color=MUTED, fontsize=10)
    save_fig(fig, "ch18_colorwheel.png")


def plot_arrow_vs_wheel():
    h, w = 160, 240
    u, v = make_swirl_flow(h, w, strength=7.0)
    # 加一块平移区，让箭头更「好看懂」
    u[:, :80] = 6.0
    v[:, :80] = 0.0

    base = np.full((h, w, 3), 245, dtype=np.uint8)
    arrow_img = draw_flow_arrows(base, u, v, step=14, scale=1.6, color_rgb=(194, 77, 40))
    color_img = flow_to_color(u, v, max_flow=10.0)

    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.6), layout="constrained")
    axes[0].imshow(arrow_img)
    axes[0].set_title("箭头图（稀疏采样）")
    axes[0].axis("off")
    axes[1].imshow(color_img)
    axes[1].set_title("色轮编码（每个像素）")
    axes[1].axis("off")
    axes[2].imshow(colorwheel_disk(180))
    axes[2].set_title("读色图例")
    axes[2].axis("off")
    fig.suptitle("同一运动场：箭头看方向/长短，色轮看整幅稠密结构", fontsize=13, color=INK)
    save_fig(fig, "ch18_arrow_vs_wheel.png")


def plot_epe_idea():
    """真值箭头 vs 估计箭头，并标 EPE。"""
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.0), layout="constrained")

    ax = axes[0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.set_aspect("equal")
    ax.set_title("端点误差 EPE：两个向量终点的距离")
    ax.axis("off")
    # 起点
    ax.plot(2.5, 3.5, "o", color=INK, ms=8)
    ax.annotate("", xy=(7.2, 5.2), xytext=(2.5, 3.5), arrowprops=dict(arrowstyle="->", color=TEAL, lw=2.4))
    ax.annotate("", xy=(6.4, 4.1), xytext=(2.5, 3.5), arrowprops=dict(arrowstyle="->", color=WAVE, lw=2.4))
    ax.text(7.3, 5.35, "真值终点", color=TEAL, fontsize=11)
    ax.text(6.5, 3.85, "估计终点", color=WAVE, fontsize=11)
    ax.plot([6.4, 7.2], [4.1, 5.2], "--", color=MUTED, lw=1.5)
    ax.text(7.0, 4.55, "EPE", color=MUTED, fontsize=12)
    ax.text(2.2, 2.7, "像素起点 (x,y)", color=MUTED, fontsize=10)
    ax.text(
        0.3,
        0.6,
        r"EPE $= \sqrt{(u_{\mathrm{est}}-u_{\mathrm{gt}})^2 + (v_{\mathrm{est}}-v_{\mathrm{gt}})^2}$",
        fontsize=12,
    )

    # 右：小图上的 EPE 热力图
    h, w = 80, 120
    u_gt, v_gt = make_uniform_flow(h, w, 6.0, -2.0)
    rng = np.random.default_rng(3)
    u_est = u_gt + rng.normal(0, 1.2, (h, w)).astype(np.float32)
    v_est = v_gt + rng.normal(0, 1.2, (h, w)).astype(np.float32)
    # 故意在右侧加一块坏区
    u_est[:, 70:] += 5.0
    epe = endpoint_error(u_est, v_est, u_gt, v_gt)
    mean = float(np.mean(epe))

    ax2 = axes[1]
    im = ax2.imshow(epe, cmap="YlOrRd", vmin=0, vmax=10)
    ax2.set_title(f"EPE 热力图（均值 ≈ {mean:.2f} px）")
    ax2.axis("off")
    fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04, label="像素")
    ax2.text(0.02, -0.08, "越红越大：这片区域估错得更狠", transform=ax2.transAxes, color=MUTED, fontsize=10)
    save_fig(fig, "ch18_epe.png")


def plot_failure_modes():
    """五种失败模式：左两帧，右 EPE 或估计色轮。"""
    cases = []

    # 1) 遮挡：前景矩形右移，露出背景；边界附近难估
    bg = make_textured_scene(240, 160, seed=1)
    img1 = bg.copy()
    cv2.rectangle(img1, (70, 50), (130, 110), (40, 70, 160), -1)
    u_gt, v_gt = make_uniform_flow(160, 240, 12.0, 0.0)
    # 只有矩形区域在动；背景静止——简化：整幅平移再贴回静止背景+新位置矩形
    img2 = bg.copy()
    cv2.rectangle(img2, (82, 50), (142, 110), (40, 70, 160), -1)
    # GT：矩形内 (12,0)，外 0；遮挡带（旧位置露出）无真值
    u_gt = np.zeros((160, 240), np.float32)
    v_gt = np.zeros((160, 240), np.float32)
    u_gt[50:110, 70:130] = 12.0
    valid = np.ones((160, 240), dtype=bool)
    valid[50:110, 70:82] = False  # 被揭开的条带：旧位置像素「消失」
    flow = farneback(cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY), cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY))
    epe = endpoint_error(flow[..., 0], flow[..., 1], u_gt, v_gt)
    epe[~valid] = np.nan
    cases.append(("遮挡：物体移开后露出新背景", img1, img2, flow, epe, mean_epe(flow[..., 0], flow[..., 1], u_gt, v_gt, valid)))

    # 2) 光照突变
    img1 = make_textured_scene(240, 160, seed=2)
    u_gt, v_gt = make_uniform_flow(160, 240, 4.0, 1.0)
    img2 = warp_by_flow(img1, u_gt, v_gt)
    img2 = cv2.convertScaleAbs(img2, alpha=1.35, beta=25)  # 突然变亮
    flow = farneback(cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY), cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY))
    epe = endpoint_error(flow[..., 0], flow[..., 1], u_gt, v_gt)
    cases.append(("光照突变：亮度守恒被打破", img1, img2, flow, epe, float(np.mean(epe))))

    # 3) 运动模糊
    img1 = make_textured_scene(240, 160, seed=3)
    u_gt, v_gt = make_uniform_flow(160, 240, 5.0, 0.0)
    sharp = warp_by_flow(img1, u_gt, v_gt)
    k = np.zeros((1, 15), np.float32)
    k[0, :] = 1.0 / 15
    img2 = cv2.filter2D(sharp, -1, k)
    flow = farneback(cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY), cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY))
    epe = endpoint_error(flow[..., 0], flow[..., 1], u_gt, v_gt)
    cases.append(("运动模糊：细节被抹平", img1, img2, flow, epe, float(np.mean(epe))))

    # 4) 重复纹理
    h, w = 160, 240
    img1 = np.full((h, w, 3), 220, dtype=np.uint8)
    cell = 16
    for y in range(0, h, cell):
        for x in range(0, w, cell):
            if ((x // cell) + (y // cell)) % 2 == 0:
                img1[y : y + cell, x : x + cell] = (90, 100, 120)
    u_gt, v_gt = make_uniform_flow(h, w, 8.0, 0.0)  # 正好半格的倍数，极易歧义
    img2 = warp_by_flow(img1, u_gt, v_gt)
    # 故意用较小窗口，突出歧义
    flow = farneback(
        cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY),
        cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY),
        levels=2,
        winsize=9,
    )
    epe = endpoint_error(flow[..., 0], flow[..., 1], u_gt, v_gt)
    cases.append(("重复纹理：很多格子看起来一样", img1, img2, flow, epe, float(np.mean(epe))))

    # 5) 过大位移（少层金字塔）
    img1 = make_textured_scene(240, 160, seed=5)
    u_gt, v_gt = make_uniform_flow(160, 240, 28.0, -6.0)
    img2 = warp_by_flow(img1, u_gt, v_gt)
    flow = farneback(
        cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY),
        cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY),
        levels=1,  # 几乎不靠金字塔
        winsize=15,
    )
    epe = endpoint_error(flow[..., 0], flow[..., 1], u_gt, v_gt)
    cases.append(("过大位移：搜索范围不够", img1, img2, flow, epe, float(np.mean(epe))))

    # 总览 5 行
    fig, axes = plt.subplots(5, 2, figsize=(10.0, 14.0), layout="constrained")
    for i, (title, im1, im2, flow, epe, me) in enumerate(cases):
        gap = 4
        h, w = im1.shape[:2]
        canvas = np.full((h, 2 * w + gap, 3), 251, dtype=np.uint8)
        canvas[:, :w] = im1[:, :, ::-1]  # BGR→RGB
        canvas[:, w + gap :] = im2[:, :, ::-1]
        axes[i, 0].imshow(canvas)
        axes[i, 0].set_title(f"{title}　（平均 EPE≈{me:.2f} px）", fontsize=11)
        axes[i, 0].axis("off")
        epe_show = np.nan_to_num(epe, nan=0.0)
        im = axes[i, 1].imshow(epe_show, cmap="YlOrRd", vmin=0, vmax=max(6.0, float(np.nanpercentile(epe, 95))))
        axes[i, 1].set_title("EPE（越红越错）", fontsize=11)
        axes[i, 1].axis("off")
        # 单独保存每张失败对照
        fig_one, ax_one = plt.subplots(1, 2, figsize=(9.2, 3.2), layout="constrained")
        ax_one[0].imshow(canvas)
        ax_one[0].set_title(f"{title}\n平均 EPE≈{me:.2f} px", fontsize=11)
        ax_one[0].axis("off")
        imx = ax_one[1].imshow(epe_show, cmap="YlOrRd", vmin=0, vmax=max(6.0, float(np.nanpercentile(epe, 95))))
        ax_one[1].set_title("EPE 热力图")
        ax_one[1].axis("off")
        fig_one.colorbar(imx, ax=ax_one[1], fraction=0.046, pad=0.04)
        names = [
            "ch18_fail_occlusion.png",
            "ch18_fail_lighting.png",
            "ch18_fail_blur.png",
            "ch18_fail_texture.png",
            "ch18_fail_large.png",
        ]
        save_fig(fig_one, names[i])

    fig.suptitle("五种常见坑：左=前后帧，右=相对真值的 EPE", fontsize=14, color=INK)
    save_fig(fig, "ch18_fail_grid.png")
    return cases


def plot_sparse_vs_dense():
    img = make_textured_scene(280, 180, seed=7)
    gray1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    u_gt, v_gt = make_swirl_flow(180, 280, strength=5.0)
    u_gt += 3.0
    img2 = warp_by_flow(img, u_gt, v_gt)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # 稀疏：goodFeatures + LK
    pts = cv2.goodFeaturesToTrack(gray1, maxCorners=80, qualityLevel=0.08, minDistance=12)
    pts2, st, _ = cv2.calcOpticalFlowPyrLK(gray1, gray2, pts, None, winSize=(21, 21), maxLevel=3)
    sparse_bgr = img.copy()
    if pts is not None and pts2 is not None:
        for p0, p1, s in zip(pts[:, 0], pts2[:, 0], st[:, 0]):
            if int(s) == 0:
                continue
            x0, y0 = int(p0[0]), int(p0[1])
            x1, y1 = int(p1[0]), int(p1[1])
            cv2.arrowedLine(sparse_bgr, (x0, y0), (x1, y1), WAVE_BGR, 1, tipLength=0.35, line_type=cv2.LINE_AA)
    sparse_rgb = sparse_bgr[:, :, ::-1]

    flow = farneback(gray1, gray2, levels=3, winsize=15)
    dense_rgb = flow_to_color(flow[..., 0], flow[..., 1], max_flow=10)

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.8), layout="constrained")
    axes[0].imshow(sparse_rgb)
    axes[0].set_title("稀疏：只在角点上画箭头\n适合跟踪 / 估全局变换 / 防抖")
    axes[0].axis("off")
    axes[1].imshow(dense_rgb)
    axes[1].set_title("稠密：每个像素一个向量（色轮）\n适合运动分割 / 稠密对应")
    axes[1].axis("off")
    fig.suptitle("怎么选：看你要「几个可靠点」还是「整幅运动场」", fontsize=13)
    save_fig(fig, "ch18_sparse_vs_dense.png")


def plot_raft_glance():
    fig, ax = plt.subplots(figsize=(10.0, 4.2), layout="constrained")
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis("off")
    ax.set_title("深度学习光流（RAFT 一类）在解决什么——一瞥", fontsize=13)

    boxes = [
        (0.3, 2.8, 2.4, 1.4, "两帧图像\n特征提取", TEAL),
        (3.2, 2.8, 2.6, 1.4, "相关体积\n「谁像谁」", WAVE),
        (6.3, 2.8, 2.6, 1.4, "迭代更新\n逐步改光流", TEAL),
        (9.4, 2.8, 2.2, 1.4, "稠密光流\n(u,v) 场", WAVE),
    ]
    for x, y, w, h, text, color in boxes:
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.08,rounding_size=0.15",
                facecolor=color,
                edgecolor=INK,
                alpha=0.18,
                lw=1.2,
            )
        )
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=11, color=INK)

    for x0, x1 in ((2.7, 3.2), (5.8, 6.3), (8.9, 9.4)):
        ax.annotate("", xy=(x1, 3.5), xytext=(x0, 3.5), arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.6))

    notes = [
        "相对经典 Farneback / LK：大位移、弱纹理、跨数据集往往更稳（靠大数据学出来的先验）。",
        "代价：需要 GPU、模型权重、更多算力与显存；手机实时仍常选稀疏 LK / 轻量网络。",
        "仍怕：极端遮挡、训练分布外的物理（强 RS、极端曝光）、标注噪声。评估照样看 EPE / 可视化。",
    ]
    for i, t in enumerate(notes):
        ax.text(0.4, 2.2 - i * 0.65, "· " + t, fontsize=11, color=INK, wrap=True)

    save_fig(fig, "ch18_raft_glance.png")


def plot_selection_table():
    fig, ax = plt.subplots(figsize=(9.2, 3.4), layout="constrained")
    ax.axis("off")
    ax.set_title("稀疏 vs 稠密：选型速查")
    col_labels = ["问题", "更偏向稀疏", "更偏向稠密"]
    rows = [
        ["视频防抖估相似/仿射", "✓ 跟踪点够稳就行", "通常浪费"],
        ["运动物体分割", "点太少", "✓ 需要整幅场"],
        ["实时 / CPU", "✓ LK + 金字塔", "Farneback/网络更重"],
        ["弱纹理大位移", "点可能不够", "深度网络常更好"],
        ["有真值要刷榜", "看任务", "EPE 在稠密场上算"],
    ]
    table = ax.table(cellText=rows, colLabels=col_labels, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.15, 1.55)
    save_fig(fig, "ch18_sparse_dense_table.png")


def plot_mine():
    """「改这里」参数：均匀光流的箭头+色轮，以及带噪声的 EPE。"""
    h, w = 140, 220
    u, v = make_uniform_flow(h, w, MY_FLOW_U, MY_FLOW_V)
    base = np.full((h, w, 3), 245, dtype=np.uint8)
    # 画网格帮助看方向
    for x in range(0, w, 20):
        base[:, x] = (230, 222, 210)
    for y in range(0, h, 20):
        base[y, :] = (230, 222, 210)
    arrow = draw_flow_arrows(base, u, v, step=18, scale=2.0)
    color = flow_to_color(u, v, max_flow=max(12.0, abs(MY_FLOW_U) + abs(MY_FLOW_V) + 1))

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.5), layout="constrained")
    axes[0].imshow(arrow)
    axes[0].set_title(f"你的均匀光流箭头  u={MY_FLOW_U}, v={MY_FLOW_V}")
    axes[0].axis("off")
    axes[1].imshow(color)
    axes[1].set_title("同一场的色轮")
    axes[1].axis("off")
    save_fig(fig, "ch18_my_arrow_wheel.png")

    rng = np.random.default_rng(0)
    u_est = u + rng.normal(0, MY_NOISE_STD, (h, w)).astype(np.float32)
    v_est = v + rng.normal(0, MY_NOISE_STD, (h, w)).astype(np.float32)
    epe = endpoint_error(u_est, v_est, u, v)
    me = float(np.mean(epe))
    fig, ax = plt.subplots(figsize=(6.4, 3.4), layout="constrained")
    im = ax.imshow(epe, cmap="YlOrRd")
    ax.set_title(f"你的噪声估计：平均 EPE≈{me:.3f} px（噪声σ={MY_NOISE_STD}）")
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="EPE (px)")
    save_fig(fig, "ch18_my_epe.png")


def self_check():
    # 1) 零光流应接近白色
    rgb0 = flow_to_color(0.0, 0.0, max_flow=10.0)
    assert rgb0.shape == (3,) and int(rgb0.min()) >= 250

    # 2) 纯右向流：色相应偏红系（R 高）
    rgb_r = flow_to_color(5.0, 0.0, max_flow=10.0)
    assert rgb_r[0] > rgb_r[2]

    # 3) EPE 公式
    e = endpoint_error(np.array([[3.0]]), np.array([[4.0]]), np.array([[0.0]]), np.array([[0.0]]))
    assert abs(float(e[0, 0]) - 5.0) < 1e-5

    # 4) Farneback 对小平移应给出合理平均 EPE
    img1 = make_textured_scene(160, 120, seed=99)
    u_gt, v_gt = make_uniform_flow(120, 160, 3.0, -1.0)
    img2 = warp_by_flow(img1, u_gt, v_gt)
    flow = farneback(cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY), cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY))
    me = mean_epe(flow[..., 0], flow[..., 1], u_gt, v_gt)
    assert me < 1.5, f"小平移 EPE 过大: {me}"

    # 5) 色轮图例尺寸
    disk = colorwheel_disk(64)
    assert disk.shape == (64, 64, 3)

    print("自检通过：色轮、EPE、Farneback 小平移均正常。")


def main():
    use_style()
    self_check()
    plot_colorwheel_legend()
    plot_arrow_vs_wheel()
    plot_epe_idea()
    plot_failure_modes()
    plot_sparse_vs_dense()
    plot_raft_glance()
    plot_selection_table()
    plot_mine()
    print("课文对照图已写入 assets/figures/")
    print("你的图：ch18_my_arrow_wheel.png , ch18_my_epe.png")


if __name__ == "__main__":
    main()
