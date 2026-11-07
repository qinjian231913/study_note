"""第 4 篇：相机标定与去畸变。

    python code/ch04_calib.py

顶部「改这里」只影响 ch04_my_undistort.png（用你设的畸变系数合成再掰直）。
"""

from pathlib import Path

SHOW_WINDOW = False

import matplotlib

if not SHOW_WINDOW:
    matplotlib.use("Agg")

import cv2
import matplotlib.pyplot as plt
import numpy as np

from plot_style import MUTED, TEAL, WAVE, use_style

# ===== 改这里 =====
# 合成畸变时用的径向系数（负≈桶形，正≈枕形）；切向系数通常很小
MY_K1 = -0.35
MY_K2 = 0.08
MY_P1 = 0.0
MY_P2 = 0.0
# ==================

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "assets" / "figures"
IMG = ROOT / "assets" / "images"
WAVE_BGR = (40, 77, 194)
TEAL_BGR = (102, 107, 14)
INK_BGR = (28, 25, 21)

# 课文对照用的「真值」内参与畸变（合成数据）
BOARD_COLS = 9  # 内角点列数
BOARD_ROWS = 6  # 内角点行数
SQUARE = 25.0  # 虚拟方格边长（任意单位）
IMG_W, IMG_H = 640, 480
TRUE_FX, TRUE_FY = 520.0, 515.0
TRUE_CX, TRUE_CY = IMG_W / 2.0, IMG_H / 2.0
TRUE_K1, TRUE_K2 = -0.28, 0.06
TRUE_P1, TRUE_P2 = 0.001, -0.0008


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


def true_K():
    return np.array(
        [[TRUE_FX, 0, TRUE_CX], [0, TRUE_FY, TRUE_CY], [0, 0, 1]],
        dtype=np.float64,
    )


def true_dist():
    return np.array([TRUE_K1, TRUE_K2, TRUE_P1, TRUE_P2, 0.0], dtype=np.float64)


def object_points():
    """棋盘格三维点：Z=0 平面上的网格。"""
    obj = np.zeros((BOARD_ROWS * BOARD_COLS, 3), dtype=np.float32)
    obj[:, :2] = np.mgrid[0:BOARD_COLS, 0:BOARD_ROWS].T.reshape(-1, 2) * SQUARE
    return obj


def make_grid_scene(w=IMG_W, h=IMG_H):
    """带网格与建筑感直线的场景，用来看桶形/枕形。"""
    img = np.full((h, w, 3), 251, dtype=np.uint8)
    step = 40
    for x in range(0, w, step):
        cv2.line(img, (x, 0), (x, h - 1), (220, 210, 195), 1)
    for y in range(0, h, step):
        cv2.line(img, (0, y), (w - 1, y), (220, 210, 195), 1)
    # 粗直线：窗框感
    for x in (80, w // 2, w - 80):
        cv2.line(img, (x, 30), (x, h - 30), TEAL_BGR, 2)
    for y in (60, h // 2, h - 60):
        cv2.line(img, (40, y), (w - 40, y), WAVE_BGR, 2)
    cv2.putText(img, "straight lines", (24, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, INK_BGR, 2, cv2.LINE_AA)
    return img


def apply_distortion(ideal_bgr, K, dist):
    """把理想图「弄弯」：畸变画布上每个像素，用 undistortPoints 找回理想坐标再取样。"""
    h, w = ideal_bgr.shape[:2]
    ys, xs = np.indices((h, w), dtype=np.float32)
    pts = np.stack([xs.ravel(), ys.ravel()], axis=1).astype(np.float32).reshape(-1, 1, 2)
    und = cv2.undistortPoints(pts, K, dist, P=K)
    map_x = und[:, 0, 0].reshape(h, w).astype(np.float32)
    map_y = und[:, 0, 1].reshape(h, w).astype(np.float32)
    return cv2.remap(
        ideal_bgr,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(251, 251, 251),
    )


def apply_distortion_fast(ideal_bgr, K, dist):
    return apply_distortion(ideal_bgr, K, dist)


def draw_projected_board(shape, img_pts):
    """根据投影角点画一张「拍到的」棋盘示意（用于多姿态展示）。"""
    h, w = shape
    canvas = np.full((h, w, 3), 248, dtype=np.uint8)
    pts = img_pts.reshape(-1, 2)
    for i in range(BOARD_ROWS):
        for j in range(BOARD_COLS):
            idx = i * BOARD_COLS + j
            x, y = pts[idx]
            if 0 <= x < w and 0 <= y < h:
                color = WAVE_BGR if (i + j) % 2 == 0 else TEAL_BGR
                cv2.circle(canvas, (int(x), int(y)), 4, color, -1, lineType=cv2.LINE_AA)
    # 连线成网
    for i in range(BOARD_ROWS):
        row = pts[i * BOARD_COLS : (i + 1) * BOARD_COLS].astype(np.int32)
        cv2.polylines(canvas, [row], False, (90, 85, 80), 1, lineType=cv2.LINE_AA)
    for j in range(BOARD_COLS):
        col = pts[j :: BOARD_COLS].astype(np.int32)
        cv2.polylines(canvas, [col], False, (90, 85, 80), 1, lineType=cv2.LINE_AA)
    return canvas


def synthesize_views(n_views=8, dist=None, seed=7):
    """多姿态合成带畸变的棋盘观测，供 calibrateCamera。"""
    rng = np.random.default_rng(seed)
    K = true_K()
    if dist is None:
        dist = true_dist()
    obj = object_points()
    obj_list, img_list, vis_list = [], [], []

    # 基础姿态：板子在相机前方
    base_t = np.array([[0.0], [0.0], [450.0]])

    for i in range(n_views):
        # 随机小旋转（度）和平移
        angles = rng.uniform(-28, 28, size=3)
        rvec = np.deg2rad(angles).astype(np.float64).reshape(3, 1)
        tvec = base_t + rng.uniform(-60, 60, size=(3, 1))
        tvec[2, 0] = abs(tvec[2, 0]) + 280  # 保证在前方

        img_pts, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
        # 确保点大致落在画面内
        xy = img_pts.reshape(-1, 2)
        if np.any(xy[:, 0] < 5) or np.any(xy[:, 0] > IMG_W - 5):
            continue
        if np.any(xy[:, 1] < 5) or np.any(xy[:, 1] > IMG_H - 5):
            continue

        obj_list.append(obj.copy())
        img_list.append(img_pts.astype(np.float32))
        vis_list.append(draw_projected_board((IMG_H, IMG_W), img_pts))

    if len(obj_list) < 5:
        raise SystemExit("合成视角太少，请调姿态范围")
    return obj_list, img_list, vis_list, K, dist


def plot_why_undistort():
    """桶形把直线弄弯 → 特征位置被带偏。"""
    K = true_K()
    dist_barrel = np.array([-0.35, 0.05, 0, 0, 0], dtype=np.float64)
    ideal = make_grid_scene()
    bent = apply_distortion_fast(ideal, K, dist_barrel)

    # 在弯图上标几个「假想特征点」
    vis = bent.copy()
    # 理想直线上的点被弯到别处
    for x in (120, 320, 520):
        # 取中线附近
        y = IMG_H // 2
        cv2.circle(vis, (x, y), 6, WAVE_BGR, -1)
    cv2.putText(vis, "features on bent lines", (20, IMG_H - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, WAVE_BGR, 2, cv2.LINE_AA)

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0), layout="constrained")
    axes[0].imshow(ideal[:, :, ::-1])
    axes[0].set_title("理想：窗框是直线")
    axes[0].axis("off")
    axes[1].imshow(vis[:, :, ::-1])
    axes[1].set_title("桶形畸变后：直线变弯，特征被带偏")
    axes[1].axis("off")
    save_fig(fig, "ch04_why_undistort.png")
    save_bgr("ch04_grid_ideal.png", ideal)
    save_bgr("ch04_grid_barrel.png", bent)


def plot_pinhole_intrinsics():
    fig, ax = plt.subplots(figsize=(9.2, 4.4), layout="constrained")
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.set_title("针孔模型直觉与内参四个数")

    ax.text(0.3, 4.2, "世界一点 → 穿过「针孔」→ 落在成像平面一个像素", fontsize=13)
    ax.text(0.3, 3.35, "内参矩阵 K（3×3）：", fontsize=13)
    ax.text(
        0.3,
        2.55,
        "[[ fx,  0, cx ],\n [  0, fy, cy ],\n [  0,  0,  1 ]]",
        fontsize=13,
        family="monospace",
        color="#1c1915",
        va="top",
    )
    ax.text(5.2, 3.1, "fx, fy：焦距（像素）\n越大 → 物体在画面上越大", fontsize=12, color=WAVE, va="top")
    ax.text(5.2, 1.9, "cx, cy：主点\n光轴扎到传感器的位置\n理想时大约在画面中心", fontsize=12, color=TEAL, va="top")
    ax.text(0.3, 0.45, "先不管畸变：像素 ≈ K · (X/Z, Y/Z, 1)；再加畸变，直线才会弯", fontsize=12, color=MUTED)
    save_fig(fig, "ch04_pinhole_K.png")


def plot_barrel_vs_pincushion():
    K = true_K()
    ideal = make_grid_scene()
    barrel = apply_distortion_fast(ideal, K, np.array([-0.32, 0.04, 0, 0, 0], dtype=np.float64))
    pin = apply_distortion_fast(ideal, K, np.array([0.28, 0.02, 0, 0, 0], dtype=np.float64))
    tang = apply_distortion_fast(ideal, K, np.array([0.0, 0.0, 0.04, 0.03, 0], dtype=np.float64))

    fig, axes = plt.subplots(2, 2, figsize=(9.6, 7.2), layout="constrained")
    panels = [
        (axes[0, 0], ideal, "原图：直线网格"),
        (axes[0, 1], barrel, "径向 k1<0：桶形（中间鼓）"),
        (axes[1, 0], pin, "径向 k1>0：枕形（中间瘪）"),
        (axes[1, 1], tang, "切向 p1,p2：整体像被斜推"),
    ]
    for ax, im, title in panels:
        ax.imshow(im[:, :, ::-1])
        ax.set_title(title)
        ax.axis("off")
    save_fig(fig, "ch04_distortion_types.png")
    save_bgr("ch04_barrel.png", barrel)
    save_bgr("ch04_pincushion.png", pin)


def plot_before_after_undistort():
    K = true_K()
    dist = true_dist()
    ideal = make_grid_scene()
    distorted = apply_distortion_fast(ideal, K, dist)
    fixed = cv2.undistort(distorted, K, dist, None, K)

    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.8), layout="constrained")
    axes[0].imshow(ideal[:, :, ::-1])
    axes[0].set_title("理想直线")
    axes[0].axis("off")
    axes[1].imshow(distorted[:, :, ::-1])
    axes[1].set_title("加畸变（合成）")
    axes[1].axis("off")
    axes[2].imshow(fixed[:, :, ::-1])
    axes[2].set_title("undistort 之后")
    axes[2].axis("off")
    save_fig(fig, "ch04_undistort_compare.png")
    save_bgr("ch04_distorted.png", distorted)
    save_bgr("ch04_undistorted.png", fixed)


def plot_calib_pipeline():
    from matplotlib.patches import FancyBboxPatch

    fig, ax = plt.subplots(figsize=(9.2, 3.8), layout="constrained")
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.set_title("棋盘格标定流程（白话）")
    steps = [
        (0.4, "① 多姿态\n拍照", WAVE),
        (2.6, "② 找内角点\nfindChessboard", TEAL),
        (5.0, "③ 估 K 与\n畸变系数", WAVE),
        (7.4, "④ undistort\n掰直画面", TEAL),
    ]
    for x, text, color in steps:
        ax.add_patch(
            FancyBboxPatch(
                (x, 1.0),
                1.9,
                2.0,
                boxstyle="round,pad=0.05,rounding_size=0.15",
                facecolor="#fffdf8",
                edgecolor=color,
                lw=2,
            )
        )
        ax.text(x + 0.95, 2.0, text, ha="center", va="center", fontsize=12, color="#1c1915")
    for x in (2.3, 4.7, 7.1):
        ax.annotate(
            "",
            xy=(x + 0.25, 2.0),
            xytext=(x - 0.25, 2.0),
            arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.5),
        )
    ax.text(0.4, 0.35, "姿态要多样：正面、斜侧、靠近边缘……只拍一张定不稳", fontsize=11, color=MUTED)
    save_fig(fig, "ch04_calib_pipeline.png")


def run_calibration_demo():
    obj_list, img_list, vis_list, K_true, dist_true = synthesize_views(n_views=12)

    # 拼几张姿态示意
    n_show = min(4, len(vis_list))
    fig, axes = plt.subplots(1, n_show, figsize=(11.0, 3.2), layout="constrained")
    if n_show == 1:
        axes = [axes]
    for i in range(n_show):
        axes[i].imshow(vis_list[i][:, :, ::-1])
        axes[i].set_title(f"姿态 {i + 1}")
        axes[i].axis("off")
    save_fig(fig, "ch04_chess_poses.png")
    save_bgr("ch04_chess_pose0.png", vis_list[0])

    flags = 0
    rms, K_est, dist_est, rvecs, tvecs = cv2.calibrateCamera(
        obj_list,
        img_list,
        (IMG_W, IMG_H),
        None,
        None,
        flags=flags,
    )

    # 数值对照图
    fig, ax = plt.subplots(figsize=(8.8, 4.0), layout="constrained")
    ax.axis("off")
    ax.set_title(f"calibrateCamera 结果（重投影 RMS ≈ {rms:.4f} px）")
    lines = [
        f"真值 fx,fy = {TRUE_FX:.1f}, {TRUE_FY:.1f}　　估计 fx,fy = {K_est[0,0]:.1f}, {K_est[1,1]:.1f}",
        f"真值 cx,cy = {TRUE_CX:.1f}, {TRUE_CY:.1f}　　估计 cx,cy = {K_est[0,2]:.1f}, {K_est[1,2]:.1f}",
        f"真值 k1,k2 = {TRUE_K1:.3f}, {TRUE_K2:.3f}",
        f"估计 k1,k2 = {dist_est[0,0]:.3f}, {dist_est[0,1]:.3f}",
        f"真值 p1,p2 = {TRUE_P1:.4f}, {TRUE_P2:.4f}",
        f"估计 p1,p2 = {dist_est[0,2]:.4f}, {dist_est[0,3]:.4f}",
    ]
    ax.text(0.04, 0.72, "\n".join(lines), fontsize=12, transform=ax.transAxes, va="top")
    ax.text(
        0.04,
        0.12,
        "合成角点无噪声时会非常接近；真机照片有噪声，要更多张、姿态更散。",
        fontsize=11,
        color=MUTED,
        transform=ax.transAxes,
    )
    save_fig(fig, "ch04_calib_result.png")

    # 用估计参数去畸变
    scene = make_grid_scene()
    distorted = apply_distortion_fast(scene, K_true, dist_true)
    fixed = cv2.undistort(distorted, K_est, dist_est, None, K_est)

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.0), layout="constrained")
    axes[0].imshow(distorted[:, :, ::-1])
    axes[0].set_title("标定前：弯的")
    axes[0].axis("off")
    axes[1].imshow(fixed[:, :, ::-1])
    axes[1].set_title("用估出的 K/dist undistort")
    axes[1].axis("off")
    save_fig(fig, "ch04_calib_undistort.png")

    # 最小可运行示例代码截图式说明
    fig, ax = plt.subplots(figsize=(9.0, 3.6), layout="constrained")
    ax.axis("off")
    ax.set_title("最小调用（概念）")
    code = (
        "rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(\n"
        "    obj_points, img_points, image_size, None, None)\n"
        "straight = cv2.undistort(warped, K, dist)"
    )
    ax.text(0.04, 0.55, code, fontsize=13, family="monospace", transform=ax.transAxes, va="center")
    save_fig(fig, "ch04_api_snippet.png")

    return K_est, dist_est, rms


def plot_mine():
    """改这里：用读者设的畸变合成，再 undistort。"""
    K = true_K()
    dist = np.array([MY_K1, MY_K2, MY_P1, MY_P2, 0.0], dtype=np.float64)
    ideal = make_grid_scene()
    bent = apply_distortion_fast(ideal, K, dist)
    fixed = cv2.undistort(bent, K, dist, None, K)

    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.8), layout="constrained")
    axes[0].imshow(ideal[:, :, ::-1])
    axes[0].set_title("理想")
    axes[0].axis("off")
    axes[1].imshow(bent[:, :, ::-1])
    axes[1].set_title(f"你的畸变 k1={MY_K1}")
    axes[1].axis("off")
    axes[2].imshow(fixed[:, :, ::-1])
    axes[2].set_title("undistort")
    axes[2].axis("off")
    save_fig(fig, "ch04_my_undistort.png")
    save_bgr("ch04_my_bent.png", bent)


def self_check():
    K = true_K()
    assert K.shape == (3, 3)
    assert abs(K[0, 0] - TRUE_FX) < 1e-9

    # undistort 理想点应近似回到附近（零畸变时恒等）
    zero = np.zeros(5, dtype=np.float64)
    img = make_grid_scene()
    out = cv2.undistort(img, K, zero)
    assert out.shape == img.shape

    # 合成标定：估计 fx 应接近真值
    obj_list, img_list, _, _, _ = synthesize_views(n_views=10, seed=3)
    rms, K_est, dist_est, _, _ = cv2.calibrateCamera(obj_list, img_list, (IMG_W, IMG_H), None, None)
    assert rms < 1.0
    assert abs(K_est[0, 0] - TRUE_FX) < 40.0
    assert abs(dist_est[0, 0] - TRUE_K1) < 0.15

    # 畸变再去畸变：与理想图的中心区域应更接近
    dist = true_dist()
    bent = apply_distortion_fast(img, K, dist)
    fixed = cv2.undistort(bent, K, dist, None, K)
    cy0, cx0 = IMG_H // 2, IMG_W // 2
    patch_i = img[cy0 - 40 : cy0 + 40, cx0 - 40 : cx0 + 40].astype(np.float32)
    patch_f = fixed[cy0 - 40 : cy0 + 40, cx0 - 40 : cx0 + 40].astype(np.float32)
    patch_b = bent[cy0 - 40 : cy0 + 40, cx0 - 40 : cx0 + 40].astype(np.float32)
    err_fixed = np.mean(np.abs(patch_i - patch_f))
    err_bent = np.mean(np.abs(patch_i - patch_b))
    assert err_fixed < err_bent + 5.0  # 掰直后应不差于弯的（中心附近）

    print("自检通过：内参、标定 RMS、去畸变通路正常。")


def main():
    use_style()
    self_check()
    plot_why_undistort()
    plot_pinhole_intrinsics()
    plot_barrel_vs_pincushion()
    plot_before_after_undistort()
    plot_calib_pipeline()
    run_calibration_demo()
    plot_mine()
    print("课文对照图已写入 assets/figures/")
    print("你的图：ch04_my_undistort.png")


if __name__ == "__main__":
    main()
