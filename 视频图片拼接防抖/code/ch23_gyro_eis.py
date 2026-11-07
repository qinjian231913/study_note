"""第 23 篇：陀螺仪电子防抖与 Rolling Shutter 校正。

在「视频图片拼接防抖」文件夹里运行：

    MPLCONFIGDIR=/tmp/mpl-cache .venv/bin/python code/ch23_gyro_eis.py

顶部「改这里」只影响 ch23_my_*.png；课文对照图每次按固定参数重画。
"""

from pathlib import Path

SHOW_WINDOW = False

import matplotlib

if not SHOW_WINDOW:
    matplotlib.use("Agg")

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Rectangle

from plot_style import INK, MUTED, TEAL, WAVE, finish, use_style

WAVE_BGR = (40, 77, 194)
TEAL_BGR = (102, 107, 14)
INK_BGR = (21, 25, 28)

# ===== 改这里 =====
# 只影响 ch23_my_eis.png / ch23_my_rs.png / ch23_my_rs_fix.png
MY_SHAKE_DEG = 4.5        # 手持抖动峰峰值（度）。越大，未稳像越晃
MY_SMOOTH_WIN = 9         # 平滑窗口（奇数更合适）。越大目标姿态越「粘」、裁切越多
MY_OMEGA_YAW_DPS = 220.0  # RS 演示：偏航角速度（度/秒）。越大竖线越斜（教学用偏大）
MY_ROW_TIME_US = 55.0     # 相邻两行曝光间隔（微秒）。手机常见约 10–40 μs
MY_DELAY_MS = 0.0         # IMU 相对曝光中心的延迟（毫秒）。故意改成 2～8 看「越校正越歪」
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


# ---------- 姿态与旋转工具 ----------

def rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)


def rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)


def rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


def euler_zyx(yaw, pitch, roll):
    """相机姿态：先 yaw(绕竖轴) → pitch → roll。小角度时接近手持晃动。"""
    return rot_z(yaw) @ rot_y(pitch) @ rot_x(roll)


def so3_exp(w):
    """Rodrigues：角速度向量 w（弧度，轴角）→ 旋转矩阵。"""
    theta = float(np.linalg.norm(w))
    if theta < 1e-12:
        return np.eye(3)
    k = w / theta
    K = np.array(
        [[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]],
        dtype=np.float64,
    )
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)


def rotation_to_homography(R, K):
    """纯旋转稳像：H = K R K^{-1}。把「当前姿态」变到「参考姿态」。"""
    return K @ R @ np.linalg.inv(K)


def smooth_angles(angles, win):
    """对欧拉角做滑动平均，当作「平滑目标姿态」。"""
    win = max(3, int(win) | 1)  # 强制奇数
    k = np.ones(win) / win
    out = np.convolve(angles, k, mode="same")
    half = win // 2
    out[:half] = angles[:half]
    out[-half:] = angles[-half:]
    return out


def make_shake_sequence(n=48, shake_deg=4.0, seed=7):
    """合成手持姿态：缓慢运镜 + 高频抖动（弧度）。"""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, n)
    # 故意慢转（运镜）
    yaw0 = np.deg2rad(8.0) * t
    pitch0 = np.deg2rad(2.0) * np.sin(np.pi * t)
    roll0 = np.zeros(n)
    # 高频抖动
    amp = np.deg2rad(shake_deg)
    yaw = yaw0 + amp * 0.55 * np.sin(2 * np.pi * 6.5 * t) + amp * 0.2 * rng.normal(0, 1, n)
    pitch = pitch0 + amp * 0.7 * np.sin(2 * np.pi * 5.2 * t + 0.3) + amp * 0.18 * rng.normal(0, 1, n)
    roll = roll0 + amp * 0.35 * np.sin(2 * np.pi * 7.1 * t + 1.1) + amp * 0.12 * rng.normal(0, 1, n)
    return t, yaw, pitch, roll, yaw0, pitch0, roll0


def camera_K(w, h, fov_deg=60.0):
    f = 0.5 * w / np.tan(np.deg2rad(fov_deg) / 2)
    return np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], dtype=np.float64)


# ---------- 合成场景与成像 ----------

def make_world_grid(w=640, h=400):
    """远景「墙」：竖条 + 楼 + 路面，RS 斜切时一眼能看出歪。"""
    img = np.full((h, w, 3), 236, dtype=np.uint8)
    img[: h // 2] = (228, 220, 208)
    img[h // 2 :] = (200, 188, 170)
    # 远山
    hills = np.array([[0, 190], [100, 120], [240, 170], [380, 100], [520, 160], [640, 180], [640, 200], [0, 200]], np.int32)
    cv2.fillPoly(img, [hills], (175, 165, 150))
    # 竖条楼——RS 时会变斜
    for x, bw, bh, color in (
        (60, 36, 150, (85, 105, 145)),
        (140, 30, 120, (70, 95, 130)),
        (260, 44, 170, (100, 90, 70)),
        (360, 34, 130, (75, 115, 105)),
        (470, 40, 155, (105, 85, 95)),
        (560, 38, 140, (90, 100, 120)),
    ):
        top = h // 2 - bh
        cv2.rectangle(img, (x, top), (x + bw, h // 2), color, -1)
        cv2.rectangle(img, (x, top), (x + bw, h // 2), (40, 40, 40), 1)
        # 楼内窗格
        for wy in range(top + 12, h // 2 - 8, 18):
            cv2.rectangle(img, (x + 6, wy), (x + bw - 6, wy + 8), (210, 200, 180), -1)
    # 几条醒目竖线
    for x in (100, 220, 320, 440, 540):
        cv2.line(img, (x, 40), (x, h - 40), (50, 45, 40), 2)
    # 路面虚线
    y = h // 2 + 50
    for x0 in range(20, w - 20, 48):
        cv2.line(img, (x0, y), (x0 + 26, y), (250, 250, 250), 3)
    return img


def project_grid_with_rotation(R, out_w=320, out_h=240, fov_deg=60.0, src=None):
    """用纯旋转把「虚拟墙」投到相机平面。

    约定：R 把相机坐标变到「墙/世界」坐标。输出像素射线 d=K^{-1}p，
    投到源图：p_src ~ K_src R^{-1} d，即 warp 矩阵 H = K_src R^{-1} K^{-1}。
    """
    if src is None:
        src = make_world_grid()
    sh, sw = src.shape[:2]
    K = camera_K(out_w, out_h, fov_deg)
    K_src = camera_K(sw, sh, fov_deg)
    H = K_src @ np.linalg.inv(R) @ np.linalg.inv(K)
    return cv2.warpPerspective(
        src,
        H,
        (out_w, out_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(245, 245, 245),
    )


def apply_frame_eis(frame, R_cam, R_target, K):
    """相对平滑目标姿态做 warp：把当前姿态扳回目标。"""
    R_rel = R_target.T @ R_cam
    H = rotation_to_homography(np.linalg.inv(R_rel), K)
    h, w = frame.shape[:2]
    return cv2.warpPerspective(
        frame,
        H,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(245, 245, 245),
    )


def row_shift_pixels(omega_yaw_rad_s, row_time_s, y, mid, focal_px, delay_s=0.0):
    """一阶 RS 模型：偏航角速度 → 第 y 行相对中心行的水平像素位移。

    白话：转过的小角度 θ≈ω·Δt，在焦距 f 上大约挪 f·θ 个像素。
    """
    t = delay_s + (y - mid) * row_time_s
    return float(focal_px * omega_yaw_rad_s * t)


def simulate_rolling_shutter(gs_frame, omega_yaw_rad_s, row_time_s, delay_s=0.0, fov_deg=60.0):
    """从一张全局快门图出发，按行水平挪动，合成「只有偏航」的 RS 斜切。

    这是教学用一阶近似；完整做法是每行一个旋转单应 H(y)=K R(y) K^{-1}。
    """
    h, w = gs_frame.shape[:2]
    K = camera_K(w, h, fov_deg)
    f = float(K[0, 0])
    mid = (h - 1) / 2.0
    out = np.full_like(gs_frame, 245)
    xs = np.arange(w, dtype=np.float32)
    ys = np.zeros(w, dtype=np.float32)
    for y in range(h):
        dx = row_shift_pixels(omega_yaw_rad_s, row_time_s, y, mid, f, delay_s)
        map_x = (xs - dx).reshape(1, w)  # 输出 x 来自源 x-dx？内容右移时源要从左边取
        # 若整行内容应向右移 dx：out[y,x] = src[y, x-dx]
        map_y = np.full((1, w), y, dtype=np.float32)
        row = cv2.remap(
            gs_frame,
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(245, 245, 245),
        )
        out[y] = row[0]
    return out


def correct_rolling_shutter(rs_frame, omega_yaw_rad_s, row_time_s, delay_s=0.0, fov_deg=60.0):
    """按行反向平移，把斜切掰直（与 simulate 同一公式、相反符号）。"""
    h, w = rs_frame.shape[:2]
    K = camera_K(w, h, fov_deg)
    f = float(K[0, 0])
    mid = (h - 1) / 2.0
    out = np.full_like(rs_frame, 245)
    xs = np.arange(w, dtype=np.float32)
    for y in range(h):
        dx = row_shift_pixels(omega_yaw_rad_s, row_time_s, y, mid, f, delay_s)
        # 合成时 out=src(x-dx)；还原时 corrected(x)=rs(x+dx)
        map_x = (xs + dx).reshape(1, w)
        map_y = np.full((1, w), y, dtype=np.float32)
        row = cv2.remap(
            rs_frame,
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(245, 245, 245),
        )
        out[y] = row[0]
    return out


def vertical_line_span(image_bgr, x_guess=160, half_w=80):
    """量一根竖线的歪斜：每行在搜索带内最暗列的水平跨度（像素）。"""
    h, w = image_bgr.shape[:2]
    x0 = max(0, int(x_guess - half_w))
    x1 = min(w, int(x_guess + half_w))
    band = image_bgr[:, x0:x1, 0].astype(np.float32)
    useful = band.min(axis=1) < 100
    if useful.sum() < 20:
        return 0.0
    xs = np.argmin(band, axis=1).astype(np.float64) + x0
    xs = xs[useful]
    return float(xs.max() - xs.min())


def make_line_probe(w=320, h=240):
    """自检专用：白底一根竖线，量歪斜最干净。"""
    img = np.full((h, w, 3), 245, dtype=np.uint8)
    cv2.line(img, (w // 2, 10), (w // 2, h - 10), (20, 20, 20), 3)
    return img


# ---------- 作图 ----------

def plot_pipeline():
    fig, ax = plt.subplots(figsize=(10.2, 3.4), layout="constrained")
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 3.2)
    ax.axis("off")
    ax.set_title("陀螺 EIS 流水线（本篇主线）")

    boxes = [
        (0.2, 1.1, 2.0, 1.2, "角速度 ω(t)\n陀螺仪", WAVE),
        (2.6, 1.1, 2.0, 1.2, "积分 → 姿态\nR_cam(t)", WAVE),
        (5.0, 1.1, 2.2, 1.2, "平滑目标\nR_target(t)", TEAL),
        (7.6, 1.1, 2.0, 1.2, "相对旋转\nR_rel", TEAL),
        (10.0, 1.1, 1.8, 1.2, "H=KRK⁻¹\nwarp", INK),
    ]
    for x, y, w, h, text, color in boxes:
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05,rounding_size=0.15",
                                    facecolor="#fffdf8", edgecolor=color, lw=1.8))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=11, color=INK)
    for x0 in (2.2, 4.6, 7.2, 9.6):
        ax.annotate("", xy=(x0 + 0.35, 1.7), xytext=(x0, 1.7),
                    arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.6))
    ax.text(6.0, 0.35, "RS 校正：把「每帧一个 R」换成「每一行一个 R(y)」", ha="center", fontsize=11, color=MUTED)
    save_fig(fig, "ch23_pipeline.png")


def plot_attitude_paths():
    t, yaw, pitch, roll, *_ = make_shake_sequence(shake_deg=4.0)
    sy = smooth_angles(yaw, 9)
    sp = smooth_angles(pitch, 9)
    fig, axes = plt.subplots(2, 1, figsize=(9.6, 5.4), layout="constrained", sharex=True)

    ax = axes[0]
    ax.plot(t, np.rad2deg(yaw), color=WAVE, lw=1.6, label="原始 yaw（陀螺积分）")
    ax.plot(t, np.rad2deg(sy), color=TEAL, lw=2.2, label="平滑目标 yaw")
    finish(ax)
    ax.set_ylabel("偏航角（度）")
    ax.set_title("姿态序列：抖的是高频，目标是压高频、留运镜")
    ax.legend(frameon=False, fontsize=10)

    ax = axes[1]
    ax.plot(t, np.rad2deg(pitch), color=WAVE, lw=1.6, label="原始 pitch")
    ax.plot(t, np.rad2deg(sp), color=TEAL, lw=2.2, label="平滑目标 pitch")
    ax.plot(t, np.rad2deg(roll), color=MUTED, lw=1.2, alpha=0.8, label="原始 roll")
    finish(ax)
    ax.set_xlabel("时间（归一化）")
    ax.set_ylabel("角度（度）")
    ax.legend(frameon=False, fontsize=10)
    save_fig(fig, "ch23_attitude_paths.png")


def plot_eis_frames(shake_deg=4.0, smooth_win=9, name="ch23_warp_compare.png", title_extra=""):
    src = make_world_grid()
    t, yaw, pitch, roll, *_ = make_shake_sequence(n=36, shake_deg=shake_deg)
    sy, sp, sr = smooth_angles(yaw, smooth_win), smooth_angles(pitch, smooth_win), smooth_angles(roll, smooth_win)
    idxs = [4, 14, 24]
    raw_frames = []
    eis_frames = []
    K = camera_K(320, 240)
    for i in idxs:
        R = euler_zyx(yaw[i], pitch[i], roll[i])
        Rt = euler_zyx(sy[i], sp[i], sr[i])
        fr = project_grid_with_rotation(R, src=src)
        st = apply_frame_eis(fr, R, Rt, K)
        # 标帧号
        for img, label_color in ((fr, WAVE_BGR), (st, TEAL_BGR)):
            cv2.putText(img, f"t={i}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, label_color, 2, cv2.LINE_AA)
        raw_frames.append(fr)
        eis_frames.append(st)

    fig, axes = plt.subplots(2, 3, figsize=(10.5, 5.6), layout="constrained")
    for j, fr in enumerate(raw_frames):
        axes[0, j].imshow(fr[:, :, ::-1])
        axes[0, j].set_title(f"未稳像 #{idxs[j]}")
        axes[0, j].axis("off")
    for j, fr in enumerate(eis_frames):
        axes[1, j].imshow(fr[:, :, ::-1])
        axes[1, j].set_title(f"陀螺 EIS #{idxs[j]}")
        axes[1, j].axis("off")
    fig.suptitle(f"相对平滑目标姿态做 warp{title_extra}", fontsize=13)
    save_fig(fig, name)

    # OpenCV 横条：晃 vs 稳
    gap = 8
    h, w = raw_frames[0].shape[:2]
    canvas = np.full((h * 2 + gap + 28, 3 * w + 2 * gap, 3), 251, dtype=np.uint8)
    for j, (a, b) in enumerate(zip(raw_frames, eis_frames)):
        x0 = j * (w + gap)
        canvas[28 : 28 + h, x0 : x0 + w] = a
        canvas[28 + h + gap : 28 + h + gap + h, x0 : x0 + w] = b
    cv2.putText(canvas, "raw shake", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, WAVE_BGR, 2, cv2.LINE_AA)
    cv2.putText(canvas, "gyro EIS", (8, 28 + h + gap - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, TEAL_BGR, 2, cv2.LINE_AA)
    save_bgr("ch23_synth_strip.png", canvas)
    return raw_frames, eis_frames


def plot_visual_vs_gyro():
    fig, ax = plt.subplots(figsize=(10.0, 4.6), layout="constrained")
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.set_title("纯视觉 EIS vs 陀螺 EIS（对照表）")

    ax.add_patch(FancyBboxPatch((0.3, 0.4), 4.5, 5.0, boxstyle="round,pad=0.08,rounding_size=0.12",
                                facecolor="#fde8df", edgecolor=WAVE, lw=1.6))
    ax.add_patch(FancyBboxPatch((5.2, 0.4), 4.5, 5.0, boxstyle="round,pad=0.08,rounding_size=0.12",
                                facecolor="#e5f3f1", edgecolor=TEAL, lw=1.6))
    ax.text(2.55, 5.0, "纯视觉 EIS", ha="center", fontsize=14, color=WAVE, fontweight="bold")
    ax.text(7.45, 5.0, "陀螺仪 EIS", ha="center", fontsize=14, color=TEAL, fontweight="bold")

    left = [
        "靠特征 / 光流估帧间变换",
        "低纹理、糊、夜景：容易丢",
        "快速甩动：跟踪断、估错",
        "可观察到漂移并慢慢拉回",
        "不需要 IMU 外参与延迟",
    ]
    right = [
        "靠角速度积分得姿态",
        "与画面内容无关，糊也能稳",
        "高频抖动响应快",
        "长时间会漂（需视觉补）",
        "必须标定：外参 + 时间延迟",
    ]
    for i, (L, R) in enumerate(zip(left, right)):
        y = 4.3 - i * 0.7
        ax.text(0.55, y, "· " + L, fontsize=11, va="center")
        ax.text(5.45, y, "· " + R, fontsize=11, va="center")
    ax.text(5.0, 0.15, "工程上常「陀螺扛高频 + 视觉修低频/漂移」——视觉-惯性入口", ha="center", fontsize=11, color=MUTED)
    save_fig(fig, "ch23_visual_vs_gyro.png")


def plot_rs_pair(omega_yaw_dps=220.0, row_time_us=55.0, delay_ms=0.0, name_skew="ch23_rs_skew.png", name_fix="ch23_rs_corrected.png"):
    """演示参数故意偏大，让竖线斜切在 320×240 上肉眼可见。"""
    src = make_world_grid()
    gs = project_grid_with_rotation(np.eye(3), src=src)
    # 在 GS 图上画几根超明显的竖线，方便量歪斜
    for x in (80, 160, 240):
        cv2.line(gs, (x, 20), (x, 220), (30, 30, 30), 2)
    omega = np.deg2rad(omega_yaw_dps)
    row_t = row_time_us * 1e-6
    delay = delay_ms * 1e-3
    rs = simulate_rolling_shutter(gs, omega, row_t, delay_s=delay)
    fixed = correct_rolling_shutter(rs, omega, row_t, delay_s=delay)

    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.6), layout="constrained")
    for ax, im, title in (
        (axes[0], gs, "全局快门（对照）"),
        (axes[1], rs, f"RS：ω={omega_yaw_dps:.0f}°/s"),
        (axes[2], fixed, "按行平移校正后"),
    ):
        ax.imshow(im[:, :, ::-1])
        ax.set_title(title)
        ax.axis("off")
    span_rs = vertical_line_span(rs, x_guess=160)
    span_fix = vertical_line_span(fixed, x_guess=160)
    fig.suptitle(
        f"行时间={row_time_us:.0f} μs，延迟={delay_ms:.1f} ms；中线跨度 {span_rs:.1f}→{span_fix:.1f} px",
        fontsize=12,
    )
    save_fig(fig, name_skew)

    fig2, axes2 = plt.subplots(1, 2, figsize=(8.8, 3.5), layout="constrained")
    axes2[0].imshow(rs[:, :, ::-1])
    axes2[0].set_title("卷帘：竖线被拧斜")
    axes2[0].axis("off")
    axes2[1].imshow(fixed[:, :, ::-1])
    axes2[1].set_title("陀螺 RS 校正：按行扳回")
    axes2[1].axis("off")
    save_fig(fig2, name_fix)
    save_bgr("ch23_rs_demo.png", np.hstack([rs, np.full((rs.shape[0], 8, 3), 251, np.uint8), fixed]))
    return rs, fixed


def plot_extrinsic_delay():
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.0), layout="constrained")

    ax = axes[0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.axis("off")
    ax.set_title("外参：陀螺坐标系 → 相机坐标系")
    # 手机示意
    ax.add_patch(Rectangle((3.2, 1.5), 3.6, 4.2, fill=False, ec=INK, lw=2))
    ax.plot([5.0], [3.6], "o", color=WAVE, ms=10)
    ax.text(5.0, 4.2, "陀螺仪", ha="center", color=WAVE, fontsize=11)
    ax.annotate("", xy=(7.5, 5.2), xytext=(5.3, 3.9),
                arrowprops=dict(arrowstyle="->", color=TEAL, lw=2))
    ax.text(7.7, 5.4, "相机光轴", color=TEAL, fontsize=11)
    ax.text(5.0, 0.6, r"$R_{cam}=R_{ext}\,R_{imu}$", ha="center", fontsize=13)
    ax.text(5.0, 0.15, "装反、绕轴差 90°，稳像会「越稳越歪」", ha="center", fontsize=10, color=MUTED)

    ax = axes[1]
    ax.set_title("时间延迟：IMU 与曝光中心没对齐")
    t = np.linspace(0, 1, 200)
    omega = np.sin(2 * np.pi * 3 * t)
    delay = 0.08
    ax.plot(t, omega, color=WAVE, lw=1.8, label="真实角速度")
    ax.plot(t, np.sin(2 * np.pi * 3 * (t - delay)), color=TEAL, lw=1.8, label="带延迟的读数")
    ax.axvline(0.5, color=MUTED, ls="--", lw=1)
    ax.text(0.52, 0.7, "某帧曝光中心", color=MUTED, fontsize=10)
    finish(ax)
    ax.set_xlabel("时间")
    ax.set_ylabel("ω")
    ax.legend(frameon=False, fontsize=10)
    save_fig(fig, "ch23_extrinsic_delay.png")


def plot_vio_entry():
    fig, ax = plt.subplots(figsize=(9.8, 3.8), layout="constrained")
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 4)
    ax.axis("off")
    ax.set_title("和光流联合的入口（视觉-惯性稳像）")

    items = [
        (0.3, "陀螺\n高频姿态"),
        (3.0, "光流/特征\n帧间变换"),
        (5.7, "融合\n（互补/滤波）"),
        (8.5, "平滑目标\n+ warp"),
    ]
    for x, text in items:
        ax.add_patch(FancyBboxPatch((x, 1.2), 2.1, 1.6, boxstyle="round,pad=0.06,rounding_size=0.12",
                                    facecolor="#fffdf8", edgecolor=TEAL if x > 4 else WAVE, lw=1.7))
        ax.text(x + 1.05, 2.0, text, ha="center", va="center", fontsize=12)
    for x0 in (2.4, 5.1, 7.8):
        ax.annotate("", xy=(x0 + 0.55, 2.0), xytext=(x0, 2.0),
                    arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.6))
    ax.text(5.5, 0.55, "直觉：陀螺告诉你「这一瞬间怎么转」，光流告诉你「画面长期漂了多少」",
            ha="center", fontsize=11, color=MUTED)
    ax.text(5.5, 0.15, "下一层（第 24 篇）还会把路径放到 3D 虚拟相机里规划", ha="center", fontsize=10, color=MUTED)
    save_fig(fig, "ch23_vio_entry.png")


def plot_mine():
    plot_eis_frames(
        shake_deg=MY_SHAKE_DEG,
        smooth_win=MY_SMOOTH_WIN,
        name="ch23_my_eis.png",
        title_extra=f"（shake={MY_SHAKE_DEG}°, win={MY_SMOOTH_WIN}）",
    )
    plot_rs_pair(
        omega_yaw_dps=MY_OMEGA_YAW_DPS,
        row_time_us=MY_ROW_TIME_US,
        delay_ms=MY_DELAY_MS,
        name_skew="ch23_my_rs.png",
        name_fix="ch23_my_rs_fix.png",
    )


def self_check():
    # 1) 小角度 Rodrigues ≈ I + [w]_x
    w = np.array([0.01, -0.02, 0.015])
    R = so3_exp(w)
    if abs(np.linalg.det(R) - 1.0) > 1e-6:
        raise SystemExit("so3_exp 行列式应≈1")
    if np.linalg.norm(R @ R.T - np.eye(3)) > 1e-6:
        raise SystemExit("so3_exp 应正交")

    # 2) 纯旋转单应：零旋转 → 恒等
    K = camera_K(320, 240)
    H0 = rotation_to_homography(np.eye(3), K)
    if np.linalg.norm(H0 / H0[2, 2] - np.eye(3)) > 1e-8:
        raise SystemExit("零旋转单应应为恒等")

    # 3) 平滑窗口不增大直流趋势的均值过多
    t, yaw, *_ = make_shake_sequence(n=40, shake_deg=3.0)
    sy = smooth_angles(yaw, 7)
    if abs(sy.mean() - yaw.mean()) > np.deg2rad(0.5):
        raise SystemExit("平滑不应显著改变平均姿态")

    # 4) RS：有角速度时应比静止更「斜」
    probe = make_line_probe()
    still = simulate_rolling_shutter(probe, 0.0, 55e-6)
    omega = np.deg2rad(220.0)
    moving = simulate_rolling_shutter(probe, omega, 55e-6)
    if np.allclose(still, moving):
        raise SystemExit("有角速度的 RS 图不应与静止相同")
    span_still = vertical_line_span(still)
    span_move = vertical_line_span(moving)
    if span_move <= span_still + 8.0:
        raise SystemExit(f"RS 应变斜：still={span_still:.1f}, move={span_move:.1f}")

    # 5) 校正后竖线应掰直；角速度估错（例如只用了一半）会留下残余斜切
    fixed = correct_rolling_shutter(moving, omega, 55e-6)
    span_fix = vertical_line_span(fixed)
    if not (span_fix < max(3.0, span_move * 0.25)):
        raise SystemExit(f"RS 校正应掰直竖线：move={span_move:.1f}, fix={span_fix:.1f}")
    under = correct_rolling_shutter(moving, omega * 0.45, 55e-6)
    span_under = vertical_line_span(under)
    if span_under <= span_fix + 5.0:
        raise SystemExit(f"低估角速度应留下斜切：fix={span_fix:.1f}, under={span_under:.1f}")

    print("自检通过：旋转、单应、平滑、RS 合成与校正均正常。")


def main():
    use_style()
    self_check()
    plot_pipeline()
    plot_attitude_paths()
    plot_eis_frames(shake_deg=4.0, smooth_win=9)
    plot_visual_vs_gyro()
    plot_rs_pair(omega_yaw_dps=220.0, row_time_us=55.0, delay_ms=0.0)
    plot_extrinsic_delay()
    plot_vio_entry()
    plot_mine()
    print("课文对照图已写入 assets/figures/ 与 assets/images/")
    print("你的图：ch23_my_eis.png、ch23_my_rs.png、ch23_my_rs_fix.png（只跟随顶部参数）")


if __name__ == "__main__":
    main()
