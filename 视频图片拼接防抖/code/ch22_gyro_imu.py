"""第 22 篇：陀螺仪与 IMU——角速度如何变成姿态。

    MPLCONFIGDIR=/tmp/mpl-cache .venv/bin/python code/ch22_gyro_imu.py

顶部「改这里」只影响 ch22_my_*.png。
课文对照图与合成 IMU 日志按固定参数生成。
"""

from pathlib import Path

SHOW_WINDOW = False

import matplotlib

if not SHOW_WINDOW:
    matplotlib.use("Agg")

import csv

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

from plot_style import MUTED, TEAL, WAVE, finish, use_style

# ===== 改这里 =====
# 陀螺仪零偏（°/s），加在真角速度上再积分 → 姿态会慢慢飘
GYRO_BIAS_DEG = 0.8
# 互补滤波：越接近 1，越信陀螺；越小，越信「姿态观测」（本例用加速度计估倾角）
COMP_ALPHA = 0.98
# 视频帧曝光时长（毫秒）。对齐时取曝光中心 = 快门打开时刻 + 曝光/2
EXPOSURE_MS = 10.0
# ==================

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "assets" / "figures"
IMG = ROOT / "assets" / "images"

DEG2RAD = np.pi / 180.0
RAD2DEG = 180.0 / np.pi


def save_fig(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    path = FIG / name
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"已保存 {path}")


def save_text(name, text):
    IMG.mkdir(parents=True, exist_ok=True)
    path = IMG / name
    path.write_text(text, encoding="utf-8")
    print(f"已保存 {path}")


# ---------------------------------------------------------------------------
# 合成运动：一段「手持拍视频」——缓慢左右摇 + 高频手抖 + 一次甩头
# ---------------------------------------------------------------------------

def true_motion(t):
    """返回真姿态(°)：roll, pitch, yaw；以及真角速度(°/s)。"""
    # 慢摇（故意运镜）+ 手抖 + 中段甩头
    yaw = 8.0 * np.sin(0.7 * t) + 1.2 * np.sin(18.0 * t)
    pitch = 3.0 * np.sin(0.9 * t + 0.4) + 0.6 * np.sin(22.0 * t)
    roll = 2.0 * np.sin(1.1 * t + 1.0) + 0.4 * np.sin(25.0 * t)

    # 约 t∈[1.8, 2.3] 甩一下头（yaw 快速变化）
    sway = 25.0 * np.exp(-((t - 2.05) ** 2) / (2 * 0.08**2))
    yaw = yaw + sway

    # 数值微分得角速度（对合成函数也可用解析导数；这里统一用差分思路的解析式）
    wy = (
        8.0 * 0.7 * np.cos(0.7 * t)
        + 1.2 * 18.0 * np.cos(18.0 * t)
        + 25.0 * (-(t - 2.05) / (0.08**2)) * np.exp(-((t - 2.05) ** 2) / (2 * 0.08**2))
    )
    wp = 3.0 * 0.9 * np.cos(0.9 * t + 0.4) + 0.6 * 22.0 * np.cos(22.0 * t)
    wr = 2.0 * 1.1 * np.cos(1.1 * t + 1.0) + 0.4 * 25.0 * np.cos(25.0 * t)
    return np.array([roll, pitch, yaw]), np.array([wr, wp, wy])


def simulate_imu(
    duration=4.0,
    fs_imu=200.0,
    bias_deg=None,
    noise_std_deg=0.15,
    seed=7,
):
    """生成 IMU 采样：时间戳(s)、角速度测量(°/s)、加速度(g)、真姿态(°)。"""
    if bias_deg is None:
        bias_deg = np.array([0.05, -0.03, 0.4])  # roll,pitch,yaw 零偏 °/s
    rng = np.random.default_rng(seed)
    n = int(duration * fs_imu) + 1
    t = np.arange(n) / fs_imu
    omega_meas = np.zeros((n, 3))
    accel = np.zeros((n, 3))
    attitude_true = np.zeros((n, 3))

    for i, ti in enumerate(t):
        att, w_true = true_motion(ti)
        attitude_true[i] = att
        noise = rng.normal(0.0, noise_std_deg, size=3)
        omega_meas[i] = w_true + bias_deg + noise

        # 简化：静止附近加速度计主要感受重力在机体系的投影（只用 roll/pitch）
        r, p = att[0] * DEG2RAD, att[1] * DEG2RAD
        # 机体系加速度（g）：忽略线加速
        accel[i] = np.array(
            [
                -np.sin(p),
                np.sin(r) * np.cos(p),
                np.cos(r) * np.cos(p),
            ]
        )
        accel[i] += rng.normal(0.0, 0.01, size=3)

    return t, omega_meas, accel, attitude_true, np.asarray(bias_deg, dtype=float)


def integrate_gyro(t, omega_deg, att0=None):
    """梯形积分：姿态(°) = 累加 ω·Δt。仅作小角度演示，轴近似解耦。"""
    n = len(t)
    att = np.zeros((n, 3))
    if att0 is not None:
        att[0] = att0
    for i in range(1, n):
        dt = t[i] - t[i - 1]
        att[i] = att[i - 1] + 0.5 * (omega_deg[i] + omega_deg[i - 1]) * dt
    return att


def accel_to_tilt_deg(accel):
    """用加速度计估 roll/pitch（°）。yaw 无法从重力得到。"""
    ax, ay, az = accel[:, 0], accel[:, 1], accel[:, 2]
    # 与 simulate 中重力投影一致的反解
    pitch = np.arcsin(np.clip(-ax, -1.0, 1.0)) * RAD2DEG
    roll = np.arctan2(ay, az) * RAD2DEG
    yaw = np.zeros_like(roll)  # 不可观
    return np.column_stack([roll, pitch, yaw])


def complementary_filter(t, omega_deg, accel, alpha=0.98, att0=None):
    """一阶互补滤波直觉版：高频信陀螺积分，低频用加速度计倾角拉回。"""
    n = len(t)
    att = np.zeros((n, 3))
    tilt = accel_to_tilt_deg(accel)
    if att0 is not None:
        att[0] = att0
    else:
        att[0] = tilt[0].copy()
        att[0, 2] = 0.0  # yaw 仍从 0 积
    for i in range(1, n):
        dt = t[i] - t[i - 1]
        gyro_pred = att[i - 1] + 0.5 * (omega_deg[i] + omega_deg[i - 1]) * dt
        # roll/pitch：融合；yaw：只能信陀螺（本例无磁力计）
        att[i, 0] = alpha * gyro_pred[0] + (1.0 - alpha) * tilt[i, 0]
        att[i, 1] = alpha * gyro_pred[1] + (1.0 - alpha) * tilt[i, 1]
        att[i, 2] = gyro_pred[2]
    return att


def write_imu_log(path, t, omega_deg, accel):
    """写合成 IMU 日志：时间戳微秒 + 角速度 rad/s + 加速度 g。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "timestamp_us",
                "gx_rad_s",
                "gy_rad_s",
                "gz_rad_s",
                "ax_g",
                "ay_g",
                "az_g",
            ]
        )
        t0_us = 1_700_000_000_000_000  # 任意起点
        for i in range(len(t)):
            ts = t0_us + int(round(t[i] * 1e6))
            gx, gy, gz = omega_deg[i] * DEG2RAD
            ax, ay, az = accel[i]
            w.writerow(
                [
                    ts,
                    f"{gx:.8f}",
                    f"{gy:.8f}",
                    f"{gz:.8f}",
                    f"{ax:.6f}",
                    f"{ay:.6f}",
                    f"{az:.6f}",
                ]
            )
    print(f"已保存 {path}")


def load_imu_log(path):
    """读回日志，返回 t(s 从 0 起)、omega(°/s)、accel(g)。"""
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    ts = np.array([int(r["timestamp_us"]) for r in rows], dtype=np.int64)
    t = (ts - ts[0]) * 1e-6
    omega = (
        np.column_stack(
            [
                [float(r["gx_rad_s"]) for r in rows],
                [float(r["gy_rad_s"]) for r in rows],
                [float(r["gz_rad_s"]) for r in rows],
            ]
        )
        * RAD2DEG
    )
    accel = np.column_stack(
        [
            [float(r["ax_g"]) for r in rows],
            [float(r["ay_g"]) for r in rows],
            [float(r["az_g"]) for r in rows],
        ]
    )
    return t, omega, accel


# ---------------------------------------------------------------------------
# 作图
# ---------------------------------------------------------------------------

def plot_three_axes():
    fig, ax = plt.subplots(figsize=(8.6, 4.2), layout="constrained")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis("off")
    ax.set_title("手机里的三轴角速度：陀螺仪测的是「转多快」，不是「朝哪」")

    # 手机轮廓
    phone = FancyBboxPatch(
        (3.6, 1.2), 2.8, 2.6, boxstyle="round,pad=0.08,rounding_size=0.2",
        facecolor="#f3e6d8", edgecolor="#1c1915", lw=1.6,
    )
    ax.add_patch(phone)
    ax.text(5.0, 3.5, "屏幕朝你", ha="center", fontsize=11, color=MUTED)

    # 轴箭头（机体系示意）
    ox, oy = 5.0, 2.4
    ax.annotate("", xy=(ox + 1.8, oy), xytext=(ox, oy),
                arrowprops=dict(arrowstyle="->", color=WAVE, lw=2))
    ax.text(ox + 1.9, oy + 0.15, "X / roll\n绕光轴滚", fontsize=11, color=WAVE)

    ax.annotate("", xy=(ox, oy + 1.3), xytext=(ox, oy),
                arrowprops=dict(arrowstyle="->", color=TEAL, lw=2))
    ax.text(ox + 0.15, oy + 1.35, "Y / pitch\n抬头低头", fontsize=11, color=TEAL)

    ax.annotate("", xy=(ox - 1.5, oy + 0.9), xytext=(ox, oy),
                arrowprops=dict(arrowstyle="->", color="#1c1915", lw=2))
    ax.text(ox - 2.6, oy + 1.0, "Z / yaw\n左右转头", fontsize=11)

    ax.text(
        0.4, 0.55,
        "输出单位常见：rad/s 或 °/s。三个数合在一起叫角速度向量 ω。\n"
        "姿态（朝哪）要靠「对角速度积分」才能得到——就像速度积分得路程。",
        fontsize=12, color=MUTED, va="bottom",
    )
    save_fig(fig, "ch22_axes.png")


def plot_omega_curves(t, omega, attitude_true):
    fig, axes = plt.subplots(2, 1, figsize=(9.2, 5.6), layout="constrained", sharex=True)
    labels = ["roll (绕 X)", "pitch (绕 Y)", "yaw (绕 Z)"]
    colors = [WAVE, TEAL, "#1c1915"]
    for k in range(3):
        axes[0].plot(t, omega[:, k], color=colors[k], lw=1.2, label=labels[k])
    finish(axes[0])
    axes[0].set_ylabel("角速度 (°/s)")
    axes[0].set_title("合成 IMU：三轴角速度曲线（含轻微噪声与零偏）")
    axes[0].legend(loc="upper right", fontsize=10, frameon=False)

    for k in range(3):
        axes[1].plot(t, attitude_true[:, k], color=colors[k], lw=1.4, label=labels[k])
    finish(axes[1])
    axes[1].set_xlabel("时间 (s)")
    axes[1].set_ylabel("真姿态 (°)")
    axes[1].set_title("同一段运动的真姿态（脚本里已知；真机没有这个「标准答案」）")
    axes[1].legend(loc="upper right", fontsize=10, frameon=False)
    # 标注甩头
    axes[0].axvspan(1.85, 2.25, color="#fde8df", alpha=0.7, zorder=0)
    axes[0].text(2.05, axes[0].get_ylim()[1] * 0.85, "甩头", ha="center", color=WAVE, fontsize=11)
    save_fig(fig, "ch22_omega_curve.png")


def plot_integrate_drift(t, omega, attitude_true, bias_extra_yaw_deg=0.0):
    """对比：无偏理想积分 vs 带零偏积分。"""
    # 理想：用真角速度积分；带偏：用测量（含零偏+噪声）再叠加可调 yaw 偏
    w_true = np.zeros_like(omega)
    for i, ti in enumerate(t):
        _, wi = true_motion(ti)
        w_true[i] = wi

    att_ideal = integrate_gyro(t, w_true, att0=attitude_true[0])
    omega_biased = omega.copy()
    omega_biased[:, 2] += bias_extra_yaw_deg
    att_drift = integrate_gyro(t, omega_biased, att0=attitude_true[0])

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.0), layout="constrained")
    axes[0].plot(t, attitude_true[:, 2], color=MUTED, lw=1.6, label="真 yaw", ls="--")
    axes[0].plot(t, att_ideal[:, 2], color=TEAL, lw=1.4, label="理想积分（无偏）")
    axes[0].plot(t, att_drift[:, 2], color=WAVE, lw=1.4, label="带零偏积分")
    finish(axes[0])
    axes[0].set_xlabel("时间 (s)")
    axes[0].set_ylabel("yaw (°)")
    axes[0].set_title("积分得姿态：零偏会让角度「慢慢溜走」")
    axes[0].legend(loc="best", fontsize=9, frameon=False)

    err = att_drift[:, 2] - attitude_true[:, 2]
    axes[1].plot(t, err, color=WAVE, lw=1.5)
    finish(axes[1])
    axes[1].set_xlabel("时间 (s)")
    axes[1].set_ylabel("yaw 误差 (°)")
    axes[1].set_title("误差近似随时间线性涨（常值零偏 → 斜坡漂移）")
    # 斜率注释
    if t[-1] > 0:
        slope = (err[-1] - err[0]) / t[-1]
        axes[1].text(
            0.05, 0.9, f"斜率 ≈ {slope:.2f} °/s\n≈ 零偏量级",
            transform=axes[1].transAxes, fontsize=11, color=MUTED, va="top",
        )
    save_fig(fig, "ch22_integrate_drift.png")
    return att_ideal, att_drift


def plot_sample_vs_frame(t, omega):
    fs_imu = 1.0 / np.median(np.diff(t))
    fps = 30.0
    # 帧时间戳：曝光中心
    t_frame = np.arange(0.0, t[-1] + 1e-9, 1.0 / fps)

    fig, ax = plt.subplots(figsize=(9.2, 3.8), layout="constrained")
    ax.plot(t, omega[:, 2], color=TEAL, lw=1.0, alpha=0.85, label=f"IMU yaw 角速度 (~{fs_imu:.0f} Hz)")
    # 帧时刻竖线
    for i, tf in enumerate(t_frame):
        ax.axvline(tf, color=WAVE, lw=0.7, alpha=0.35 if i else 0.7, zorder=0)
    ax.plot(t_frame, np.interp(t_frame, t, omega[:, 2]), "o", color=WAVE, ms=4, label=f"视频帧时刻 ({fps:.0f} fps)")
    finish(ax)
    ax.set_xlim(1.7, 2.5)
    ax.set_xlabel("时间 (s)　（放大甩头段）")
    ax.set_ylabel("ω_yaw (°/s)")
    ax.set_title("采样率 vs 帧率：帧与帧之间，IMU 已经记了很多次转动")
    ax.legend(loc="upper right", fontsize=10, frameon=False)
    save_fig(fig, "ch22_sample_vs_frame.png")


def plot_timestamp_align(exposure_ms=10.0):
    fig, ax = plt.subplots(figsize=(9.0, 3.6), layout="constrained")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.axis("off")
    ax.set_title("时间戳对齐：视频帧要对准「曝光中心」，再去 IMU 时间轴上取值")

    # 时间轴
    ax.annotate("", xy=(9.5, 1.2), xytext=(0.5, 1.2),
                arrowprops=dict(arrowstyle="->", color="#1c1915", lw=1.4))
    ax.text(9.6, 1.2, "t", va="center", fontsize=12)

    # IMU ticks
    for x in np.linspace(1.0, 8.5, 16):
        ax.plot([x, x], [1.05, 1.35], color=TEAL, lw=1.2)
    ax.text(1.0, 0.55, "IMU 样本（密）", color=TEAL, fontsize=11)

    # exposure window
    t_open, t_close = 4.0, 4.0 + exposure_ms / 10.0  # 图上刻度夸张一点好看
    t_close = 4.0 + 1.2
    t_mid = 0.5 * (t_open + t_close)
    ax.add_patch(
        FancyBboxPatch(
            (t_open, 2.0), t_close - t_open, 0.7,
            boxstyle="round,pad=0.02,rounding_size=0.05",
            facecolor="#fde8df", edgecolor=WAVE, lw=1.5,
        )
    )
    ax.text(t_mid, 2.35, "一帧曝光窗口", ha="center", va="center", fontsize=11, color=WAVE)
    ax.plot([t_open, t_open], [1.2, 2.0], color=MUTED, ls="--", lw=1)
    ax.plot([t_close, t_close], [1.2, 2.0], color=MUTED, ls="--", lw=1)
    ax.plot([t_mid, t_mid], [1.2, 2.0], color=WAVE, lw=2)
    ax.scatter([t_mid], [1.2], s=60, color=WAVE, zorder=5)
    ax.text(t_mid, 1.55, "曝光中心\n(常用对齐点)", ha="center", fontsize=11, color=WAVE)

    ax.text(
        0.5, 3.4,
        "常见坑：① 相机时钟与 IMU 时钟不是同一个零点（要标定延迟）　"
        "② 用了「帧到达时刻」而不是曝光中心　③ 单位混用（µs / ns / s）",
        fontsize=11, color=MUTED,
    )
    save_fig(fig, "ch22_timestamp_align.png")


def plot_complementary(t, omega, accel, attitude_true, alpha=0.98):
    att_gyro = integrate_gyro(t, omega, att0=attitude_true[0])
    att_comp = complementary_filter(t, omega, accel, alpha=alpha, att0=attitude_true[0])
    tilt = accel_to_tilt_deg(accel)

    fig, axes = plt.subplots(2, 1, figsize=(9.2, 5.8), layout="constrained", sharex=True)
    # pitch
    axes[0].plot(t, attitude_true[:, 1], color=MUTED, lw=1.6, ls="--", label="真 pitch")
    axes[0].plot(t, att_gyro[:, 1], color=WAVE, lw=1.2, label="纯陀螺积分")
    axes[0].plot(t, tilt[:, 1], color="#a08060", lw=0.9, alpha=0.7, label="加速度计倾角（噪声大）")
    axes[0].plot(t, att_comp[:, 1], color=TEAL, lw=1.5, label=f"互补滤波 α={alpha}")
    finish(axes[0])
    axes[0].set_ylabel("pitch (°)")
    axes[0].set_title("互补滤波直觉：陀螺管「短时间变化」，加速度计管「长时间别飘」")
    axes[0].legend(loc="upper right", fontsize=9, ncol=2, frameon=False)

    axes[1].plot(t, att_gyro[:, 1] - attitude_true[:, 1], color=WAVE, lw=1.2, label="纯陀螺误差")
    axes[1].plot(t, att_comp[:, 1] - attitude_true[:, 1], color=TEAL, lw=1.4, label="互补滤波误差")
    finish(axes[1])
    axes[1].set_xlabel("时间 (s)")
    axes[1].set_ylabel("pitch 误差 (°)")
    axes[1].set_title("纯积分误差会漂；融合后误差被拉回附近")
    axes[1].legend(loc="upper right", fontsize=10, frameon=False)
    save_fig(fig, "ch22_complementary.png")
    return att_gyro, att_comp


def plot_pipeline():
    fig, ax = plt.subplots(figsize=(9.0, 3.2), layout="constrained")
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 3)
    ax.axis("off")
    ax.set_title("本篇一条线：从 IMU 读数到「这一帧相机朝哪」")

    boxes = [
        (0.3, "ω 三轴\n角速度"),
        (3.0, "按时间\n积分"),
        (5.7, "姿态 θ\n(有漂移)"),
        (8.4, "对齐到\n曝光中心"),
    ]
    for x, text in boxes:
        ax.add_patch(
            FancyBboxPatch(
                (x, 0.9), 2.2, 1.3,
                boxstyle="round,pad=0.05,rounding_size=0.12",
                facecolor="#e5f3f1" if x > 5 else "#fde8df",
                edgecolor="#1c1915",
                lw=1.3,
            )
        )
        ax.text(x + 1.1, 1.55, text, ha="center", va="center", fontsize=12)
    for x in (2.5, 5.2, 7.9):
        ax.annotate("", xy=(x + 0.45, 1.55), xytext=(x, 1.55),
                    arrowprops=dict(arrowstyle="->", color=WAVE, lw=1.8))
    ax.text(
        6.0, 0.35,
        "下一篇会把姿态差变成像素 warp，做陀螺 EIS / RS 校正",
        ha="center", fontsize=11, color=MUTED,
    )
    save_fig(fig, "ch22_pipeline.png")


def plot_mine(bias_deg, alpha, exposure_ms):
    """跟随「改这里」：零偏、互补滤波 α、曝光中心示意。"""
    t, omega, accel, att_true, _ = simulate_imu(
        bias_deg=np.array([0.05, -0.03, float(bias_deg)]),
        seed=11,
    )
    att_gyro = integrate_gyro(t, omega, att0=att_true[0])
    att_comp = complementary_filter(t, omega, accel, alpha=alpha, att0=att_true[0])

    fig, axes = plt.subplots(2, 1, figsize=(9.0, 5.6), layout="constrained", sharex=True)
    axes[0].plot(t, att_true[:, 2], "--", color=MUTED, lw=1.4, label="真 yaw")
    axes[0].plot(t, att_gyro[:, 2], color=WAVE, lw=1.3, label=f"纯积分（yaw 零偏={bias_deg}°/s）")
    finish(axes[0])
    axes[0].set_ylabel("yaw (°)")
    axes[0].set_title("你的参数：零偏越大，积分姿态漂得越快")
    axes[0].legend(loc="best", fontsize=10, frameon=False)

    axes[1].plot(t, att_true[:, 1], "--", color=MUTED, lw=1.4, label="真 pitch")
    axes[1].plot(t, att_comp[:, 1], color=TEAL, lw=1.4, label=f"互补滤波 α={alpha}")
    finish(axes[1])
    axes[1].set_xlabel("时间 (s)")
    axes[1].set_ylabel("pitch (°)")
    axes[1].set_title("你的参数：α 越大越跟陀螺；越小越贴加速度计（但更抖）")
    axes[1].legend(loc="best", fontsize=10, frameon=False)
    save_fig(fig, "ch22_my_attitude.png")

    # 曝光中心数值示意
    fig, ax = plt.subplots(figsize=(7.5, 2.6), layout="constrained")
    ax.axis("off")
    t_center = exposure_ms / 2.0
    ax.set_title("你的曝光参数：曝光中心 = 打开时刻 + 曝光时长/2")
    ax.text(
        0.05, 0.55,
        f"EXPOSURE_MS = {exposure_ms}\n"
        f"若帧打开于 t0，则对齐时刻 t_align = t0 + {t_center:.2f} ms\n"
        f"再到 IMU 日志里找最接近 t_align 的样本（或线性插值）。",
        fontsize=13, transform=ax.transAxes, va="center",
    )
    save_fig(fig, "ch22_my_exposure.png")


def plot_log_preview(path):
    """把日志前几行画成「读日志」示意。"""
    t, omega, accel = load_imu_log(path)
    fig, ax = plt.subplots(figsize=(9.0, 3.4), layout="constrained")
    ax.plot(t[:80], omega[:80, 0], color=WAVE, label="gx→roll")
    ax.plot(t[:80], omega[:80, 1], color=TEAL, label="gy→pitch")
    ax.plot(t[:80], omega[:80, 2], color="#1c1915", label="gz→yaw")
    finish(ax)
    ax.set_xlabel("时间 (s)　从日志第一行起算")
    ax.set_ylabel("角速度 (°/s)")
    ax.set_title(f"读入 {path.name}：前 80 个样本")
    ax.legend(loc="upper right", fontsize=10, frameon=False)
    save_fig(fig, "ch22_log_preview.png")


# ---------------------------------------------------------------------------
# 自检 ≥ 5
# ---------------------------------------------------------------------------

def self_check():
    checks = []

    # 1) 常值角速度积分 → 线性姿态
    t = np.linspace(0, 2.0, 201)
    w = np.zeros((len(t), 3))
    w[:, 2] = 10.0  # 10 °/s
    att = integrate_gyro(t, w, att0=np.zeros(3))
    assert abs(att[-1, 2] - 20.0) < 1e-6
    checks.append("常值ω积分得线性姿态")

    # 2) 常值零偏 → 姿态误差近似 bias * T
    t2, omega, accel, att_true, bias = simulate_imu(
        duration=3.0,
        fs_imu=100.0,
        bias_deg=np.array([0.05, -0.03, 0.5]),
        noise_std_deg=0.05,
        seed=1,
    )
    att_d = integrate_gyro(t2, omega, att0=att_true[0])
    # yaw 误差末值应接近 bias_yaw * T（噪声会扰动）
    err = att_d[-1, 2] - att_true[-1, 2]
    expected = bias[2] * t2[-1]
    assert abs(err - expected) < 1.0
    checks.append("零偏导致近似线性漂移")

    # 3) 互补滤波 pitch 误差 RMS < 纯陀螺（给够 pitch 零偏与时长）
    t3, omega3, accel3, att3, _ = simulate_imu(
        duration=6.0,
        fs_imu=200.0,
        bias_deg=np.array([0.4, 0.5, 0.3]),
        noise_std_deg=0.05,
        seed=2,
    )
    att_g = integrate_gyro(t3, omega3, att0=att3[0])
    att_c = complementary_filter(t3, omega3, accel3, alpha=0.98, att0=att3[0])
    rms_g = np.sqrt(np.mean((att_g[:, 1] - att3[:, 1]) ** 2))
    rms_c = np.sqrt(np.mean((att_c[:, 1] - att3[:, 1]) ** 2))
    assert rms_c < rms_g
    checks.append("互补滤波 pitch 误差小于纯积分")

    # 4) 日志读写往返
    tmp = IMG / "ch22_imu_log.csv"
    write_imu_log(tmp, t2, omega, accel)
    t_l, w_l, a_l = load_imu_log(tmp)
    assert len(t_l) == len(t2)
    assert np.allclose(w_l, omega, atol=1e-4)
    assert np.allclose(a_l, accel, atol=1e-5)
    checks.append("IMU 日志写读一致")

    # 5) 曝光中心：打开 + 曝光/2
    t_open_ms = 1000.0
    exposure_ms = 10.0
    t_align = t_open_ms + 0.5 * exposure_ms
    assert abs(t_align - 1005.0) < 1e-9
    checks.append("曝光中心对齐公式")

    # 6) 采样率：200Hz 时 Δt≈5ms；30fps 帧间隔≈33.3ms，中间可插多笔 IMU
    dt_imu = 1.0 / 200.0
    dt_frame = 1.0 / 30.0
    assert dt_frame / dt_imu > 5
    checks.append("IMU 采样密于视频帧率")

    # 7) 加速度计估倾角：水平时应接近 0
    flat = np.array([[0.0, 0.0, 1.0]])
    tilt = accel_to_tilt_deg(flat)
    assert abs(tilt[0, 0]) < 1e-6 and abs(tilt[0, 1]) < 1e-6
    checks.append("水平姿态加速度计倾角≈0")

    print("自检通过：" + "；".join(checks) + f"。共 {len(checks)} 项。")
    return checks


def main():
    use_style()
    FIG.mkdir(parents=True, exist_ok=True)
    IMG.mkdir(parents=True, exist_ok=True)

    self_check()

    t, omega, accel, att_true, bias = simulate_imu(duration=4.0, fs_imu=200.0)
    log_path = IMG / "ch22_imu_log.csv"
    write_imu_log(log_path, t, omega, accel)

    # 附带一份「怎么读」的短说明
    save_text(
        "ch22_imu_log_readme.txt",
        "ch22_imu_log.csv 字段说明\n"
        "timestamp_us : 微秒时间戳（任意起点，用差值算 Δt）\n"
        "gx,gy,gz     : 角速度，单位 rad/s（脚本画图时会换成 °/s）\n"
        "ax,ay,az     : 加速度，单位 g（约等于重力投影；本日志为合成数据）\n"
        "\n"
        "合成场景：约 4 秒手持，含慢摇、手抖、t≈2.05s 甩头。\n"
        "内含小零偏与噪声，用于演示积分漂移与互补滤波。\n",
    )

    plot_pipeline()
    plot_three_axes()
    plot_omega_curves(t, omega, att_true)
    plot_integrate_drift(t, omega, att_true, bias_extra_yaw_deg=0.0)
    plot_sample_vs_frame(t, omega)
    plot_timestamp_align(exposure_ms=EXPOSURE_MS)
    plot_complementary(t, omega, accel, att_true, alpha=0.98)
    plot_log_preview(log_path)
    plot_mine(GYRO_BIAS_DEG, COMP_ALPHA, EXPOSURE_MS)

    print("课文对照图已写入 assets/figures/")
    print(f"合成 IMU 日志：{log_path}")
    print("你的图：ch22_my_attitude.png、ch22_my_exposure.png")


if __name__ == "__main__":
    main()
