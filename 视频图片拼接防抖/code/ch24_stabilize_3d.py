"""第 24 篇：3D 稳像与虚拟相机路径。

    MPLCONFIGDIR=/tmp/mpl-cache .venv/bin/python code/ch24_stabilize_3d.py

顶部「改这里」只影响 ch24_my_virtual_path.png（以及同参数下的数值示意）。
"""

from pathlib import Path

SHOW_WINDOW = False

import matplotlib

if not SHOW_WINDOW:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle

from plot_style import INK, MUTED, TEAL, WAVE, use_style

# ===== 改这里 =====
# 抖动幅度（相机位置的高频扰动，单位：任意长度）
SHAKE_AMP = 0.35
# 平滑窗口半宽（越大虚拟路径越「黏」，故意运镜也会被压扁）
SMOOTH_HALF = 4
# 近景点深度（越小视差越大；远景固定为 8.0）
NEAR_DEPTH = 1.2
# ==================

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "assets" / "figures"

N_FRAMES = 48
FAR_DEPTH = 8.0
FOCAL = 2.0  # 简化针孔焦距


def save_fig(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    path = FIG / name
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"已保存 {path}")


def intended_path(n=N_FRAMES):
    """故意向右平移的「运镜」意图（低频）。"""
    t = np.linspace(0.0, 1.0, n)
    x = 0.15 * t  # 缓慢右移
    y = 0.02 * np.sin(2 * np.pi * t)  # 轻微上下起伏（跟拍感）
    return np.stack([x, y], axis=1)


def shaky_path(intended, amp=SHAKE_AMP, seed=7):
    """在意图路径上叠加高频手抖。"""
    rng = np.random.default_rng(seed)
    n = len(intended)
    # 高频 + 一点中频，模拟手持
    t = np.arange(n, dtype=np.float64)[:, None]
    shake = amp * (
        0.55 * np.sin(0.9 * t + np.array([[0.0, 1.2]]))
        + 0.35 * np.sin(2.3 * t + np.array([[0.4, -0.7]]))
        + 0.25 * rng.normal(size=(n, 2))
    )
    return intended + shake


def smooth_path(path, half=SMOOTH_HALF):
    """简单滑动平均，得到「虚拟相机」目标路径。"""
    half = max(0, int(half))
    if half == 0:
        return path.copy()
    kernel = np.ones(2 * half + 1, dtype=np.float64)
    kernel /= kernel.sum()
    out = np.empty_like(path)
    for d in range(path.shape[1]):
        # 边缘用对称填充，避免端点塌缩
        padded = np.pad(path[:, d], half, mode="edge")
        out[:, d] = np.convolve(padded, kernel, mode="valid")
    return out


def project_point(cam_xy, world_xy, depth, f=FOCAL):
    """极简针孔：相机在 (cx,cy)，看向 +Z；世界点在同一平面深度 Z。

    像素坐标 u = f * (X - cx) / Z
    （只做一维横向示意 + 纵向同样公式，便于画图。）
    """
    rel = world_xy - cam_xy
    return f * rel / depth


def residual_2d_affine_fail(cam_path, near_xy, far_xy, near_z, far_z):
    """演示：若用「远景」估出的平移去补偿，近景仍晃。

    假设 2D 稳像只看到远景的像移，估计出相机抖动并反向补偿；
    近景因视差更大，补偿后仍有残差。
    """
    # 参考帧 0
    u_far0 = project_point(cam_path[0], far_xy, far_z)
    u_near0 = project_point(cam_path[0], near_xy, near_z)

    far_res = []
    near_res = []
    for cam in cam_path:
        u_far = project_point(cam, far_xy, far_z)
        u_near = project_point(cam, near_xy, near_z)
        # 2D：用远景位移当「整帧平移」去补偿
        delta = u_far - u_far0
        far_res.append(np.linalg.norm((u_far - delta) - u_far0))
        near_res.append(np.linalg.norm((u_near - delta) - u_near0))
    return np.asarray(far_res), np.asarray(near_res)


def reproject_to_virtual(cam, virt, u, depth):
    """已知深度：像素 u（在真实相机）→ 3D → 虚拟相机像素。"""
    world = cam + u * depth / FOCAL
    return project_point(virt, world, depth)


def residual_3d_virtual(cam_path, virt_path, near_xy, far_xy, near_z, far_z):
    """3D：重投影结果应等于「虚拟相机直接看静态场景」——误差≈0。

    注意：虚拟路径本身可以缓慢移动（保留运镜），所以不能要求贴死第 0 帧。
    """
    far_res = []
    near_res = []
    for cam, virt in zip(cam_path, virt_path):
        u_far = project_point(cam, far_xy, far_z)
        u_near = project_point(cam, near_xy, near_z)
        u_far_v = reproject_to_virtual(cam, virt, u_far, far_z)
        u_near_v = reproject_to_virtual(cam, virt, u_near, near_z)
        u_far_ideal = project_point(virt, far_xy, far_z)
        u_near_ideal = project_point(virt, near_xy, near_z)
        far_res.append(np.linalg.norm(u_far_v - u_far_ideal))
        near_res.append(np.linalg.norm(u_near_v - u_near_ideal))
    return np.asarray(far_res), np.asarray(near_res)


def image_jitter_energy(u_series):
    """二阶差分能量：越大说明帧间越「毛」。"""
    if len(u_series) < 3:
        return 0.0
    d2 = np.diff(u_series, n=2, axis=0)
    return float(np.mean(np.sum(d2**2, axis=1)))


def tracked_series_2d(cam_path, near_xy, far_xy, near_z, far_z):
    """2D 远景对齐后，近/远点在「补偿坐标系」里的轨迹。"""
    u_far0 = project_point(cam_path[0], far_xy, far_z)
    near_traj = []
    far_traj = []
    for cam in cam_path:
        u_far = project_point(cam, far_xy, far_z)
        u_near = project_point(cam, near_xy, near_z)
        delta = u_far - u_far0
        far_traj.append(u_far - delta)
        near_traj.append(u_near - delta)
    return np.asarray(near_traj), np.asarray(far_traj)


def tracked_series_3d(cam_path, virt_path, near_xy, far_xy, near_z, far_z):
    """3D 重投影到虚拟相机后的近/远轨迹。"""
    near_traj = []
    far_traj = []
    for cam, virt in zip(cam_path, virt_path):
        u_far = project_point(cam, far_xy, far_z)
        u_near = project_point(cam, near_xy, near_z)
        far_traj.append(reproject_to_virtual(cam, virt, u_far, far_z))
        near_traj.append(reproject_to_virtual(cam, virt, u_near, near_z))
    return np.asarray(near_traj), np.asarray(far_traj)


def plot_parallax_fail():
    """图：近远景视差导致 2D 整帧 warp 顾此失彼。"""
    intended = intended_path()
    cam = shaky_path(intended, amp=0.4, seed=3)
    near_xy = np.array([0.0, 0.0])
    far_xy = np.array([0.5, 0.1])
    far_r, near_r = residual_2d_affine_fail(cam, near_xy, far_xy, NEAR_DEPTH, FAR_DEPTH)

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.0), layout="constrained")

    ax = axes[0]
    ax.set_title("现象：同一抖动，近景像移更大")
    # 示意：相机左右抖，近/远点投影位移不同
    frames = [0, 8, 16]
    colors = [MUTED, TEAL, WAVE]
    for fi, c in zip(frames, colors):
        cam_i = cam[fi]
        for label, depth, y0, marker in (
            ("近景", NEAR_DEPTH, 0.15, "o"),
            ("远景", FAR_DEPTH, -0.15, "s"),
        ):
            world = near_xy if label == "近景" else far_xy
            u = project_point(cam_i, world, depth)
            ax.scatter([u[0]], [y0], color=c, marker=marker, s=70, zorder=3)
            if fi == frames[0]:
                ax.text(u[0] + 0.02, y0 + 0.05, label, fontsize=10, color=MUTED)
        ax.annotate(
            f"帧{fi}",
            xy=(project_point(cam_i, near_xy, NEAR_DEPTH)[0], 0.15),
            xytext=(0, 0.35 + 0.08 * frames.index(fi)),
            textcoords=("data", "data"),
            fontsize=9,
            color=c,
            arrowprops=dict(arrowstyle="-", color=c, lw=0.8),
        )
    ax.axhline(0, color="#e6dccb", lw=1)
    ax.set_xlabel("图像横向坐标（示意）")
    ax.set_yticks([])
    ax.set_ylim(-0.55, 0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    ax = axes[1]
    ax.set_title("用远景估平移去补偿后的残差")
    t = np.arange(len(far_r))
    ax.plot(t, far_r, color=TEAL, lw=2.0, label="远景残差（被对上）")
    ax.plot(t, near_r, color=WAVE, lw=2.0, label="近景残差（仍在晃）")
    ax.set_xlabel("帧")
    ax.set_ylabel("相对参考帧的像素残差（示意）")
    ax.legend(frameon=False, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, color="#e6dccb", lw=0.8)
    ax.set_axisbelow(True)

    save_fig(fig, "ch24_parallax_fail.png")


def plot_pipeline():
    """概念流水线。"""
    fig, ax = plt.subplots(figsize=(10.2, 3.8), layout="constrained")
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4)
    ax.axis("off")
    ax.set_title("3D 稳像流水线（概念）")

    boxes = [
        (0.3, 1.3, "1. 估计真实\n相机路径\n(+ 姿态)"),
        (3.1, 1.3, "2. 估计场景\n深度 / 结构\n(或平面近似)"),
        (5.9, 1.3, "3. 规划平滑\n虚拟相机\n路径"),
        (8.7, 1.3, "4. 按深度\n重投影 +\n裁切输出"),
    ]
    for x, y, text in boxes:
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                2.3,
                1.8,
                boxstyle="round,pad=0.08,rounding_size=0.15",
                facecolor="#fffdf8",
                edgecolor=INK,
                lw=1.4,
            )
        )
        ax.text(x + 1.15, y + 0.9, text, ha="center", va="center", fontsize=11)

    for x0 in (2.6, 5.4, 8.2):
        ax.annotate(
            "",
            xy=(x0 + 0.45, 2.2),
            xytext=(x0, 2.2),
            arrowprops=dict(arrowstyle="->", color=WAVE, lw=1.8),
        )

    ax.text(
        6.0,
        0.55,
        "关键：补偿发生在 3D（或分层深度）里，不再假设「整幅共用一个 2D 仿射」。",
        ha="center",
        fontsize=12,
        color=MUTED,
    )
    save_fig(fig, "ch24_pipeline.png")


def plot_virtual_path(shake_amp=SHAKE_AMP, smooth_half=SMOOTH_HALF, name="ch24_virtual_path.png", title_extra=""):
    intended = intended_path()
    cam = shaky_path(intended, amp=shake_amp)
    virt = smooth_path(cam, half=smooth_half)

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.1), layout="constrained")

    ax = axes[0]
    ax.set_title("俯视图：真实路径 vs 虚拟路径" + title_extra)
    ax.plot(cam[:, 0], cam[:, 1], color=WAVE, lw=1.4, alpha=0.85, label="真实相机（抖）")
    ax.plot(virt[:, 0], virt[:, 1], color=TEAL, lw=2.4, label="虚拟相机（平滑）")
    ax.plot(intended[:, 0], intended[:, 1], color=MUTED, lw=1.2, ls="--", label="故意运镜（低频）")
    ax.scatter([cam[0, 0]], [cam[0, 1]], color=WAVE, s=40, zorder=3)
    ax.scatter([virt[0, 0]], [virt[0, 1]], color=TEAL, s=40, zorder=3)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(frameon=False, fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, color="#e6dccb", lw=0.8)
    ax.set_axisbelow(True)

    ax = axes[1]
    ax.set_title("沿时间：X 坐标")
    t = np.arange(len(cam))
    ax.plot(t, cam[:, 0], color=WAVE, lw=1.3, label="真实 X")
    ax.plot(t, virt[:, 0], color=TEAL, lw=2.2, label="虚拟 X")
    ax.plot(t, intended[:, 0], color=MUTED, lw=1.1, ls="--", label="意图 X")
    ax.set_xlabel("帧")
    ax.set_ylabel("X")
    ax.legend(frameon=False, fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, color="#e6dccb", lw=0.8)
    ax.set_axisbelow(True)

    save_fig(fig, name)


def plot_reproject_idea():
    """重投影直觉：同一 3D 点，从真实相机改投到虚拟相机。"""
    fig, ax = plt.subplots(figsize=(9.6, 4.2), layout="constrained")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis("off")
    ax.set_title("重投影直觉：先回到 3D，再投到虚拟相机")

    # 地面深度线
    ax.plot([1.2, 8.8], [1.2, 1.2], color="#e6dccb", lw=2)
    ax.text(9.0, 1.15, "场景深度面", fontsize=10, color=MUTED, va="center")

    # 3D 点
    ax.add_patch(Circle((5.5, 1.2), 0.12, color=INK, zorder=4))
    ax.text(5.5, 0.75, "3D 点 P", ha="center", fontsize=11)

    # 真实相机
    ax.add_patch(Rectangle((2.2, 3.4), 0.9, 0.55, facecolor="#fde8df", edgecolor=WAVE, lw=1.5))
    ax.text(2.65, 4.2, "真实相机 C", ha="center", color=WAVE, fontsize=11)
    ax.plot([2.65, 5.5], [3.4, 1.32], color=WAVE, lw=1.5)
    ax.plot([2.65, 2.65], [3.4, 2.6], color=WAVE, lw=1.0, ls=":")
    ax.text(1.5, 2.85, "像素 u", color=WAVE, fontsize=10)

    # 虚拟相机
    ax.add_patch(Rectangle((6.9, 3.55), 0.9, 0.55, facecolor="#e5f3f1", edgecolor=TEAL, lw=1.5))
    ax.text(7.35, 4.35, "虚拟相机 V", ha="center", color=TEAL, fontsize=11)
    ax.plot([7.35, 5.5], [3.55, 1.32], color=TEAL, lw=1.8)
    ax.plot([7.35, 7.35], [3.55, 2.75], color=TEAL, lw=1.0, ls=":")
    ax.text(7.7, 2.95, "像素 u′", color=TEAL, fontsize=10)

    ax.annotate(
        "",
        xy=(6.9, 3.8),
        xytext=(3.2, 3.7),
        arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.4, connectionstyle="arc3,rad=-0.2"),
    )
    ax.text(5.0, 4.55, "路径规划：C 序列 → 更平滑的 V 序列", ha="center", color=MUTED, fontsize=11)

    ax.text(
        5.0,
        0.25,
        "有深度才能从 u 反投到 P；深度错了，近景/遮挡边界就会糊或「粘纸」变形。",
        ha="center",
        fontsize=11,
        color=MUTED,
    )
    save_fig(fig, "ch24_reproject.png")


def plot_capability():
    """能力边界对照。"""
    fig, ax = plt.subplots(figsize=(10.2, 4.4), layout="constrained")
    ax.axis("off")
    ax.set_title("2D 视觉 EIS / 陀螺 EIS / 3D 稳像：能力边界（示意）")

    col_labels = ["方法", "主要输入", "擅长", "薄弱", "算力/成本"]
    rows = [
        [
            "2D 视觉 EIS",
            "特征/光流\n估仿射·相似",
            "平面/远景\n小视差",
            "近景大视差\n低纹理、模糊",
            "低–中",
        ],
        [
            "陀螺 EIS",
            "IMU 姿态\n(+ 外参)",
            "旋转抖动\nRS 行旋转\n暗光甩动",
            "纯平移视差\n外参/延迟误差",
            "低（实时友好）",
        ],
        [
            "3D 稳像",
            "位姿 + 深度\n/ 分层 / SfM",
            "大视差近景\n可规划虚拟运镜",
            "深度不准\n动态物体\n算力与延迟",
            "中–高",
        ],
    ]
    table = ax.table(cellText=rows, colLabels=col_labels, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.12, 2.15)
    for (r, c), cell in table.get_celld().items():
        if r == 0:
            cell.set_facecolor("#f3e6d8")
            cell.set_text_props(weight="bold")
        elif r == 1:
            cell.set_facecolor("#fffdf8")
        elif r == 2:
            cell.set_facecolor("#e5f3f1")
        else:
            cell.set_facecolor("#fde8df")
    save_fig(fig, "ch24_capability.png")


def plot_numeric_demo(shake_amp=SHAKE_AMP, smooth_half=SMOOTH_HALF, near_z=NEAR_DEPTH, name="ch24_numeric.png"):
    """简化数值：左看 2D 残差，右看补偿后轨迹的「毛刺能量」。"""
    intended = intended_path()
    cam = shaky_path(intended, amp=shake_amp, seed=11)
    virt = smooth_path(cam, half=smooth_half)
    near_xy = np.array([0.0, 0.0])
    far_xy = np.array([0.4, 0.05])

    far2, near2 = residual_2d_affine_fail(cam, near_xy, far_xy, near_z, FAR_DEPTH)
    near_2d, far_2d = tracked_series_2d(cam, near_xy, far_xy, near_z, FAR_DEPTH)
    near_3d, far_3d = tracked_series_3d(cam, virt, near_xy, far_xy, near_z, FAR_DEPTH)
    # 重投影几何误差（应≈0）
    err_far, err_near = residual_3d_virtual(cam, virt, near_xy, far_xy, near_z, FAR_DEPTH)

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.0), layout="constrained")
    t = np.arange(len(cam))

    ax = axes[0]
    ax.set_title("2D：用远景平移去「冻帧」后的残差")
    ax.plot(t, far2, color=TEAL, lw=2.0, label="远景")
    ax.plot(t, near2, color=WAVE, lw=2.0, label="近景")
    ax.set_xlabel("帧")
    ax.set_ylabel("相对参考帧残差")
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, color="#e6dccb", lw=0.8)
    ax.set_axisbelow(True)

    ax = axes[1]
    ax.set_title("轨迹毛刺能量（越小越稳）")
    labels = ["2D·近", "2D·远", "3D·近", "3D·远"]
    vals = [
        image_jitter_energy(near_2d),
        image_jitter_energy(far_2d),
        image_jitter_energy(near_3d),
        image_jitter_energy(far_3d),
    ]
    colors = [WAVE, TEAL, WAVE, TEAL]
    bars = ax.bar(labels, vals, color=colors, edgecolor=INK, lw=0.8, alpha=0.9)
    # 3D 柱加纹样区分
    bars[2].set_hatch("//")
    bars[3].set_hatch("//")
    ax.set_ylabel("二阶差分能量")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="y", color="#e6dccb", lw=0.8)
    ax.set_axisbelow(True)
    ax.text(
        0.5,
        -0.18,
        f"3D 重投影几何误差 max≈{max(err_far.max(), err_near.max()):.1e}（深度已知时应≈0）",
        transform=ax.transAxes,
        ha="center",
        fontsize=9,
        color=MUTED,
    )

    fig.suptitle(
        f"数值示意 · 近景深度={near_z:.2f} · 平滑半宽={smooth_half} · 抖动={shake_amp:.2f}",
        fontsize=12,
        color=MUTED,
    )
    save_fig(fig, name)


def plot_crop_tradeoff():
    """虚拟路径越平滑，往往需要更大裁切余量。"""
    intended = intended_path()
    cam = shaky_path(intended, amp=0.45, seed=5)
    halves = [1, 3, 6, 10]
    fig, ax = plt.subplots(figsize=(9.2, 4.0), layout="constrained")
    ax.set_title("平滑越狠，真实相机相对虚拟相机的偏离越大 → 裁切更大")

    for i, h in enumerate(halves):
        virt = smooth_path(cam, half=h)
        # 相对偏离：用于估计「还能看到多少有效画面」的余量
        offset = np.linalg.norm(cam - virt, axis=1)
        ax.plot(
            np.arange(len(offset)),
            offset,
            color=[WAVE, "#b87a3c", TEAL, MUTED][i],
            lw=1.8,
            label=f"半宽={h}（均值 {offset.mean():.3f}）",
        )
    ax.set_xlabel("帧")
    ax.set_ylabel("|C − V|（路径空间）")
    ax.legend(frameon=False, fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, color="#e6dccb", lw=0.8)
    ax.set_axisbelow(True)
    save_fig(fig, "ch24_crop_tradeoff.png")


def plot_mine():
    plot_virtual_path(
        shake_amp=SHAKE_AMP,
        smooth_half=SMOOTH_HALF,
        name="ch24_my_virtual_path.png",
        title_extra="（你的参数）",
    )
    plot_numeric_demo(
        shake_amp=SHAKE_AMP,
        smooth_half=SMOOTH_HALF,
        near_z=NEAR_DEPTH,
        name="ch24_my_numeric.png",
    )


def self_check():
    # 1) 平滑后高频能量应下降
    intended = intended_path(32)
    cam = shaky_path(intended, amp=0.5, seed=0)
    virt = smooth_path(cam, half=5)
    high_cam = np.diff(cam[:, 0], n=2)
    high_virt = np.diff(virt[:, 0], n=2)
    assert np.mean(high_virt**2) < np.mean(high_cam**2)

    # 2) 远景 2D 补偿残差应远小于近景
    near_xy = np.array([0.0, 0.0])
    far_xy = np.array([0.3, 0.0])
    far_r, near_r = residual_2d_affine_fail(cam, near_xy, far_xy, 1.0, 8.0)
    assert near_r.mean() > far_r.mean() * 2.0

    # 3) 深度已知时，3D 重投影应等于虚拟相机直视静态场景
    far3, near3 = residual_3d_virtual(cam, virt, near_xy, far_xy, 1.0, 8.0)
    assert far3.max() < 1e-9 and near3.max() < 1e-9

    # 4) 近景更近 → 同样相机抖动，像移更大（未补偿）
    u_near = [
        np.linalg.norm(project_point(c, near_xy, 1.0) - project_point(cam[0], near_xy, 1.0)) for c in cam
    ]
    u_far = [
        np.linalg.norm(project_point(c, far_xy, 8.0) - project_point(cam[0], far_xy, 8.0)) for c in cam
    ]
    assert np.mean(u_near) > np.mean(u_far)

    # 5) 半宽为 0 时虚拟路径 = 真实路径；且 3D 后近景毛刺应低于 2D 补偿
    assert np.allclose(smooth_path(cam, half=0), cam)
    near_2d, far_2d = tracked_series_2d(cam, near_xy, far_xy, 1.0, 8.0)
    near_3d, far_3d = tracked_series_3d(cam, virt, near_xy, far_xy, 1.0, 8.0)
    assert image_jitter_energy(near_3d) < image_jitter_energy(near_2d)
    assert image_jitter_energy(far_2d) < image_jitter_energy(near_2d)

    print("自检通过：平滑降抖、2D 近景失败、3D 重投影、视差与毛刺能量。")


def main():
    use_style()
    self_check()
    plot_parallax_fail()
    plot_pipeline()
    plot_virtual_path()
    plot_reproject_idea()
    plot_capability()
    plot_numeric_demo()
    plot_crop_tradeoff()
    plot_mine()
    print("课文对照图已写入 assets/figures/")
    print("你的图：ch24_my_virtual_path.png、ch24_my_numeric.png")


if __name__ == "__main__":
    main()
