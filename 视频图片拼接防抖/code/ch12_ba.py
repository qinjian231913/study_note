"""第 12 篇：Bundle Adjustment——多图一起纠偏。

    MPLCONFIGDIR=/tmp/mpl-cache python code/ch12_ba.py

顶部「改这里」只影响 ch12_my_ba.png 与控制台打印的自跑结果。
不依赖 scipy：用纯 numpy 做小规模高斯–牛顿示意。
"""

from pathlib import Path

SHOW_WINDOW = False

import matplotlib

if not SHOW_WINDOW:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyBboxPatch

from plot_style import INK, MUTED, TEAL, WAVE, use_style

# ===== 改这里 =====
# 合成观测噪声（度）。越大，成对角度越「飘」，越需要 BA。
NOISE_DEG = 1.8
# 高斯–牛顿迭代步数
N_ITERS = 12
# 是否加入闭环观测（第 0 张与最后一张之间的相对角）。360° 全景常有。
USE_LOOP = True
# 初始猜测扰动：把真值角度再加一点偏置（度），模拟「链式累加起点」
INIT_BIAS_DEG = 0.0
# ==================

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "assets" / "figures"
IMG = ROOT / "assets" / "images"

RNG = np.random.default_rng(12)


def save_fig(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    path = FIG / name
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"已保存 {path}")


# ---------------------------------------------------------------------------
# 玩具模型 A：一维「朝向角」位姿图（全景里最常见的简化）
# ---------------------------------------------------------------------------

def true_yaw_deg(n=6):
    """等间隔朝向（度）。相机 0 固定为 0°。"""
    return np.linspace(0.0, 75.0, n)


def synth_relative_obs(yaw_deg, noise_deg, use_loop, rng):
    """相邻（及可选闭环）相对角观测：δ_ij ≈ θ_j − θ_i + 噪声。"""
    n = len(yaw_deg)
    edges = [(i, i + 1) for i in range(n - 1)]
    if use_loop and n >= 3:
        edges.append((n - 1, 0))
    obs = []
    for i, j in edges:
        true_d = yaw_deg[j] - yaw_deg[i]
        if use_loop and (i, j) == (n - 1, 0):
            # 闭环：从最后回到第一，相对角应为 −总跨度（或 360−跨度 的补角语义）
            true_d = yaw_deg[0] - yaw_deg[n - 1]
        noisy = true_d + rng.normal(0.0, noise_deg)
        obs.append((i, j, noisy))
    return obs


def chain_compose(obs, n):
    """只信相邻边、从 0 往前累加——误差会越积越大。"""
    theta = np.zeros(n)
    for i in range(n - 1):
        # 找边 (i, i+1)
        d = next(d for a, b, d in obs if a == i and b == i + 1)
        theta[i + 1] = theta[i] + d
    return theta


def ba_cost(theta, obs):
    return 0.5 * sum((theta[j] - theta[i] - d) ** 2 for i, j, d in obs)


def ba_gauss_newton(obs, n, n_iters=12, init=None):
    """一维拉格朗日：固定 θ0=0，对其余角做高斯–牛顿。"""
    theta = np.zeros(n) if init is None else np.asarray(init, dtype=np.float64).copy()
    theta[0] = 0.0
    history = [ba_cost(theta, obs)]
    traj = [theta.copy()]
    free = list(range(1, n))  # 可优化下标
    m = len(free)
    for _ in range(n_iters):
        # 残差 r_k = θ_j − θ_i − δ
        r = []
        J = []
        for i, j, d in obs:
            r.append(theta[j] - theta[i] - d)
            row = np.zeros(m)
            if i in free:
                row[free.index(i)] = -1.0
            if j in free:
                row[free.index(j)] = 1.0
            J.append(row)
        r = np.asarray(r)
        J = np.asarray(J)
        # 法方程 JᵀJ Δ = −Jᵀr
        jtj = J.T @ J
        jtr = J.T @ r
        # 轻微阻尼，防奇异
        jtj = jtj + 1e-8 * np.eye(m)
        try:
            delta = -np.linalg.solve(jtj, jtr)
        except np.linalg.LinAlgError:
            delta = -np.linalg.lstsq(jtj, jtr, rcond=None)[0]
        for k, idx in enumerate(free):
            theta[idx] += delta[k]
        history.append(ba_cost(theta, obs))
        traj.append(theta.copy())
    return theta, history, traj


# ---------------------------------------------------------------------------
# 玩具模型 B：二维「位姿 + 点」迷你 BA（展示完整名词）
# ---------------------------------------------------------------------------

def synth_2d_scene(rng):
    """3 个相机（仅平移）、4 个平面点。相机 0 固定在原点。"""
    cams_true = np.array(
        [
            [0.0, 0.0],
            [2.5, 0.3],
            [5.0, -0.2],
        ]
    )
    pts_true = np.array(
        [
            [1.2, 1.5],
            [2.0, -1.0],
            [3.5, 1.2],
            [4.2, -0.8],
        ]
    )
    # 观测：每个相机看见所有点（简单：观测 = 点 − 相机中心 + 噪声）
    noise = 0.08
    obs = {}
    for ci, c in enumerate(cams_true):
        for pj, p in enumerate(pts_true):
            z = p - c + rng.normal(0.0, noise, size=2)
            obs[(ci, pj)] = z
    return cams_true, pts_true, obs


def pack_2d(cams, pts):
    # cam0 固定；优化 cam1,cam2 与全部点
    return np.concatenate([cams[1:].ravel(), pts.ravel()])


def unpack_2d(x, n_cam=3, n_pt=4):
    cams = np.zeros((n_cam, 2))
    cams[1:] = x[: (n_cam - 1) * 2].reshape(n_cam - 1, 2)
    pts = x[(n_cam - 1) * 2 :].reshape(n_pt, 2)
    return cams, pts


def cost_2d(x, obs):
    cams, pts = unpack_2d(x)
    s = 0.0
    for (ci, pj), z in obs.items():
        pred = pts[pj] - cams[ci]
        e = pred - z
        s += 0.5 * float(e @ e)
    return s


def ba_2d_gn(obs, n_iters=20, init_cams=None, init_pts=None, rng=None):
    rng = rng or RNG
    if init_cams is None:
        init_cams = np.array([[0.0, 0.0], [2.2, 0.8], [5.3, 0.5]])
    if init_pts is None:
        init_pts = np.array(
            [
                [1.0, 1.2],
                [2.3, -0.6],
                [3.2, 0.9],
                [4.5, -0.5],
            ]
        )
    x = pack_2d(init_cams, init_pts)
    hist = [cost_2d(x, obs)]
    for _ in range(n_iters):
        cams, pts = unpack_2d(x)
        # 解析雅可比：残差对 (cam_k, pt_m)
        residuals = []
        rows = []
        for (ci, pj), z in obs.items():
            pred = pts[pj] - cams[ci]
            e = pred - z
            residuals.extend([e[0], e[1]])
            # 参数布局：[c1x,c1y,c2x,c2y, p0x,p0y,...]
            J2 = np.zeros((2, x.size))
            if ci >= 1:
                base = (ci - 1) * 2
                J2[0, base] = -1.0
                J2[1, base + 1] = -1.0
            pbase = 4 + pj * 2  # (3-1)*2 = 4
            J2[0, pbase] = 1.0
            J2[1, pbase + 1] = 1.0
            rows.append(J2)
        r = np.asarray(residuals)
        J = np.vstack(rows)
        jtj = J.T @ J + 1e-6 * np.eye(x.size)
        delta = -np.linalg.solve(jtj, J.T @ r)
        x = x + delta
        hist.append(cost_2d(x, obs))
    return unpack_2d(x), hist, unpack_2d(pack_2d(init_cams, init_pts))


# ---------------------------------------------------------------------------
# 作图
# ---------------------------------------------------------------------------

def plot_pair_chain_strip():
    """一对一对累加：多格动画条。"""
    yaw = true_yaw_deg(5)
    obs = synth_relative_obs(yaw, noise_deg=2.5, use_loop=False, rng=np.random.default_rng(7))
    chain = chain_compose(obs, len(yaw))

    fig, axes = plt.subplots(1, 4, figsize=(11.2, 3.4), layout="constrained")
    stages = [1, 2, 3, 4]
    for ax, k in zip(axes, stages):
        ax.set_aspect("equal")
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.axis("off")
        ax.set_title(f"累加到图 {k}")
        circ = Circle((0, 0), 1.0, fill=False, ec=MUTED, lw=1.0, ls="--")
        ax.add_patch(circ)
        # 真值：青色短线
        for i in range(k + 1):
            a = np.deg2rad(yaw[i])
            ax.plot([0, np.cos(a)], [0, np.sin(a)], color=TEAL, lw=2.0, alpha=0.85)
            ax.scatter([np.cos(a)], [np.sin(a)], color=TEAL, s=28, zorder=3)
            ax.text(1.08 * np.cos(a), 1.08 * np.sin(a), f"真{i}", fontsize=9, color=TEAL, ha="center")
        # 链式估计：橙色
        for i in range(k + 1):
            a = np.deg2rad(chain[i])
            ax.plot([0, 0.85 * np.cos(a)], [0, 0.85 * np.sin(a)], color=WAVE, lw=2.2, alpha=0.9)
            ax.scatter([0.85 * np.cos(a)], [0.85 * np.sin(a)], color=WAVE, s=36, zorder=4)
        # 终点误差弧
        err = abs(chain[k] - yaw[k])
        ax.text(0, -1.15, f"末端偏 {err:.1f}°", ha="center", fontsize=10, color=WAVE)
    fig.suptitle("一对一对估相对角，再链式拼起来 → 末端越偏越大", fontsize=13)
    save_fig(fig, "ch12_pair_chain.png")


def plot_accumulate_curve():
    yaw = true_yaw_deg(8)
    rng = np.random.default_rng(3)
    trials = []
    for t in range(40):
        obs = synth_relative_obs(yaw, noise_deg=1.5, use_loop=False, rng=rng)
        chain = chain_compose(obs, len(yaw))
        trials.append(np.abs(chain - yaw))
    trials = np.asarray(trials)
    mean_err = trials.mean(axis=0)
    p90 = np.percentile(trials, 90, axis=0)

    fig, ax = plt.subplots(figsize=(8.2, 4.0), layout="constrained")
    xs = np.arange(len(yaw))
    ax.fill_between(xs, 0, p90, color=WAVE, alpha=0.15, label="90% 分位（多次噪声）")
    ax.plot(xs, mean_err, color=WAVE, lw=2.4, marker="o", label="平均 |链式角 − 真值|")
    ax.set_xlabel("图像序号（从第 0 张往前累加）")
    ax.set_ylabel("朝向误差（度）")
    ax.set_title("链式误差会随「站数」往上爬")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.35)
    save_fig(fig, "ch12_accumulate_curve.png")


def plot_reprojection_idea():
    fig, ax = plt.subplots(figsize=(8.8, 4.2), layout="constrained")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis("off")
    ax.set_title("重投影误差：预测落点 vs 图上真实看到的点")

    # 相机
    ax.add_patch(FancyBboxPatch((0.6, 1.8), 1.6, 1.4, boxstyle="round,pad=0.05", fc="#fde8df", ec=WAVE, lw=1.5))
    ax.text(1.4, 2.5, "相机 i", ha="center", va="center", fontsize=12, color=WAVE)
    # 三维点
    ax.scatter([5.2], [3.6], s=90, color=TEAL, zorder=3)
    ax.text(5.2, 4.15, "三维点 X", ha="center", color=TEAL, fontsize=12)
    # 图像平面
    ax.plot([7.2, 7.2], [0.8, 4.4], color=INK, lw=2)
    ax.text(7.45, 4.55, "图像平面", fontsize=11)
    # 预测投影
    ax.annotate(
        "",
        xy=(7.2, 3.2),
        xytext=(5.2, 3.6),
        arrowprops=dict(arrowstyle="->", color=TEAL, lw=1.6),
    )
    ax.scatter([7.2], [3.2], s=70, color=TEAL, zorder=3)
    ax.text(7.55, 3.2, "预测 û", va="center", color=TEAL, fontsize=12)
    # 观测
    ax.scatter([7.2], [2.2], s=70, color=WAVE, zorder=3)
    ax.text(7.55, 2.2, "观测 u", va="center", color=WAVE, fontsize=12)
    ax.annotate(
        "",
        xy=(7.2, 2.2),
        xytext=(7.2, 3.2),
        arrowprops=dict(arrowstyle="<->", color=MUTED, lw=1.8),
    )
    ax.text(6.35, 2.7, "残差\n‖û−u‖", ha="center", fontsize=11, color=MUTED)

    ax.text(
        0.5,
        0.45,
        "BA 要做的事：同时轻轻拧「各相机位姿」和「各三维点」，让所有残差的平方和变小。",
        fontsize=12,
        color=INK,
    )
    save_fig(fig, "ch12_reproj.png")


def plot_pipeline():
    fig, ax = plt.subplots(figsize=(9.6, 3.0), layout="constrained")
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 3)
    ax.axis("off")
    ax.set_title("BA 在全景流水线里站哪一站")

    boxes = [
        (0.3, "特征\n匹配"),
        (2.5, "成对估 H\n/ 相对位姿"),
        (5.0, "Bundle\nAdjustment"),
        (7.5, "统一\n投影"),
        (9.8, "融合\n接缝"),
    ]
    for x, label in boxes:
        highlight = "Bundle" in label
        fc = "#fde8df" if highlight else "#e5f3f1"
        ec = WAVE if highlight else TEAL
        ax.add_patch(
            FancyBboxPatch((x, 0.9), 1.9, 1.3, boxstyle="round,pad=0.08", fc=fc, ec=ec, lw=1.8)
        )
        ax.text(x + 0.95, 1.55, label, ha="center", va="center", fontsize=11, color=ec if highlight else INK)
    for x in (2.2, 4.7, 7.2, 9.5):
        ax.annotate("", xy=(x + 0.25, 1.55), xytext=(x - 0.15, 1.55), arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.4))
    ax.text(5.95, 0.35, "可选但很关键：多图全局纠偏，专治「局部还行、整体拧巴」", ha="center", fontsize=11, color=MUTED)
    save_fig(fig, "ch12_pipeline.png")


def plot_ba_before_after():
    yaw = true_yaw_deg(6)
    obs = synth_relative_obs(yaw, noise_deg=2.2, use_loop=True, rng=np.random.default_rng(11))
    chain = chain_compose(obs, len(yaw))
    # 用链式结果当初值，再 BA
    theta_ba, hist, _ = ba_gauss_newton(obs, len(yaw), n_iters=15, init=chain)

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.2), layout="constrained")
    for ax, angles, title, color in (
        (axes[0], chain, "只链式累加（BA 前）", WAVE),
        (axes[1], theta_ba, "全局 BA 后", TEAL),
    ):
        ax.set_aspect("equal")
        ax.set_xlim(-1.25, 1.25)
        ax.set_ylim(-1.25, 1.25)
        ax.axis("off")
        ax.add_patch(Circle((0, 0), 1.0, fill=False, ec=MUTED, lw=1.0, ls="--"))
        for i, a_true in enumerate(yaw):
            a = np.deg2rad(a_true)
            ax.plot([0, np.cos(a)], [0, np.sin(a)], color=MUTED, lw=1.0, alpha=0.5)
        for i, a_est in enumerate(angles):
            a = np.deg2rad(a_est)
            ax.plot([0, 0.9 * np.cos(a)], [0, 0.9 * np.sin(a)], color=color, lw=2.2)
            ax.scatter([0.9 * np.cos(a)], [0.9 * np.sin(a)], color=color, s=40, zorder=3)
            ax.text(1.08 * np.cos(a), 1.08 * np.sin(a), str(i), fontsize=10, color=color, ha="center")
        rmse = np.sqrt(np.mean((angles - yaw) ** 2))
        ax.set_title(f"{title}\n朝向 RMSE = {rmse:.2f}°")
    save_fig(fig, "ch12_ba_before_after.png")

    fig, ax = plt.subplots(figsize=(7.6, 3.6), layout="constrained")
    ax.plot(hist, color=TEAL, lw=2.2, marker="o", markersize=4)
    ax.set_xlabel("高斯–牛顿迭代")
    ax.set_ylabel("代价 ½∑残差²")
    ax.set_title("BA 迭代：整体重投影（此处是相对角）代价往下走")
    ax.grid(True, alpha=0.35)
    save_fig(fig, "ch12_cost_curve.png")


def plot_2d_ba_scene():
    cams_t, pts_t, obs = synth_2d_scene(np.random.default_rng(5))
    (cams_ba, pts_ba), hist, (cams0, pts0) = ba_2d_gn(obs, n_iters=25, rng=np.random.default_rng(5))

    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.2), layout="constrained")
    for ax, cams, pts, title in (
        (axes[0], cams0, pts0, "初值（拧巴）"),
        (axes[1], cams_ba, pts_ba, "迷你 BA 后"),
    ):
        ax.scatter(pts_t[:, 0], pts_t[:, 1], s=80, facecolors="none", edgecolors=MUTED, lw=1.4, label="真值点")
        ax.scatter(pts[:, 0], pts[:, 1], s=55, color=TEAL, label="估计点", zorder=3)
        ax.scatter(cams_t[:, 0], cams_t[:, 1], marker="^", s=120, facecolors="none", edgecolors=MUTED, lw=1.4, label="真值相机")
        ax.scatter(cams[:, 0], cams[:, 1], marker="^", s=100, color=WAVE, label="估计相机", zorder=3)
        for i, c in enumerate(cams):
            ax.text(c[0], c[1] - 0.35, f"C{i}", ha="center", color=WAVE, fontsize=10)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        ax.set_title(title)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
    axes[0].legend(frameon=False, fontsize=9, loc="upper right")
    save_fig(fig, "ch12_points_ba.png")

    fig, ax = plt.subplots(figsize=(7.2, 3.4), layout="constrained")
    ax.plot(hist, color=WAVE, lw=2.2)
    ax.set_title(f"二维迷你 BA 代价（初值 {hist[0]:.2f} → 末值 {hist[-1]:.3f}）")
    ax.set_xlabel("迭代")
    ax.set_ylabel("½∑‖预测−观测‖²")
    ax.grid(True, alpha=0.35)
    save_fig(fig, "ch12_points_cost.png")


def plot_loop_idea():
    fig, ax = plt.subplots(figsize=(8.4, 3.8), layout="constrained")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.axis("off")
    ax.set_title("为什么闭环能「掰直」整条链")
    xs = np.linspace(1, 8.5, 6)
    ys = 2.2 + 0.15 * np.sin(np.linspace(0, 3, 6))
    ax.plot(xs, ys, color=WAVE, lw=2.2, marker="o", label="链式漂移路径")
    ax.plot([xs[0], xs[-1]], [2.2, 2.2], color=TEAL, lw=1.6, ls="--")
    ax.annotate(
        "闭环约束：\n首尾也该对上",
        xy=(xs[-1], ys[-1]),
        xytext=(7.2, 3.3),
        fontsize=11,
        color=TEAL,
        arrowprops=dict(arrowstyle="->", color=TEAL),
    )
    ax.text(1.0, 0.6, "只有相邻边：误差单向堆积。加上首尾观测：BA 会把中间各站一起往后拽正。", fontsize=12, color=MUTED)
    save_fig(fig, "ch12_loop.png")


def plot_mine():
    """用户可调参数跑一遍，写出对照图。"""
    yaw = true_yaw_deg(6)
    obs = synth_relative_obs(yaw, noise_deg=NOISE_DEG, use_loop=USE_LOOP, rng=np.random.default_rng(21))
    chain = chain_compose(obs, len(yaw))
    init = chain + INIT_BIAS_DEG
    init[0] = 0.0
    theta_ba, hist, _ = ba_gauss_newton(obs, len(yaw), n_iters=N_ITERS, init=init)

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0), layout="constrained")
    for ax, ang, title, col in (
        (axes[0], chain, f"链式（噪声 {NOISE_DEG}°）", WAVE),
        (axes[1], theta_ba, f"BA {N_ITERS} 步 / 闭环={USE_LOOP}", TEAL),
    ):
        ax.set_aspect("equal")
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.axis("off")
        ax.add_patch(Circle((0, 0), 1.0, fill=False, ec=MUTED, lw=1.0, ls="--"))
        for a in yaw:
            rad = np.deg2rad(a)
            ax.plot([0, np.cos(rad)], [0, np.sin(rad)], color=MUTED, alpha=0.4, lw=1)
        for a in ang:
            rad = np.deg2rad(a)
            ax.plot([0, 0.9 * np.cos(rad)], [0, 0.9 * np.sin(rad)], color=col, lw=2.2)
            ax.scatter([0.9 * np.cos(rad)], [0.9 * np.sin(rad)], color=col, s=36, zorder=3)
        rmse = np.sqrt(np.mean((ang - yaw) ** 2))
        ax.set_title(f"{title}\nRMSE={rmse:.2f}°  代价末={hist[-1]:.3f}")
    fig.suptitle("你的参数：ch12_my_ba.png", fontsize=13)
    save_fig(fig, "ch12_my_ba.png")
    print(
        f"改这里结果：噪声={NOISE_DEG}° 迭代={N_ITERS} 闭环={USE_LOOP} "
        f"链式RMSE={np.sqrt(np.mean((chain - yaw) ** 2)):.3f}° "
        f"BA_RMSE={np.sqrt(np.mean((theta_ba - yaw) ** 2)):.3f}°"
    )


def self_check():
    yaw = true_yaw_deg(5)
    obs = synth_relative_obs(yaw, noise_deg=0.0, use_loop=True, rng=np.random.default_rng(0))
    # 无噪声时链式应几乎等于真值
    chain = chain_compose(obs, len(yaw))
    assert np.allclose(chain, yaw, atol=1e-9)

    # 有噪声时 BA 代价应下降，且优于纯链式
    obs_n = synth_relative_obs(yaw, noise_deg=2.0, use_loop=True, rng=np.random.default_rng(1))
    chain_n = chain_compose(obs_n, len(yaw))
    theta, hist, _ = ba_gauss_newton(obs_n, len(yaw), n_iters=20, init=chain_n)
    assert hist[-1] <= hist[0] + 1e-9
    rmse_chain = np.sqrt(np.mean((chain_n - yaw) ** 2))
    rmse_ba = np.sqrt(np.mean((theta - yaw) ** 2))
    assert rmse_ba < rmse_chain

    # 二维迷你 BA：代价下降
    _, _, obs2 = synth_2d_scene(np.random.default_rng(2))
    _, hist2, _ = ba_2d_gn(obs2, n_iters=15)
    assert hist2[-1] < hist2[0]

    # 残差定义：预测 − 观测
    pred = np.array([3.0, 4.0])
    meas = np.array([2.5, 4.2])
    assert np.allclose(pred - meas, [0.5, -0.2])

    print("自检通过：无噪链式还原、BA 降代价且优于链式、二维迷你 BA 收敛。")


def main():
    use_style()
    self_check()
    plot_pair_chain_strip()
    plot_accumulate_curve()
    plot_reprojection_idea()
    plot_pipeline()
    plot_loop_idea()
    plot_ba_before_after()
    plot_2d_ba_scene()
    plot_mine()
    print("课文对照图已写入 assets/figures/")
    print("你的图：ch12_my_ba.png")


if __name__ == "__main__":
    main()
