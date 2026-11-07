"""第 21 篇：Rolling Shutter——一行一行的时间差。

    MPLCONFIGDIR=/tmp/mpl-cache .venv/bin/python code/ch21_rolling_shutter.py

顶部「改这里」只影响 ch21_my_rs.png（以及同名说明图）。
"""

from pathlib import Path

SHOW_WINDOW = False

import matplotlib

if not SHOW_WINDOW:
    matplotlib.use("Agg")

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Rectangle

from plot_style import INK, MUTED, TEAL, WAVE, use_style

# ===== 改这里 =====
# 卷帘读出行时间（相对一帧时长的比例）。越大，上下行时间差越大，斜切越狠。
READOUT_FRAC = 0.55
# 纯旋转：偏航角速度（度/帧）。正 = 相机向右转，竖线向左斜。
YAW_DEG_PER_FRAME = 8.0
# 纯平移：横向速度（像素/帧，按「中景」深度换算后的像平面速度）。
TRANS_PX_PER_FRAME = 28.0
# 场景里竖柱子数量（改大可看更密的「弯柱子」）
N_PILLARS = 5
# ==================

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "assets" / "figures"
IMG = ROOT / "assets" / "images"
WAVE_BGR = (40, 77, 194)
TEAL_BGR = (102, 107, 14)
INK_BGR = (28, 25, 21)


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


def make_pillar_scene(w=360, h=240, n_pillars=5, seed=7):
    """合成「街景」：竖直柱子 + 地面线，方便看出斜切。"""
    rng = np.random.default_rng(seed)
    img = np.full((h, w, 3), 248, dtype=np.uint8)
    # 天空与地面
    img[: h // 2] = (236, 228, 214)
    img[h // 2 :] = (214, 206, 192)
    cv2.line(img, (0, h // 2), (w - 1, h // 2), (170, 160, 148), 2)

    xs = np.linspace(40, w - 40, n_pillars)
    colors = [
        WAVE_BGR,
        TEAL_BGR,
        (60, 90, 160),
        (40, 120, 90),
        (30, 70, 180),
    ]
    for i, x in enumerate(xs):
        x0 = int(x)
        color = colors[i % len(colors)]
        width = 14 + int(rng.integers(0, 6))
        cv2.rectangle(img, (x0 - width // 2, 20), (x0 + width // 2, h - 24), color, -1)
        # 柱顶小帽，帮助辨认「弯」
        cv2.rectangle(
            img,
            (x0 - width // 2 - 4, 12),
            (x0 + width // 2 + 4, 22),
            (50, 45, 40),
            -1,
        )

    # 远处一排窗户横线（旋转时会变成斜网格）
    for y in (55, 90, 125):
        cv2.line(img, (20, y), (w - 20, y), (190, 180, 168), 1)
    # 中心十字，帮助对照
    cv2.drawMarker(img, (w // 2, h // 2), (90, 85, 78), cv2.MARKER_CROSS, 16, 1)
    return img


def row_time(y, h, readout_frac, t0=0.0):
    """第 y 行的曝光时刻（相对帧起点，单位：帧时长）。"""
    return t0 + (y / max(h - 1, 1)) * readout_frac


def apply_rolling_shutter(src, dx_per_row, fill=(251, 248, 242)):
    """按每一行的水平位移 dx_per_row[y] 做卷帘畸变（只改列，不改行）。"""
    h, w = src.shape[:2]
    assert len(dx_per_row) == h
    out = np.full_like(src, fill)
    xs = np.arange(w, dtype=np.float32)
    for y in range(h):
        dx = float(dx_per_row[y])
        map_x = xs - dx
        map_y = np.full(w, y, dtype=np.float32)
        row = cv2.remap(
            src,
            map_x.reshape(1, -1),
            map_y.reshape(1, -1),
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=fill,
        )
        out[y] = row[0]
    return out


def dx_from_yaw(h, yaw_deg_per_frame, readout_frac, focal_px=None, w=360):
    """纯偏航：像平面水平位移 ≈ f * ω * t(y)。

    yaw_deg_per_frame 是「整帧时长」内转过的角度；行 y 只经历 t(y) 这段时间。
    """
    if focal_px is None:
        focal_px = 0.9 * w
    omega = np.deg2rad(yaw_deg_per_frame)  # rad / frame
    ys = np.arange(h, dtype=np.float64)
    t = row_time(ys, h, readout_frac)
    # 相对「帧中心行」的位移，避免整幅漂走，只留下斜切
    t_mid = row_time((h - 1) / 2.0, h, readout_frac)
    dx = focal_px * omega * (t - t_mid)
    return dx.astype(np.float32)


def dx_from_translation(h, trans_px_per_frame, readout_frac):
    """纯横向平移：像平面速度近似恒定（远景平面），位移 ∝ t(y)。"""
    ys = np.arange(h, dtype=np.float64)
    t = row_time(ys, h, readout_frac)
    t_mid = row_time((h - 1) / 2.0, h, readout_frac)
    dx = trans_px_per_frame * (t - t_mid)
    return dx.astype(np.float32)


def plot_global_vs_rolling():
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2), layout="constrained")

    # 左：全局快门——整幅同一时刻
    ax = axes[0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("全局快门 Global Shutter")
    ax.add_patch(Rectangle((1.2, 0.8), 4.2, 4.2, fill=False, lw=2, edgecolor=INK))
    for i, y in enumerate(np.linspace(1.2, 4.6, 6)):
        ax.plot([1.4, 5.2], [y, y], color=TEAL, lw=2.2, alpha=0.85)
        ax.text(5.5, y - 0.1, f"t = t₀", fontsize=10, color=MUTED if i else TEAL)
    ax.annotate(
        "所有行同一时刻曝光",
        xy=(3.3, 0.35),
        ha="center",
        fontsize=11,
        color=TEAL,
    )
    ax.add_patch(
        FancyArrowPatch(
            (7.2, 1.2),
            (7.2, 4.8),
            arrowstyle="->",
            mutation_scale=12,
            color=MUTED,
            lw=1.2,
        )
    )
    ax.text(7.5, 3.0, "行号 y", rotation=90, va="center", color=MUTED, fontsize=11)

    # 右：卷帘——行间有时间差
    ax = axes[1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("卷帘快门 Rolling Shutter")
    ax.add_patch(Rectangle((1.2, 0.8), 4.2, 4.2, fill=False, lw=2, edgecolor=INK))
    # 图像坐标：上方 y 小、时刻早；下方 y 大、时刻晚（从上到下读出）
    times = ["t₀+5Δ", "t₀+4Δ", "t₀+3Δ", "t₀+2Δ", "t₀+Δ", "t₀"]
    ys_draw = np.linspace(1.2, 4.6, 6)  # 下→上画
    for i, (y, lab) in enumerate(zip(ys_draw, times)):
        # i=0 在底部（最晚），i=5 在顶部（最早）
        ax.plot([1.4, 5.2], [y, y], color=WAVE, lw=2.2, alpha=0.35 + 0.12 * (5 - i))
        ax.text(5.5, y - 0.1, lab, fontsize=10, color=WAVE if i in (0, 5) else MUTED)
    ax.annotate(
        "从上到下依次曝光（手机常见）",
        xy=(3.3, 0.35),
        ha="center",
        fontsize=11,
        color=WAVE,
    )
    ax.add_patch(
        FancyArrowPatch(
            (7.2, 4.8),
            (7.2, 1.2),
            arrowstyle="->",
            mutation_scale=12,
            color=MUTED,
            lw=1.2,
        )
    )
    ax.text(7.5, 3.0, "读出方向 ↓（时间变晚）", rotation=90, va="center", color=MUTED, fontsize=11)
    save_fig(fig, "ch21_global_vs_rolling.png")


def plot_row_time_model():
    h = 240
    readout = 0.6
    ys = np.arange(h)
    t = row_time(ys, h, readout)

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0), layout="constrained")

    ax = axes[0]
    ax.plot(t, ys, color=WAVE, lw=2.4)
    ax.invert_yaxis()
    ax.set_xlabel("曝光时刻 t（单位：帧时长）")
    ax.set_ylabel("行号 y（图像向下为正）")
    ax.set_title("逐行曝光时刻：t(y) = t₀ + (y/H)·T_readout")
    ax.axvline(0, color=MUTED, lw=0.8, ls="--")
    ax.axvline(readout, color=TEAL, lw=1.2, ls="--")
    ax.text(readout + 0.02, 20, "T_readout", color=TEAL, fontsize=11)
    ax.grid(True, alpha=0.35)

    ax = axes[1]
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_title("白话：快门像「窗帘」从上往下扫")
    lines = [
        "全局快门：整幅像「啪」一下拍下同一瞬间。",
        "卷帘快门：传感器一行行读出，上下行差一小段时间。",
        "",
        "若相机在读出期间还在动：",
        "  · 上面的行记住「较早」的姿态",
        "  · 下面的行记住「较晚」的姿态",
        "  → 同一根竖直柱子，上下对不齐，看起来被斜切。",
        "",
        "公式只做一件事：给每一行贴上自己的时钟。",
    ]
    ax.text(0.3, 9.2, "\n".join(lines), fontsize=12, va="top", family="sans-serif")
    save_fig(fig, "ch21_row_time_model.png")


def plot_phenomena():
    src = make_pillar_scene(n_pillars=5)
    h, w = src.shape[:2]

    # 斜切：中等角速度
    dx_skew = dx_from_yaw(h, yaw_deg_per_frame=10.0, readout_frac=0.7, w=w)
    skew = apply_rolling_shutter(src, dx_skew)

    # 果冻：上下半幅方向相反的剪切（模拟高频抖动半周期）
    ys = np.arange(h, dtype=np.float32)
    jelly_amp = 22.0
    dx_jelly = jelly_amp * np.sin(2 * np.pi * ys / h)
    jelly = apply_rolling_shutter(src, dx_jelly)

    # 弯柱子：角速度随时间变化（二次项）→ 竖线变弯
    t = row_time(ys.astype(np.float64), h, 0.75)
    t_mid = row_time((h - 1) / 2.0, h, 0.75)
    # 角加速度：位移含 t² 项
    dx_bend = (18.0 * (t - t_mid) + 120.0 * ((t - t_mid) ** 2)).astype(np.float32)
    bend = apply_rolling_shutter(src, dx_bend)

    fig, axes = plt.subplots(1, 4, figsize=(11.2, 3.5), layout="constrained")
    panels = [
        (src, "静止 / 全局快门\n柱子竖直"),
        (skew, "斜切 skew\n匀速转动 → 直线仍直但倾斜"),
        (jelly, "果冻 jello\n高频抖动 → 波浪扭曲"),
        (bend, "弯柱子\n变速转动 → 竖线变弯"),
    ]
    for ax, (im, title) in zip(axes, panels):
        ax.imshow(im[:, :, ::-1])
        ax.set_title(title, fontsize=11)
        ax.axis("off")
    save_fig(fig, "ch21_phenomena.png")
    save_bgr("ch21_scene.png", src)


def plot_rotation_vs_translation():
    src = make_pillar_scene(n_pillars=5)
    h, w = src.shape[:2]
    readout = 0.6

    dx_rot = dx_from_yaw(h, yaw_deg_per_frame=12.0, readout_frac=readout, w=w)
    dx_trans = dx_from_translation(h, trans_px_per_frame=36.0, readout_frac=readout)
    rot = apply_rolling_shutter(src, dx_rot)
    trans = apply_rolling_shutter(src, dx_trans)

    fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.2), layout="constrained")

    axes[0, 0].imshow(src[:, :, ::-1])
    axes[0, 0].set_title("原图（全局快门）")
    axes[0, 0].axis("off")
    axes[0, 1].imshow(rot[:, :, ::-1])
    axes[0, 1].set_title("只有偏航旋转 + RS")
    axes[0, 1].axis("off")
    axes[0, 2].plot(dx_rot, np.arange(h), color=WAVE, lw=2)
    axes[0, 2].invert_yaxis()
    axes[0, 2].set_xlabel("水平位移 Δx（px）")
    axes[0, 2].set_ylabel("行 y")
    axes[0, 2].set_title("旋转：Δx ∝ ω·t(y)\n（近似线性斜切）")
    axes[0, 2].grid(True, alpha=0.35)

    axes[1, 0].imshow(src[:, :, ::-1])
    axes[1, 0].set_title("原图（再对照）")
    axes[1, 0].axis("off")
    axes[1, 1].imshow(trans[:, :, ::-1])
    axes[1, 1].set_title("只有横向平移 + RS")
    axes[1, 1].axis("off")
    axes[1, 2].plot(dx_trans, np.arange(h), color=TEAL, lw=2)
    axes[1, 2].invert_yaxis()
    axes[1, 2].set_xlabel("水平位移 Δx（px）")
    axes[1, 2].set_ylabel("行 y")
    axes[1, 2].set_title("平移（恒定像速）：\n也是线性斜切，幅度∝v")
    axes[1, 2].grid(True, alpha=0.35)

    save_fig(fig, "ch21_rotation_vs_translation.png")
    save_bgr("ch21_rs_rotation.png", rot)
    save_bgr("ch21_rs_translation.png", trans)


def plot_flow_pollution():
    """示意：RS 让「本该竖直的边」产生伪水平位移，光流会当成真实运动。"""
    src = make_pillar_scene(w=320, h=200, n_pillars=4)
    h, w = src.shape[:2]
    dx = dx_from_yaw(h, yaw_deg_per_frame=9.0, readout_frac=0.65, w=w)
    rs = apply_rolling_shutter(src, dx)

    # 伪造「帧间光流」：把 RS 斜切误当成整幅运动场
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.6), layout="constrained")
    axes[0].imshow(src[:, :, ::-1])
    axes[0].set_title("真实场景：柱子静止竖直")
    axes[0].axis("off")

    axes[1].imshow(rs[:, :, ::-1])
    axes[1].set_title("RS 拍到的一帧：柱子已斜")
    # 画几条伪光流箭头（上行和下行方向相反相对中线）
    for y in (40, 80, 120, 160):
        x = w // 2
        axes[1].annotate(
            "",
            xy=(x + dx[y] * 0.15 + 18, y),
            xytext=(x, y),
            arrowprops=dict(arrowstyle="->", color=WAVE, lw=1.6),
        )
    axes[1].axis("off")

    axes[2].axis("off")
    axes[2].set_xlim(0, 10)
    axes[2].set_ylim(0, 10)
    axes[2].set_title("稳像 / 光流如何被骗")
    text = (
        "光流假设：灰度变化来自「物体或相机」的运动。\n"
        "RS 额外贡献：同一帧内部，上下行来自不同时刻。\n\n"
        "结果：\n"
        "· 竖直边被估出虚假水平流\n"
        "· 帧间变换（相似/仿射）被污染\n"
        "· 二维稳像按错误路径补偿 → 果冻更明显\n\n"
        "所以后面要用陀螺：角速度几乎不受纹理影响，\n"
        "还能按「每一行」反投，专门掰直斜切。"
    )
    axes[2].text(0.2, 9.4, text, fontsize=11, va="top")
    save_fig(fig, "ch21_flow_pollution.png")


def plot_readout_timeline():
    """时间轴：一行行被点亮。"""
    fig, ax = plt.subplots(figsize=(9.2, 3.4), layout="constrained")
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis("off")
    ax.set_title("一帧内部：读出指针从上扫到下")

    # 时间轴
    ax.annotate(
        "",
        xy=(11.2, 1.0),
        xytext=(0.8, 1.0),
        arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.4),
    )
    ax.text(11.3, 0.7, "时间", color=MUTED, fontsize=11)

    n = 6
    for i in range(n):
        x = 1.2 + i * 1.6
        # 小画框
        ax.add_patch(Rectangle((x, 2.0), 1.2, 2.2, fill=False, lw=1.3, edgecolor=INK))
        # 已曝光的行（从上到下）
        for r in range(i + 1):
            yy = 4.0 - r * 0.32
            ax.plot([x + 0.1, x + 1.1], [yy, yy], color=WAVE, lw=2.0, alpha=0.4 + 0.1 * r)
        ax.text(x + 0.6, 1.45, f"t={i}", ha="center", fontsize=10, color=TEAL if i == 0 else MUTED)
        if i == n - 1:
            ax.text(x + 0.6, 4.45, "读完", ha="center", fontsize=10, color=WAVE)

    ax.text(
        0.8,
        0.35,
        "全局快门没有这条「扫过」的过程；卷帘快门的每一帧，其实是一小段延时摄影。",
        fontsize=11,
        color=MUTED,
    )
    save_fig(fig, "ch21_readout_timeline.png")


def plot_mine():
    """由顶部「改这里」参数生成的对照图。"""
    src = make_pillar_scene(n_pillars=N_PILLARS)
    h, w = src.shape[:2]
    dx_r = dx_from_yaw(h, YAW_DEG_PER_FRAME, READOUT_FRAC, w=w)
    dx_t = dx_from_translation(h, TRANS_PX_PER_FRAME, READOUT_FRAC)
    img_r = apply_rolling_shutter(src, dx_r)
    img_t = apply_rolling_shutter(src, dx_t)

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.6), layout="constrained")
    axes[0].imshow(src[:, :, ::-1])
    axes[0].set_title("原图")
    axes[0].axis("off")
    axes[1].imshow(img_r[:, :, ::-1])
    axes[1].set_title(
        f"你的旋转 RS\nyaw={YAW_DEG_PER_FRAME}°/帧, readout={READOUT_FRAC}"
    )
    axes[1].axis("off")
    axes[2].imshow(img_t[:, :, ::-1])
    axes[2].set_title(
        f"你的平移 RS\nv={TRANS_PX_PER_FRAME}px/帧, readout={READOUT_FRAC}"
    )
    axes[2].axis("off")
    save_fig(fig, "ch21_my_rs.png")
    save_bgr("ch21_my_rotation.png", img_r)
    save_bgr("ch21_my_translation.png", img_t)


def self_check():
    h = 100
    # 行时刻单调递增
    t0 = row_time(0, h, 0.5)
    t1 = row_time(h - 1, h, 0.5)
    assert t0 == 0.0
    assert abs(t1 - 0.5) < 1e-9

    # 零角速度 → 零位移
    dx0 = dx_from_yaw(h, 0.0, 0.5, w=200)
    assert np.allclose(dx0, 0.0)

    # 零平移 → 零位移
    dx1 = dx_from_translation(h, 0.0, 0.5)
    assert np.allclose(dx1, 0.0)

    # 相对 t_mid 后，整段位移关于「时间中点」反对称；两端符号相反
    dx = dx_from_yaw(h, 10.0, 0.8, w=200)
    assert dx[0] * dx[-1] < 0
    assert abs(float(np.mean(dx))) < 0.5  # 中心对齐后均值接近 0

    # 读出越大，两端位移差越大
    dx_small = dx_from_yaw(h, 10.0, 0.2, w=200)
    dx_large = dx_from_yaw(h, 10.0, 0.9, w=200)
    span_small = abs(dx_small[-1] - dx_small[0])
    span_large = abs(dx_large[-1] - dx_large[0])
    assert span_large > span_small

    # remap 不改变静止图
    src = make_pillar_scene(w=80, h=60, n_pillars=3)
    out = apply_rolling_shutter(src, np.zeros(60, dtype=np.float32))
    assert out.shape == src.shape
    assert np.mean(np.abs(out.astype(np.int16) - src.astype(np.int16))) < 1.0

    print("自检通过：行时刻、旋转/平移位移、零运动不变性均正常。")


def main():
    use_style()
    self_check()
    plot_global_vs_rolling()
    plot_row_time_model()
    plot_readout_timeline()
    plot_phenomena()
    plot_rotation_vs_translation()
    plot_flow_pollution()
    plot_mine()
    print("课文对照图已写入 assets/figures/ 与 assets/images/")
    print("你的图：ch21_my_rs.png")


if __name__ == "__main__":
    main()
