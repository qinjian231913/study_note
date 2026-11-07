"""第 20 篇：裁切、路径优化与 OIS / EIS 总览。

在「视频图片拼接防抖」文件夹里运行：

    MPLCONFIGDIR=/tmp/mpl-cache .venv/bin/python code/ch20_crop_ois.py

图会保存到 assets/figures/ 与 assets/images/。
顶部「改这里」只影响 ch20_my_*.png；课文对照图按固定参数重画。
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
# 只影响 ch20_my_crop.png、ch20_my_path.png、ch20_my_budget.png
SHAKE_HIGH_HZ = 7.0  # 高频抖动「有多快」（相对归一化时间轴的振荡次数）
SHAKE_AMP = 22.0  # 高频抖动幅度（像素）。越大，补偿后黑边越宽
INTENT_AMP = 55.0  # 故意运镜幅度（慢平移/跟拍）。希望平滑后仍保留
SMOOTH_WINDOW = 9  # 滑动平均窗口。过大 → 运镜被抹掉，裁切压力也更大
CROP_RATIO = 0.86  # 输出边长 / 输入边长。越小视野越窄，能扛的抖越大
# ==================

ROOT = Path(__file__).resolve().parents[1]
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


def make_scene(width=520, height=360):
    """带竖线与路面标线的合成街景，裁切/黑边一眼能看出来。"""
    img = np.full((height, width, 3), 245, dtype=np.uint8)
    img[: height // 2] = (236, 228, 214)
    img[height // 2 :] = (210, 198, 178)
    hills = np.array(
        [[0, 180], [100, 120], [220, 155], [340, 100], [520, 165], [520, 180], [0, 180]],
        np.int32,
    )
    cv2.fillPoly(img, [hills], (180, 168, 150))
    for x, w, h, color in (
        (50, 40, 130, (90, 110, 150)),
        (130, 34, 100, (70, 95, 130)),
        (250, 48, 150, (100, 90, 70)),
        (360, 38, 115, (80, 120, 110)),
        (440, 44, 135, (110, 85, 95)),
    ):
        top = height // 2 - h
        cv2.rectangle(img, (x, top), (x + w, height // 2), color, -1)
        cv2.rectangle(img, (x, top), (x + w, height // 2), (40, 40, 40), 1)
    y = height // 2 + 36
    for x0 in range(16, width - 16, 44):
        cv2.line(img, (x0, y), (x0 + 24, y), (250, 250, 250), 3)
    cv2.rectangle(img, (210, 200), (220, 295), (60, 80, 40), -1)
    cv2.circle(img, (215, 175), 34, (50, 120, 70), -1)
    for gx in range(30, width, 70):
        cv2.line(img, (gx, 20), (gx, height - 20), (150, 135, 120), 1)
    return img


def synthesize_path(
    n=90,
    shake_amp=18.0,
    shake_hz=7.0,
    intent_amp=50.0,
    seed=4,
):
    """原始路径 = 慢运镜 + 高频手抖。返回时间、原始、运镜真值、抖动分量。"""
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 1.0, n)
    # 故意运镜：先几乎不动，中段快速跟拍平移，再稳住。
    # 这种「有转折的跟拍」最怕过平滑——大窗口会把转折抹成慢斜坡。
    sweep = 1.0 / (1.0 + np.exp(-16.0 * (t - 0.48)))
    intent_x = 30.0 + intent_amp * (0.12 * t + 0.88 * sweep)
    intent_y = 18.0 + 0.22 * intent_amp * np.sin(np.pi * sweep)
    # 高频抖动
    jitter_x = (
        shake_amp * 0.55 * np.sin(2 * np.pi * shake_hz * t)
        + shake_amp * 0.25 * np.sin(2 * np.pi * (shake_hz * 1.4) * t + 0.3)
        + shake_amp * 0.18 * rng.normal(0, 1, n)
    )
    jitter_y = (
        shake_amp * 0.5 * np.sin(2 * np.pi * (shake_hz * 0.85) * t + 0.5)
        + shake_amp * 0.2 * rng.normal(0, 1, n)
    )
    raw_x = intent_x + jitter_x
    raw_y = intent_y + jitter_y
    return t, raw_x, raw_y, intent_x, intent_y, jitter_x, jitter_y


def moving_average(x, window):
    """奇数窗口滑动平均；两端用镜像填充，避免长度变化。"""
    w = max(1, int(window))
    if w % 2 == 0:
        w += 1
    if w == 1:
        return x.copy()
    pad = w // 2
    xp = np.pad(x, pad, mode="edge")
    kernel = np.ones(w) / w
    return np.convolve(xp, kernel, mode="valid")


def optimize_path(raw_x, raw_y, window=7, follow_strength=0.35):
    """极简路径优化：适度低通压高频，再按 follow_strength 往原始路径拉一点。

    - window 越大 → 越稳，但故意运镜也更容易被抹掉，裁切预算更大
    - follow_strength ≈ 0 → 几乎只信平滑（易「漂」、运镜伤）
    - follow_strength ≈ 1 → 几乎跟原始（抖也留下）
    工程里还会加「裁切框内」硬约束；这里先建立「压高频、留运镜」的直觉。
    """
    smooth_x = moving_average(raw_x, window)
    smooth_y = moving_average(raw_y, window)
    fs = float(np.clip(follow_strength, 0.0, 1.0))
    target_x = (1.0 - fs) * smooth_x + fs * raw_x
    target_y = (1.0 - fs) * smooth_y + fs * raw_y
    return target_x, target_y, smooth_x, smooth_y


def compensation(raw_x, raw_y, target_x, target_y):
    """补偿量：要把画面从 raw 扳到 target，warp 平移 ≈ target − raw（符号约定示意）。"""
    return target_x - raw_x, target_y - raw_y


def crop_budget_needed(dx, dy, margin=4.0):
    """由最大补偿量估算至少需要裁掉多少边（像素，单侧近似）。"""
    return float(np.max(np.abs(dx)) + margin), float(np.max(np.abs(dy)) + margin)


def apply_translation_with_black(scene, dx, dy):
    """平移整幅图；移出的区域填黑，模拟补偿后的黑边。"""
    h, w = scene.shape[:2]
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(
        scene,
        M,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )


def center_crop(image, ratio):
    """按比例中心裁切，再放大回原尺寸（EIS 常见输出方式）。"""
    h, w = image.shape[:2]
    ratio = float(np.clip(ratio, 0.5, 1.0))
    cw, ch = int(w * ratio), int(h * ratio)
    x0 = (w - cw) // 2
    y0 = (h - ch) // 2
    crop = image[y0 : y0 + ch, x0 : x0 + cw]
    return cv2.resize(crop, (w, h), interpolation=cv2.INTER_LINEAR), (x0, y0, cw, ch)


def plot_black_border():
    scene = make_scene()
    # 故意补偿一个较大平移 + 轻微旋转感用切变近似
    warped = apply_translation_with_black(scene, 28, -18)
    M = cv2.getRotationMatrix2D((scene.shape[1] / 2, scene.shape[0] / 2), 3.5, 1.0)
    rotated = cv2.warpAffine(
        scene,
        M,
        (scene.shape[1], scene.shape[0]),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )

    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.8), layout="constrained")
    axes[0].imshow(scene[:, :, ::-1])
    axes[0].set_title("原始一帧")
    axes[0].axis("off")
    axes[1].imshow(warped[:, :, ::-1])
    axes[1].set_title("平移补偿 → 黑边")
    axes[1].axis("off")
    axes[2].imshow(rotated[:, :, ::-1])
    axes[2].set_title("旋转补偿 → 黑角")
    axes[2].axis("off")
    save_fig(fig, "ch20_black_border.png")

    # OpenCV 导出：黑边帧上标裁切框
    demo = warped.copy()
    ratio = 0.84
    _, (x0, y0, cw, ch) = center_crop(warped, ratio)
    cv2.rectangle(demo, (x0, y0), (x0 + cw - 1, y0 + ch - 1), TEAL_BGR, 2)
    cv2.putText(
        demo,
        f"crop {ratio:.0%}",
        (x0 + 8, y0 + 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        TEAL_BGR,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        demo,
        "black border",
        (8, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        WAVE_BGR,
        2,
        cv2.LINE_AA,
    )
    save_bgr("ch20_crop_box.png", demo)


def plot_crop_ratio():
    """裁切比 ↔ 可补偿预算的示意。"""
    ratios = np.array([1.0, 0.95, 0.9, 0.85, 0.8])
    # 正方形边长 1 时，单侧可挪预算 ≈ (1 − ratio) / 2
    budget = (1.0 - ratios) / 2.0 * 100  # 换成「相对原边长的 %」

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.0), layout="constrained")

    ax = axes[0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("同一帧：外框=原图，内框=裁切后输出")
    # 原图
    ax.add_patch(Rectangle((1.2, 1.0), 7.0, 5.0, fill=False, ec=INK, lw=2.0))
    ax.text(4.7, 6.3, "原始画幅", ha="center", color=INK, fontsize=12)
    # 三种裁切
    for ratio, color, ynote in (
        (0.92, TEAL, 0.35),
        (0.84, WAVE, -0.05),
        (0.76, MUTED, -0.45),
    ):
        w = 7.0 * ratio
        h = 5.0 * ratio
        x = 1.2 + (7.0 - w) / 2
        y = 1.0 + (5.0 - h) / 2
        ax.add_patch(Rectangle((x, y), w, h, fill=False, ec=color, lw=2.0, ls="--"))
        ax.text(8.6, y + h / 2, f"{ratio:.0%}", color=color, va="center", fontsize=11)

    ax = axes[1]
    ax.plot(ratios * 100, budget, "o-", color=WAVE, lw=2.2, ms=8)
    for r, b in zip(ratios, budget):
        ax.annotate(
            f"{b:.1f}%",
            (r * 100, b),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=10,
            color=MUTED,
        )
    finish(ax)
    ax.set_xlabel("裁切比（输出边长 / 输入边长，%）")
    ax.set_ylabel("大约单侧可补偿预算（相对边长 %）")
    ax.set_title("裁切越狠，能扛的抖越大")
    ax.set_xlim(78, 102)
    ax.invert_xaxis()

    save_fig(fig, "ch20_crop_ratio.png")


def plot_path_optimize(
    shake_amp=18.0,
    shake_hz=7.0,
    intent_amp=50.0,
    window=9,
    name="ch20_path_optimize.png",
    title_suffix="",
):
    t, raw_x, raw_y, intent_x, intent_y, _, _ = synthesize_path(
        shake_amp=shake_amp, shake_hz=shake_hz, intent_amp=intent_amp
    )
    target_x, target_y, over_x, over_y = optimize_path(
        raw_x, raw_y, window=window, follow_strength=0.25
    )
    # 过平滑：窗口很大
    over_x2 = moving_average(raw_x, max(window * 3, 21))
    over_y2 = moving_average(raw_y, max(window * 3, 21))

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.3), layout="constrained")

    ax = axes[0]
    ax.plot(t, raw_x, color=WAVE, lw=1.5, alpha=0.9, label="原始路径（含抖）")
    ax.plot(t, intent_x, color=MUTED, lw=1.6, ls="--", label="故意运镜（真值）")
    ax.plot(t, over_x2, color="#b08968", lw=2.0, label="过平滑（运镜也伤）")
    ax.plot(t, target_x, color=TEAL, lw=2.4, label="路径优化目标")
    finish(ax)
    ax.set_xlabel("时间（归一化）")
    ax.set_ylabel("水平位置（像素）")
    ax.set_title(f"压高频、留运镜{title_suffix}")
    ax.legend(frameon=False, fontsize=9)

    ax = axes[1]
    # 残差：原始 − 目标 = 被补偿掉的部分，希望主要是高频
    ax.plot(t, raw_x - intent_x, color=WAVE, lw=1.4, label="真实抖动分量")
    ax.plot(t, raw_x - target_x, color=TEAL, lw=2.0, label="实际补偿量")
    ax.plot(t, intent_x - over_x2, color="#b08968", lw=1.6, ls="--", label="过平滑误伤运镜")
    finish(ax)
    ax.set_xlabel("时间（归一化）")
    ax.set_ylabel("像素")
    ax.set_title("补偿应贴近「抖」，别贴近「运镜」")
    ax.legend(frameon=False, fontsize=9)

    save_fig(fig, name)
    return t, raw_x, raw_y, target_x, target_y, intent_x, over_x2


def plot_freq_split():
    t, raw_x, _, intent_x, _, jitter_x, _ = synthesize_path()
    fig, axes = plt.subplots(3, 1, figsize=(9.2, 6.2), layout="constrained", sharex=True)
    axes[0].plot(t, raw_x, color=WAVE, lw=1.8)
    axes[0].set_ylabel("原始")
    axes[0].set_title("路径 = 低频运镜 + 高频抖动")
    finish(axes[0])

    axes[1].plot(t, intent_x, color=TEAL, lw=2.2)
    axes[1].set_ylabel("运镜")
    axes[1].set_title("想保留：慢平移 / 跟拍")
    finish(axes[1])

    axes[2].plot(t, jitter_x, color=WAVE, lw=1.6)
    axes[2].set_ylabel("抖动")
    axes[2].set_xlabel("时间（归一化）")
    axes[2].set_title("想压掉：手抖、脚步颠簸")
    finish(axes[2])
    save_fig(fig, "ch20_freq_split.png")


def plot_crop_vs_smooth():
    """平滑越狠 → 补偿越大 → 需要的裁切越多；同时运镜误差上升。"""
    windows = [3, 5, 9, 15, 25, 41]
    t, raw_x, raw_y, intent_x, intent_y, _, _ = synthesize_path(shake_amp=20, intent_amp=55)
    budgets = []
    intent_err = []
    for w in windows:
        tx = moving_average(raw_x, w)
        ty = moving_average(raw_y, w)
        dx, dy = compensation(raw_x, raw_y, tx, ty)
        bx, by = crop_budget_needed(dx, dy, margin=2.0)
        budgets.append(max(bx, by))
        intent_err.append(float(np.mean(np.hypot(tx - intent_x, ty - intent_y))))

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.0), layout="constrained")
    ax = axes[0]
    ax.plot(windows, budgets, "o-", color=WAVE, lw=2.2)
    finish(ax)
    ax.set_xlabel("滑动平均窗口（越大越「稳」）")
    ax.set_ylabel("大约需要的单侧裁切预算（像素）")
    ax.set_title("越平滑，裁切压力越大")

    ax = axes[1]
    ax.plot(windows, intent_err, "o-", color=TEAL, lw=2.2)
    finish(ax)
    ax.set_xlabel("滑动平均窗口")
    ax.set_ylabel("相对故意运镜的平均误差（像素）")
    ax.set_title("过平滑会把跟拍也抹平")
    save_fig(fig, "ch20_smooth_cost.png")


def plot_ois_eis():
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.4), layout="constrained")

    def panel(ax, title, bullets, accent):
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 8)
        ax.axis("off")
        ax.add_patch(
            FancyBboxPatch(
                (0.4, 0.5),
                9.2,
                7.0,
                boxstyle="round,pad=0.15,rounding_size=0.3",
                linewidth=2.0,
                edgecolor=accent,
                facecolor="#fffdf8",
            )
        )
        ax.text(5, 6.8, title, ha="center", va="center", fontsize=14, color=accent)
        y = 5.7
        for line in bullets:
            ax.text(0.9, y, "•  " + line, ha="left", va="center", fontsize=11.5, color=INK)
            y -= 1.05

    panel(
        axes[0],
        "OIS 光学防抖",
        [
            "镜组或传感器做物理微移",
            "曝光进行时就稳住进光",
            "擅长：高频小抖、减轻模糊",
            "局限：行程有限；大幅晃不动",
            "通常发生在「拍照/录像硬件层」",
        ],
        WAVE,
    )
    panel(
        axes[1],
        "EIS 电子防抖",
        [
            "数字估计路径 + warp 补偿",
            "用裁切换稳定视野",
            "擅长：较大位移、可做路径规划",
            "局限：吃像素；难修运动模糊",
            "可纯视觉，也可陀螺辅助（后文）",
        ],
        TEAL,
    )
    save_fig(fig, "ch20_ois_eis.png")


def plot_pipeline():
    """为陀螺 EIS 铺路：整条链路地图。"""
    fig, ax = plt.subplots(figsize=(10.8, 3.6), layout="constrained")
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 4)
    ax.axis("off")
    ax.set_title("从二维视觉 EIS 到陀螺 EIS：本篇站在哪")

    boxes = [
        (0.3, "估路径\n视觉/光流", WAVE),
        (3.0, "路径优化\n压抖留运镜", TEAL),
        (5.7, "补偿 warp\n+ 裁切", WAVE),
        (8.4, "OIS 硬件\n减模糊", MUTED),
        (11.1, "陀螺 EIS\n第 22–23 篇", TEAL),
    ]
    for x, text, color in boxes:
        ax.add_patch(
            FancyBboxPatch(
                (x, 1.1),
                2.3,
                1.8,
                boxstyle="round,pad=0.12,rounding_size=0.25",
                linewidth=1.8,
                edgecolor=color,
                facecolor="#fffdf8",
            )
        )
        ax.text(x + 1.15, 2.0, text, ha="center", va="center", fontsize=11, color=INK)

    for x in (2.6, 5.3, 8.0, 10.7):
        ax.annotate(
            "",
            xy=(x + 0.35, 2.0),
            xytext=(x, 2.0),
            arrowprops=dict(arrowstyle="->", color=INK, lw=1.6),
        )

    ax.text(
        7,
        0.45,
        "本篇重点：黑边/裁切比、路径优化直觉、OIS 与 EIS 分工。卷帘快门与陀螺仪在后面展开。",
        ha="center",
        fontsize=11,
        color=MUTED,
    )
    save_fig(fig, "ch20_pipeline.png")


def plot_budget_demo(crop_ratio=0.86, shake_amp=22.0, window=9, name="ch20_budget.png"):
    """给定裁切比，看补偿是否「装得下」。"""
    t, raw_x, raw_y, _, _, _, _ = synthesize_path(shake_amp=shake_amp, intent_amp=INTENT_AMP)
    target_x, target_y, _, _ = optimize_path(raw_x, raw_y, window=window, follow_strength=0.25)
    dx, dy = compensation(raw_x, raw_y, target_x, target_y)
    need_x, need_y = crop_budget_needed(dx, dy)
    # 假设画幅 400×300，单侧预算
    W, H = 400.0, 300.0
    allow_x = W * (1.0 - crop_ratio) / 2.0
    allow_y = H * (1.0 - crop_ratio) / 2.0
    ok = (need_x <= allow_x) and (need_y <= allow_y)

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.1), layout="constrained")
    ax = axes[0]
    ax.plot(t, dx, color=WAVE, lw=1.6, label="水平补偿")
    ax.plot(t, dy, color=TEAL, lw=1.6, label="竖直补偿")
    ax.axhline(allow_x, color=WAVE, ls="--", lw=1.2, label=f"水平预算 ±{allow_x:.1f}px")
    ax.axhline(-allow_x, color=WAVE, ls="--", lw=1.2)
    ax.axhline(allow_y, color=TEAL, ls=":", lw=1.2, label=f"竖直预算 ±{allow_y:.1f}px")
    ax.axhline(-allow_y, color=TEAL, ls=":", lw=1.2)
    finish(ax)
    ax.set_xlabel("时间")
    ax.set_ylabel("补偿量（像素）")
    ax.set_title("补偿曲线 vs 裁切预算")
    ax.legend(frameon=False, fontsize=9)

    ax = axes[1]
    ax.axis("off")
    status = "装得下：裁切框内不会露黑边" if ok else "装不下：某帧补偿超出预算 → 仍会露边或需更大裁切"
    ax.text(0.05, 0.85, f"裁切比 CROP_RATIO = {crop_ratio:.0%}", fontsize=13, color=INK)
    ax.text(0.05, 0.68, f"需要预算 ≈ ({need_x:.1f}, {need_y:.1f}) px", fontsize=13, color=WAVE)
    ax.text(0.05, 0.52, f"允许预算 ≈ ({allow_x:.1f}, {allow_y:.1f}) px", fontsize=13, color=TEAL)
    ax.text(0.05, 0.30, status, fontsize=13, color=TEAL if ok else WAVE, wrap=True)
    ax.text(
        0.05,
        0.10,
        "公式直觉：单侧预算 ≈ 边长 × (1 − 裁切比) / 2",
        fontsize=12,
        color=MUTED,
    )
    ax.set_title("裁切预算自检")
    save_fig(fig, name)
    return ok, need_x, need_y, allow_x, allow_y


def plot_before_after_frames():
    """同一场景：抖的一帧、补偿后带黑边、裁切放大后。"""
    scene = make_scene(480, 320)
    # 取路径中抖动较大的一帧
    _, raw_x, raw_y, _, _, _, _ = synthesize_path(n=60, shake_amp=24)
    target_x, target_y, _, _ = optimize_path(raw_x, raw_y, window=9, follow_strength=0.2)
    i = int(np.argmax(np.hypot(target_x - raw_x, target_y - raw_y)))
    dx = float(target_x[i] - raw_x[i])
    dy = float(target_y[i] - raw_y[i])
    # 用场景中心裁一块当作「当前帧」，再补偿
    frame = scene.copy()
    compensated = apply_translation_with_black(frame, dx, dy)
    cropped, _ = center_crop(compensated, 0.86)

    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.6), layout="constrained")
    for ax, img, title in (
        (axes[0], frame[:, :, ::-1], "输入帧"),
        (axes[1], compensated[:, :, ::-1], f"补偿 dx={dx:.0f}, dy={dy:.0f}"),
        (axes[2], cropped[:, :, ::-1], "裁切放大（无黑边输出）"),
    ):
        ax.imshow(img)
        ax.set_title(title)
        ax.axis("off")
    save_fig(fig, "ch20_before_after.png")


def plot_mine():
    plot_path_optimize(
        shake_amp=SHAKE_AMP,
        shake_hz=SHAKE_HIGH_HZ,
        intent_amp=INTENT_AMP,
        window=SMOOTH_WINDOW,
        name="ch20_my_path.png",
        title_suffix=f"（窗={SMOOTH_WINDOW}, 抖={SHAKE_AMP}）",
    )
    plot_budget_demo(
        crop_ratio=CROP_RATIO,
        shake_amp=SHAKE_AMP,
        window=SMOOTH_WINDOW,
        name="ch20_my_budget.png",
    )

    scene = make_scene()
    warped = apply_translation_with_black(scene, SHAKE_AMP * 0.9, -SHAKE_AMP * 0.55)
    out, (x0, y0, cw, ch) = center_crop(warped, CROP_RATIO)
    canvas = np.full((scene.shape[0], scene.shape[1] * 2 + 12, 3), 251, dtype=np.uint8)
    left = warped.copy()
    cv2.rectangle(left, (x0, y0), (x0 + cw - 1, y0 + ch - 1), TEAL_BGR, 2)
    canvas[:, : scene.shape[1]] = left
    canvas[:, scene.shape[1] + 12 :] = out
    cv2.putText(canvas, "with crop box", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, WAVE_BGR, 2)
    cv2.putText(
        canvas,
        f"output CROP={CROP_RATIO:.0%}",
        (scene.shape[1] + 22, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        TEAL_BGR,
        2,
    )
    save_bgr("ch20_my_crop.png", canvas)
    # 也存一份 matplotlib 对照，方便和 figures 一起看
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.6), layout="constrained")
    axes[0].imshow(left[:, :, ::-1])
    axes[0].set_title(f"补偿后 + 裁切框（比={CROP_RATIO:.0%}）")
    axes[0].axis("off")
    axes[1].imshow(out[:, :, ::-1])
    axes[1].set_title("裁切放大输出")
    axes[1].axis("off")
    save_fig(fig, "ch20_my_crop_view.png")


def self_check():
    # 1) 裁切预算公式
    W, ratio = 400.0, 0.9
    allow = W * (1.0 - ratio) / 2.0
    if abs(allow - 20.0) > 1e-9:
        raise SystemExit(f"裁切预算公式自检失败：got {allow}")

    # 2) 中心裁切尺寸
    img = np.zeros((200, 300, 3), dtype=np.uint8)
    cropped, (x0, y0, cw, ch) = center_crop(img, 0.8)
    if cropped.shape != (200, 300, 3) or cw != 240 or ch != 160:
        raise SystemExit(f"中心裁切自检失败：{cropped.shape}, box={[x0,y0,cw,ch]}")

    # 3) 路径：优化后高频能量应下降（用二阶差分近似）
    _, raw_x, raw_y, intent_x, intent_y, _, _ = synthesize_path(shake_amp=20)
    tx, ty, _, _ = optimize_path(raw_x, raw_y, window=9, follow_strength=0.25)

    def highfreq_energy(x):
        return float(np.mean(np.diff(x, n=2) ** 2))

    if highfreq_energy(tx) >= highfreq_energy(raw_x) * 0.95:
        raise SystemExit("路径优化未压高频")

    # 4) 过平滑会抹掉跟拍转折：在中段（t≈0.35–0.65）误差应更大
    over = moving_average(raw_x, 41)
    mild = moving_average(raw_x, 9)
    mid = slice(int(0.35 * len(raw_x)), int(0.65 * len(raw_x)))
    err_mild_mid = float(np.mean(np.abs(mild[mid] - intent_x[mid])))
    err_over_mid = float(np.mean(np.abs(over[mid] - intent_x[mid])))
    if err_over_mid <= err_mild_mid:
        raise SystemExit("过平滑应在跟拍转折段伤得更重")
    # 优化路径整体应比原始路径更接近运镜（抖被压了）
    if float(np.mean(np.abs(tx - intent_x))) >= float(np.mean(np.abs(raw_x - intent_x))):
        raise SystemExit("路径优化后应比原始更接近运镜")

    # 5) 黑边：平移后边缘应有纯黑像素
    scene = make_scene(120, 80)
    warped = apply_translation_with_black(scene, 15, 0)
    if not np.all(warped[:, :10] == 0):
        raise SystemExit("黑边自检失败：左侧应有黑边")

    print("自检通过：裁切预算、中心裁切、路径优化压高频留运镜、黑边填充均正常。")


def main():
    use_style()
    self_check()
    plot_black_border()
    plot_crop_ratio()
    plot_freq_split()
    plot_path_optimize()
    plot_crop_vs_smooth()
    plot_ois_eis()
    plot_pipeline()
    plot_before_after_frames()
    plot_budget_demo(crop_ratio=0.86, shake_amp=18.0, window=9, name="ch20_budget.png")
    plot_mine()
    print("课文对照图已写入 assets/figures/ 与 assets/images/")
    print("你的图：ch20_my_path.png、ch20_my_budget.png、ch20_my_crop.png（只跟随顶部参数）")


if __name__ == "__main__":
    main()
