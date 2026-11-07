"""第 11 篇：曝光补偿、鬼影与常见失败。

在「视频图片拼接防抖」文件夹里运行：

    MPLCONFIGDIR=/tmp/mpl-cache .venv/bin/python code/ch11_exposure.py

图会保存到 assets/figures/ 与 assets/images/。
顶部「改这里」只影响 ch11_my_exposure.png。
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
from matplotlib.patches import FancyBboxPatch, Rectangle

from plot_style import INK, MUTED, TEAL, WAVE, use_style

WAVE_BGR = (40, 77, 194)
TEAL_BGR = (102, 107, 14)
INK_BGR = (21, 25, 28)

# ===== 改这里 =====
# 只影响 ch11_my_exposure.png
# 右图相对左图再「暗」多少倍（1.0=一样亮；越大右图越暗，接缝越明显）
BRIGHTNESS_RATIO = 1.8
# 是否对右图做简单增益补偿后再拼（True 后接缝应变淡）
APPLY_GAIN_FIX = True
# 动态物体水平偏移像素（越大鬼影越重；0 表示人没动）
GHOST_SHIFT_PX = 36
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


# 右图相对左图在大场景上的水平平移；重叠宽度 = 图宽 - SHIFT
SCENE_SHIFT = 120


def make_panorama_pair(width=420, height=260, person_x=None, person_shift=0):
    """合成一对有重叠的「左右取景」。可放一个可移动的行人色块。"""
    canvas_w = width + SCENE_SHIFT
    scene = np.full((height, canvas_w, 3), 236, dtype=np.uint8)
    # 天空 / 地面
    scene[: height // 2] = (232, 220, 200)
    scene[height // 2 :] = (198, 186, 168)

    # 远山
    hills = np.array(
        [[0, 130], [90, 80], [180, 120], [280, 70], [400, 115], [540, 75], [canvas_w, 125], [canvas_w, 130], [0, 130]],
        np.int32,
    )
    cv2.fillPoly(scene, [hills], (168, 158, 142))

    # 楼宇竖条（方便看出错位 / 亮度）
    for x, w, h, color in (
        (40, 38, 110, (95, 115, 155)),
        (110, 32, 90, (75, 100, 135)),
        (220, 48, 130, (105, 95, 75)),
        (310, 36, 100, (85, 125, 115)),
        (400, 42, 120, (115, 90, 100)),
        (470, 34, 95, (90, 110, 140)),
    ):
        y0 = height // 2 - h
        cv2.rectangle(scene, (x, y0), (x + w, height // 2), color, -1)
        for wy in range(y0 + 12, height // 2 - 8, 18):
            for wx in range(x + 6, x + w - 6, 12):
                cv2.rectangle(scene, (wx, wy), (wx + 6, wy + 8), (210, 205, 195), -1)

    # 路面标线
    for x in range(20, canvas_w, 40):
        cv2.rectangle(scene, (x, height - 40), (x + 22, height - 32), (230, 225, 210), -1)

    # 近景「树干」——后面视差失败要用
    cv2.rectangle(scene, (255, 90), (275, height - 20), (60, 90, 70), -1)
    cv2.ellipse(scene, (265, 75), (38, 28), 0, 0, 360, (55, 110, 75), -1)

    # 动态行人（可选）
    if person_x is not None:
        px = int(person_x + person_shift)
        cv2.rectangle(scene, (px, height // 2 - 70), (px + 22, height // 2 + 10), WAVE_BGR, -1)
        cv2.circle(scene, (px + 11, height // 2 - 80), 10, WAVE_BGR, -1)

    # 左取景 / 右取景；重叠宽度 = width - SCENE_SHIFT
    left = scene[:, 0:width].copy()
    right = scene[:, SCENE_SHIFT : SCENE_SHIFT + width].copy()
    return scene, left, right


def default_overlap(left):
    return left.shape[1] - SCENE_SHIFT


def adjust_brightness(img, factor):
    """简单乘性曝光：factor>1 变亮，<1 变暗。"""
    out = img.astype(np.float32) * float(factor)
    return np.clip(out, 0, 255).astype(np.uint8)


def overlap_means(left, right, overlap=None):
    """左右图重叠区的平均亮度（灰度）。重叠区对应同一段场景。"""
    if overlap is None:
        overlap = default_overlap(left)
    la = left[:, -overlap:]
    ra = right[:, :overlap]
    ml = float(cv2.cvtColor(la, cv2.COLOR_BGR2GRAY).mean())
    mr = float(cv2.cvtColor(ra, cv2.COLOR_BGR2GRAY).mean())
    return ml, mr


def estimate_gain(left, right, overlap=None):
    """让右图乘增益后，重叠区平均亮度贴近左图。增益 = mean_L / mean_R。"""
    ml, mr = overlap_means(left, right, overlap)
    if mr < 1e-6:
        return 1.0, ml, mr
    return ml / mr, ml, mr


def hard_stitch(left, right, overlap=None):
    """硬切拼接：重叠区取右图，无羽化。"""
    if overlap is None:
        overlap = default_overlap(left)
    h, w = left.shape[:2]
    out_w = w + (w - overlap)
    canvas = np.zeros((h, out_w, 3), dtype=np.uint8)
    canvas[:, :w] = left
    canvas[:, w - overlap :] = right
    seam_x = w - overlap // 2
    return canvas, seam_x


def soft_blend_stitch(left, right, overlap=None):
    """重叠区线性羽化，只为展示融合后仍可能有亮度台阶。"""
    if overlap is None:
        overlap = default_overlap(left)
    h, w = left.shape[:2]
    out_w = w + (w - overlap)
    canvas = np.zeros((h, out_w, 3), dtype=np.float32)
    canvas[:, : w - overlap] = left[:, : w - overlap]
    canvas[:, w:] = right[:, overlap:]
    alpha = np.linspace(1.0, 0.0, overlap, dtype=np.float32)[None, :, None]
    lo = left[:, -overlap:].astype(np.float32)
    ro = right[:, :overlap].astype(np.float32)
    canvas[:, w - overlap : w] = lo * alpha + ro * (1.0 - alpha)
    return np.clip(canvas, 0, 255).astype(np.uint8)


def plot_ae_jump():
    """自动曝光：左右亮度不同 → 硬缝。"""
    _, left, right = make_panorama_pair()
    left_ae = adjust_brightness(left, 1.15)
    right_ae = adjust_brightness(right, 0.62)
    stitch, seam_x = hard_stitch(left_ae, right_ae)
    cv2.line(stitch, (seam_x, 0), (seam_x, stitch.shape[0] - 1), WAVE_BGR, 1)

    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.6), layout="constrained")
    axes[0].imshow(left_ae[:, :, ::-1])
    axes[0].set_title("左图（相机自动调亮）")
    axes[0].axis("off")
    axes[1].imshow(right_ae[:, :, ::-1])
    axes[1].set_title("右图（转向阴影，自动变暗）")
    axes[1].axis("off")
    axes[2].imshow(stitch[:, :, ::-1])
    axes[2].set_title("硬拼：中间一条亮度台阶")
    axes[2].axis("off")
    save_fig(fig, "ch11_ae_jump.png")
    save_bgr("ch11_left_bright.png", left_ae)
    save_bgr("ch11_right_dark.png", right_ae)


def plot_gain_idea():
    """增益补偿白话示意图。"""
    fig, ax = plt.subplots(figsize=(9.0, 3.8), layout="constrained")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.axis("off")
    ax.set_title("增益补偿：给每张图乘一个「旋钮」")

    ax.add_patch(FancyBboxPatch((0.3, 1.4), 2.4, 1.8, boxstyle="round,pad=0.05", facecolor="#fde8df", edgecolor=INK, lw=1.2))
    ax.text(1.5, 2.5, "左图 I_L", ha="center", fontsize=13, color=WAVE)
    ax.text(1.5, 1.9, "增益 g_L≈1", ha="center", fontsize=11, color=MUTED)

    ax.add_patch(FancyBboxPatch((3.8, 1.4), 2.4, 1.8, boxstyle="round,pad=0.05", facecolor="#e5f3f1", edgecolor=INK, lw=1.2))
    ax.text(5.0, 2.5, "右图 I_R", ha="center", fontsize=13, color=TEAL)
    ax.text(5.0, 1.9, "增益 g_R = μ_L/μ_R", ha="center", fontsize=11, color=MUTED)

    ax.annotate("", xy=(3.6, 2.3), xytext=(2.85, 2.3), arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.5))
    ax.annotate("", xy=(7.4, 2.3), xytext=(6.35, 2.3), arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.5))

    ax.add_patch(FancyBboxPatch((7.4, 1.2), 2.3, 2.2, boxstyle="round,pad=0.05", facecolor="#fffdf8", edgecolor=INK, lw=1.2))
    ax.text(8.55, 2.7, "重叠区希望", ha="center", fontsize=12)
    ax.text(8.55, 2.15, r"$g_L I_L \approx g_R I_R$", ha="center", fontsize=13)
    ax.text(8.55, 1.5, "接缝不再「一半亮一半暗」", ha="center", fontsize=10, color=MUTED)

    ax.text(0.3, 0.55, "白话：先在重叠区量两边平均亮度，再把偏暗的那张整体拧亮（或反过来拧暗）。", fontsize=12, color=INK)
    ax.text(0.3, 0.15, "这是「全局增益」；更高级的会按块/按通道估，但直觉一样。", fontsize=11, color=MUTED)
    save_fig(fig, "ch11_gain_idea.png")


def plot_gain_before_after():
    _, left, right = make_panorama_pair()
    left_ae = adjust_brightness(left, 1.1)
    right_ae = adjust_brightness(right, 0.55)
    gain, ml, mr = estimate_gain(left_ae, right_ae)
    right_fix = adjust_brightness(right_ae, gain)

    before = soft_blend_stitch(left_ae, right_ae)
    after = soft_blend_stitch(left_ae, right_fix)

    fig, axes = plt.subplots(2, 2, figsize=(10.0, 6.4), layout="constrained")
    axes[0, 0].imshow(left_ae[:, :, ::-1])
    axes[0, 0].set_title(f"左图　重叠均值≈{ml:.0f}")
    axes[0, 0].axis("off")
    axes[0, 1].imshow(right_ae[:, :, ::-1])
    axes[0, 1].set_title(f"右图（未补偿）　重叠均值≈{mr:.0f}")
    axes[0, 1].axis("off")
    axes[1, 0].imshow(before[:, :, ::-1])
    axes[1, 0].set_title("羽化拼接：仍有亮度台阶")
    axes[1, 0].axis("off")
    axes[1, 1].imshow(after[:, :, ::-1])
    axes[1, 1].set_title(f"右图×{gain:.2f} 后再羽化：台阶变淡")
    axes[1, 1].axis("off")
    save_fig(fig, "ch11_gain_before_after.png")
    save_bgr("ch11_stitch_before.png", before)
    save_bgr("ch11_stitch_after.png", after)


def plot_ghost():
    """动态物体：人在两帧位置不同 → 半透明双影。"""
    _, left0, right0 = make_panorama_pair(person_x=300, person_shift=0)
    _, left1, right1 = make_panorama_pair(person_x=300, person_shift=40)
    # 左图用「人在左」的场景取景，右图用「人已走开」的场景取景
    left = left0
    right = right1
    # 人为制造：右图整体稍暗，更贴近实拍
    right = adjust_brightness(right, 0.95)

    # 简单叠：重叠区取平均 → 鬼影
    overlap = default_overlap(left)
    ghost = soft_blend_stitch(left, right)

    # 单独放大重叠区对比
    crop_l = left[:, -overlap:]
    crop_r = right[:, :overlap]
    diff = cv2.absdiff(crop_l, crop_r)

    fig, axes = plt.subplots(2, 2, figsize=(10.0, 6.2), layout="constrained")
    axes[0, 0].imshow(left[:, :, ::-1])
    axes[0, 0].set_title("左图：行人偏左")
    axes[0, 0].axis("off")
    axes[0, 1].imshow(right[:, :, ::-1])
    axes[0, 1].set_title("右图：行人已走开（或走到别处）")
    axes[0, 1].axis("off")
    axes[1, 0].imshow(ghost[:, :, ::-1])
    axes[1, 0].set_title("平均/羽化后：半透明「鬼影」")
    axes[1, 0].axis("off")
    axes[1, 1].imshow(diff[:, :, ::-1])
    axes[1, 1].set_title("重叠区绝对差：动的地方最亮")
    axes[1, 1].axis("off")
    save_fig(fig, "ch11_ghost.png")
    save_bgr("ch11_ghost_stitch.png", ghost)


def plot_parallax():
    """近景视差：同一单应把远景对齐了，近处树干却对不齐。"""
    scene, left, right = make_panorama_pair()
    # 模拟「相机平移」：右图相对左图，近处树干额外水平偏移（远景偏移小）
    h, w = left.shape[:2]
    # 远景：用平移近似「正确」对齐
    M_far = np.float32([[1, 0, -8], [0, 1, 0]])
    right_aligned = cv2.warpAffine(right, M_far, (w, h), borderValue=(236, 228, 214))

    # 近景树干额外错位：在右图上把树干区域再挪一点
    right_bad = right_aligned.copy()
    trunk = right_aligned[70:240, 130:165].copy()
    right_bad[70:240, 130:165] = right_aligned[70:240, 138:173]  # 挖空感
    # 把树干贴到更偏的位置
    x0 = 118
    right_bad[70:240, x0 : x0 + 35] = trunk

    overlay_ok = cv2.addWeighted(left, 0.5, right_aligned, 0.5, 0)
    overlay_bad = cv2.addWeighted(left, 0.5, right_bad, 0.5, 0)

    # 标注近景区域
    for img in (overlay_ok, overlay_bad):
        cv2.rectangle(img, (230, 70), (290, 240), TEAL_BGR, 2)

    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.6), layout="constrained")
    axes[0].imshow(left[:, :, ::-1])
    axes[0].set_title("左图（近处有树干）")
    axes[0].axis("off")
    axes[1].imshow(overlay_ok[:, :, ::-1])
    axes[1].set_title("理想：远近都对齐（示意）")
    axes[1].axis("off")
    axes[2].imshow(overlay_bad[:, :, ::-1])
    axes[2].set_title("单应只伺候远景：近树出双影")
    axes[2].axis("off")
    save_fig(fig, "ch11_parallax.png")
    save_bgr("ch11_parallax_fail.png", overlay_bad)


def plot_failure_gallery():
    """失败图库：四种常见坑并排。"""
    _, left, right = make_panorama_pair(person_x=290, person_shift=0)
    _, _, right_g = make_panorama_pair(person_x=290, person_shift=38)

    # 1 曝光
    ae = soft_blend_stitch(adjust_brightness(left, 1.2), adjust_brightness(right, 0.5))
    # 2 鬼影
    ghost = soft_blend_stitch(left, right_g)
    # 3 视差（复用简化版）
    h, w = left.shape[:2]
    right_shift = np.roll(right, 14, axis=1)
    parallax = cv2.addWeighted(left, 0.5, right_shift, 0.5, 0)
    cv2.rectangle(parallax, (240, 70), (290, 230), WAVE_BGR, 2)
    # 4 运动模糊 / 对焦失败示意：高斯糊
    blur = soft_blend_stitch(left, cv2.GaussianBlur(right, (21, 21), 0))

    panels = [
        (ae, "① 曝光不一致\n接缝发灰/发亮"),
        (ghost, "② 动态物体\n半透明鬼影"),
        (parallax, "③ 近景视差\n近物双影"),
        (blur, "④ 模糊/虚焦\n纹理对不齐"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.0), layout="constrained")
    for ax, (img, title) in zip(axes.ravel(), panels):
        ax.imshow(img[:, :, ::-1])
        ax.set_title(title)
        ax.axis("off")
    fig.suptitle("失败图库：先认脸，再对症下药", fontsize=14, color=INK)
    save_fig(fig, "ch11_failure_gallery.png")


def plot_shooting_tips():
    """拍摄建议一览。"""
    fig, ax = plt.subplots(figsize=(9.2, 4.6), layout="constrained")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")
    ax.set_title("拍摄时就能避开大半坑")

    tips = [
        (WAVE, "锁曝光 / 手动曝光", "环拍时别让自动曝光每张乱拧；或固定 ISO、快门。"),
        (TEAL, "绕光心转，少平移", "脚尽量不动，身体原地转身；近处物体会少很多视差。"),
        (WAVE, "等行人走开", "重叠区有人穿过，宁可多等两秒，也比后期抠鬼影省事。"),
        (TEAL, "重叠给足 20%–40%", "重叠太少，增益与接缝都没地方估。"),
        (WAVE, "别对着强逆光硬转", "一边爆一边黑，增益补偿也救不了高光裁切。"),
        (TEAL, "近景少入画（或换算法）", "必须拍近物时，单应性往往不够，后面要多平面/网格。"),
    ]
    y = 5.2
    for color, title, body in tips:
        ax.add_patch(Rectangle((0.35, y - 0.35), 0.18, 0.18, color=color))
        ax.text(0.7, y - 0.22, title, fontsize=12, color=INK, fontweight="bold", va="center")
        ax.text(3.6, y - 0.22, body, fontsize=11, color=MUTED, va="center")
        y -= 0.85
    save_fig(fig, "ch11_shooting_tips.png")


def plot_mine():
    """由「改这里」参数驱动的实验图。"""
    _, left, right0 = make_panorama_pair(person_x=295, person_shift=0)
    _, _, right1 = make_panorama_pair(person_x=295, person_shift=GHOST_SHIFT_PX)
    right = right1 if GHOST_SHIFT_PX != 0 else right0

    # 右图变暗：BRIGHTNESS_RATIO 表示「暗多少倍」，即乘 1/ratio
    factor = 1.0 / max(BRIGHTNESS_RATIO, 1e-6)
    right_dark = adjust_brightness(right, factor)
    left_use = adjust_brightness(left, 1.05)

    gain, ml, mr = estimate_gain(left_use, right_dark)
    right_use = adjust_brightness(right_dark, gain) if APPLY_GAIN_FIX else right_dark
    stitch = soft_blend_stitch(left_use, right_use)

    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.5), layout="constrained")
    axes[0].imshow(left_use[:, :, ::-1])
    axes[0].set_title(f"左　μ≈{ml:.0f}")
    axes[0].axis("off")
    axes[1].imshow(right_dark[:, :, ::-1])
    axes[1].set_title(f"右（暗×{BRIGHTNESS_RATIO:.2f}）μ≈{mr:.0f}")
    axes[1].axis("off")
    fix = f"已增益×{gain:.2f}" if APPLY_GAIN_FIX else "未做增益"
    axes[2].imshow(stitch[:, :, ::-1])
    axes[2].set_title(f"拼接　{fix}　鬼移{GHOST_SHIFT_PX}px")
    axes[2].axis("off")
    save_fig(fig, "ch11_my_exposure.png")


def self_check():
    _, left, right = make_panorama_pair()
    dark = adjust_brightness(right, 0.5)
    gain, ml, mr = estimate_gain(left, dark)
    assert gain > 1.5, "偏暗右图的增益应明显大于 1"
    fixed = adjust_brightness(dark, gain)
    ml2, mr2 = overlap_means(left, fixed)
    assert abs(ml2 - mr2) < abs(ml - mr) * 0.25 + 2.0, "补偿后重叠均值应更接近"

    # 鬼影：人移动后重叠差应增大
    _, l0, r0 = make_panorama_pair(person_x=300, person_shift=0)
    _, _, r1 = make_panorama_pair(person_x=300, person_shift=40)
    ov = default_overlap(l0)
    d_still = float(cv2.absdiff(l0[:, -ov:], r0[:, :ov]).mean())
    d_move = float(cv2.absdiff(l0[:, -ov:], r1[:, :ov]).mean())
    assert d_move > d_still, "行人移动后重叠差应更大"

    # 亮度调节可逆量级
    bright = adjust_brightness(left, 1.5)
    assert bright.mean() > left.mean()

    print("自检通过：增益估测、鬼影差分、亮度调节均符合预期。")


def main():
    use_style()
    self_check()
    plot_ae_jump()
    plot_gain_idea()
    plot_gain_before_after()
    plot_ghost()
    plot_parallax()
    plot_failure_gallery()
    plot_shooting_tips()
    plot_mine()
    print("课文对照图已写入 assets/figures/ 与 assets/images/")
    print("你的图：ch11_my_exposure.png")


if __name__ == "__main__":
    main()
