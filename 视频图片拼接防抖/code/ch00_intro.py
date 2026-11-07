"""第 0 篇：现象、目标与开发环境。

在「视频图片拼接防抖」文件夹里运行：

    python code/ch00_intro.py

图会保存到 assets/figures/ 与 assets/images/。
文件顶部「改这里」只影响 ch00_my_overlap.png 与 ch00_my_shake.png。
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

from plot_style import INK, MUTED, TEAL, WAVE, finish, use_style

# plot_style 里是 RGB；OpenCV 画图要用 BGR
WAVE_BGR = (40, 77, 194)  # #c24d28
TEAL_BGR = (102, 107, 14)  # #0e6b66
INK_BGR = (21, 25, 28)

# ===== 改这里 =====
# 只影响 ch00_my_overlap.png 与 ch00_my_shake.png
OVERLAP_SHIFT_X = 28  # 右图相对「正确位置」多偏了多少像素（越大越对不齐）
OVERLAP_SHIFT_Y = 12
SHAKE_AMPLITUDE = 18  # 晃动幅度（像素）。改大，轨迹更抖
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
    ok = cv2.imwrite(str(path), image_bgr)
    if not ok:
        raise SystemExit(f"写图失败：{path}")
    print(f"已保存 {path}")


def make_scene(width=640, height=360):
    """造一张带几何图案的合成场景，后面从中裁两张「照片」。"""
    img = np.full((height, width, 3), 245, dtype=np.uint8)

    # 地面与天空
    img[: height // 2] = (236, 228, 214)
    img[height // 2 :] = (210, 198, 178)

    # 远山（梯形色块）
    hills = np.array([[0, 180], [120, 110], [260, 160], [400, 90], [640, 170], [640, 180], [0, 180]], np.int32)
    cv2.fillPoly(img, [hills], (180, 168, 150))

    # 几栋「楼」——竖条，方便看出错位
    for x, w, h, color in (
        (70, 46, 140, (90, 110, 150)),
        (150, 38, 110, (70, 95, 130)),
        (320, 54, 160, (100, 90, 70)),
        (430, 42, 125, (80, 120, 110)),
        (520, 50, 145, (110, 85, 95)),
    ):
        top = height // 2 - h
        cv2.rectangle(img, (x, top), (x + w, height // 2), color, -1)
        cv2.rectangle(img, (x, top), (x + w, height // 2), (40, 40, 40), 1)

    # 路面标线：水平虚线，错位时特别明显
    y = height // 2 + 40
    for x0 in range(20, width - 20, 50):
        cv2.line(img, (x0, y), (x0 + 28, y), (250, 250, 250), 3)

    # 近处一棵「树」：竖干 + 圆冠，跟踪时也好认
    cv2.rectangle(img, (250, 200), (262, 300), (60, 80, 40), -1)
    cv2.circle(img, (256, 175), 38, (50, 120, 70), -1)

    # 网格点：后面画框、对点用
    for gx in range(40, width, 80):
        for gy in range(40, height, 70):
            cv2.circle(img, (gx, gy), 2, (160, 140, 120), -1)

    return img


def crop_view(scene, left, top, view_w=320, view_h=240):
    """从大图裁一块，当作一次「拍照」。"""
    h, w = scene.shape[:2]
    left = int(np.clip(left, 0, w - view_w))
    top = int(np.clip(top, 0, h - view_h))
    return scene[top : top + view_h, left : left + view_w].copy(), left, top


def side_by_side(left_bgr, right_bgr, gap=12):
    h = max(left_bgr.shape[0], right_bgr.shape[0])
    canvas = np.full((h, left_bgr.shape[1] + gap + right_bgr.shape[1], 3), 251, dtype=np.uint8)
    canvas[: left_bgr.shape[0], : left_bgr.shape[1]] = left_bgr
    x1 = left_bgr.shape[1] + gap
    canvas[: right_bgr.shape[0], x1 : x1 + right_bgr.shape[1]] = right_bgr
    return canvas


def overlay_misaligned(left_bgr, right_bgr, shift_x, shift_y, alpha=0.45):
    """把右图半透明叠到左图上，故意带上位移，露出「对不齐」。"""
    h = max(left_bgr.shape[0], right_bgr.shape[0] + abs(shift_y))
    w = max(left_bgr.shape[1], right_bgr.shape[1] + abs(shift_x))
    base = np.full((h, w, 3), 251, dtype=np.uint8)
    base[: left_bgr.shape[0], : left_bgr.shape[1]] = left_bgr

    overlay = base.copy()
    x0 = max(0, shift_x)
    y0 = max(0, shift_y)
    rh, rw = right_bgr.shape[:2]
    # 右图起点允许为负：用切片对齐
    src_x0 = max(0, -shift_x)
    src_y0 = max(0, -shift_y)
    dst_x0 = max(0, shift_x)
    dst_y0 = max(0, shift_y)
    src = right_bgr[src_y0:rh, src_x0:rw]
    dh, dw = src.shape[:2]
    region = overlay[dst_y0 : dst_y0 + dh, dst_x0 : dst_x0 + dw]
    blended = cv2.addWeighted(region, 1 - alpha, src, alpha, 0)
    overlay[dst_y0 : dst_y0 + dh, dst_x0 : dst_x0 + dw] = blended

    # 标出右图外框
    cv2.rectangle(
        overlay,
        (dst_x0, dst_y0),
        (dst_x0 + dw - 1, dst_y0 + dh - 1),
        TEAL_BGR,
        2,
    )
    return overlay


def draw_labeled_box(image_bgr, x, y, w, h, label, color):
    out = image_bgr.copy()
    cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)
    cv2.putText(out, label, (x + 4, max(18, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
    return out


def plot_two_views_and_overlay():
    scene = make_scene()
    left, lx, ly = crop_view(scene, 40, 50)
    # 正确位置应再向右 140；这里故意少移/多偏，制造错位
    right_true, _, _ = crop_view(scene, 40 + 140, 50)
    right_bad, _, _ = crop_view(scene, 40 + 140 + 28, 50 + 12)

    pair = side_by_side(left, right_bad)
    # 在并排图上画框，证明 OpenCV 画图可用
    pair = draw_labeled_box(pair, 8, 8, 120, 40, "view A", WAVE_BGR)
    pair = draw_labeled_box(pair, left.shape[1] + 12 + 8, 8, 120, 40, "view B", TEAL_BGR)
    save_bgr("ch00_two_views.png", pair)

    wrong = overlay_misaligned(left, right_bad, 140 + 28, 12)
    right = overlay_misaligned(left, right_true, 140, 0)
    # 用 matplotlib 拼成对照图（中文标题更稳）
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), layout="constrained")
    for ax, img, title in (
        (axes[0], wrong[:, :, ::-1], "硬叠：对不齐（重影）"),
        (axes[1], right[:, :, ::-1], "对齐后：重叠区重合"),
    ):
        ax.imshow(img)
        ax.set_title(title)
        ax.axis("off")
    save_fig(fig, "ch00_overlay_compare.png")

    # 场景总览：标出两块取景框
    overview = scene.copy()
    cv2.rectangle(overview, (40, 50), (40 + 320, 50 + 240), WAVE_BGR, 3)
    cv2.rectangle(overview, (40 + 140, 50), (40 + 140 + 320, 50 + 240), TEAL_BGR, 3)
    cv2.putText(overview, "A", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, WAVE_BGR, 2, cv2.LINE_AA)
    cv2.putText(overview, "B", (40 + 150, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, TEAL_BGR, 2, cv2.LINE_AA)
    save_bgr("ch00_scene_crops.png", overview)


def shaky_path(n=40, amplitude=18, seed=3):
    """模拟手持：理想直线路径 + 高频抖动。"""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, n)
    # 故意缓慢平移（运镜）+ 抖动
    ideal_x = 80 + 160 * t
    ideal_y = 70 + 20 * np.sin(np.pi * t)
    jitter_x = amplitude * 0.35 * np.sin(2 * np.pi * 7 * t) + amplitude * 0.25 * rng.normal(0, 1, n)
    jitter_y = amplitude * 0.45 * np.sin(2 * np.pi * 5 * t + 0.4) + amplitude * 0.2 * rng.normal(0, 1, n)
    raw_x = ideal_x + jitter_x
    raw_y = ideal_y + jitter_y
    # 简单滑动平均当作「平滑后的目标路径」
    k = 5
    kernel = np.ones(k) / k
    smooth_x = np.convolve(raw_x, kernel, mode="same")
    smooth_y = np.convolve(raw_y, kernel, mode="same")
    # 边界处卷积会糊，用理想路径贴边
    smooth_x[:2] = raw_x[:2]
    smooth_y[:2] = raw_y[:2]
    smooth_x[-2:] = raw_x[-2:]
    smooth_y[-2:] = raw_y[-2:]
    return t, raw_x, raw_y, smooth_x, smooth_y, ideal_x, ideal_y


def plot_shake(amplitude=18, name="ch00_shake_path.png", title_prefix=""):
    t, raw_x, raw_y, smooth_x, smooth_y, ideal_x, ideal_y = shaky_path(amplitude=amplitude)
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.3), layout="constrained")

    ax = axes[0]
    ax.plot(raw_x, raw_y, color=WAVE, lw=1.8, label="原始晃动路径")
    ax.plot(smooth_x, smooth_y, color=TEAL, lw=2.4, label="平滑后路径")
    ax.scatter([raw_x[0]], [raw_y[0]], color=WAVE, s=36, zorder=3)
    ax.scatter([raw_x[-1]], [raw_y[-1]], color=TEAL, s=36, zorder=3)
    ax.set_aspect("equal", adjustable="datalim")
    finish(ax)
    ax.set_xlabel("取景中心 x（像素）")
    ax.set_ylabel("取景中心 y（像素）")
    ax.set_title(f"{title_prefix}相机路径（俯视轨迹）")
    ax.legend(frameon=False, fontsize=10)

    ax = axes[1]
    ax.plot(t, raw_x - ideal_x, color=WAVE, lw=1.8, label="水平偏差")
    ax.plot(t, raw_y - ideal_y, color=TEAL, lw=1.8, label="竖直偏差")
    ax.axhline(0, color=MUTED, lw=0.8)
    finish(ax)
    ax.set_xlabel("时间（归一化）")
    ax.set_ylabel("相对理想运镜的偏差（像素）")
    ax.set_title("抖得有多厉害")
    ax.legend(frameon=False, fontsize=10)

    save_fig(fig, name)
    return raw_x, raw_y


def make_shake_strip():
    """从晃动路径上取几帧裁切，横排展示「同一场景却在抖」。"""
    scene = make_scene()
    _, raw_x, raw_y, *_ = shaky_path(amplitude=18)
    frames = []
    for idx in (0, 8, 16, 24, 32):
        crop, _, _ = crop_view(scene, raw_x[idx] - 160, raw_y[idx] - 120, 200, 150)
        label = crop.copy()
        cv2.putText(
            label,
            f"t={idx}",
            (8, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            WAVE_BGR,
            2,
            cv2.LINE_AA,
        )
        frames.append(label)
    gap = 8
    h = frames[0].shape[0]
    w = frames[0].shape[1]
    canvas = np.full((h, len(frames) * w + (len(frames) - 1) * gap, 3), 251, dtype=np.uint8)
    for i, fr in enumerate(frames):
        x0 = i * (w + gap)
        canvas[:, x0 : x0 + w] = fr
    save_bgr("ch00_shake_frames.png", canvas)


def plot_goals():
    """一张概念图：拼接变宽 vs 防抖变稳。"""
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.8), layout="constrained")

    ax = axes[0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.set_aspect("equal")
    from matplotlib.patches import Rectangle

    ax.add_patch(Rectangle((1.0, 1.8), 3.2, 2.4, fill=False, ec=WAVE, lw=2.0))
    ax.add_patch(Rectangle((3.0, 1.8), 3.2, 2.4, fill=False, ec=TEAL, lw=2.0))
    ax.add_patch(Rectangle((1.0, 1.8), 5.2, 2.4, fill=False, ec=INK, lw=2.6, ls=(0, (4, 3))))
    ax.text(2.4, 4.5, "图 A", color=WAVE, ha="center", fontsize=12)
    ax.text(4.8, 4.5, "图 B", color=TEAL, ha="center", fontsize=12)
    ax.text(5.0, 0.7, "拼起来 → 视野变宽", color=INK, ha="center", fontsize=12)
    ax.set_title("拼接要解决的事")
    ax.axis("off")

    ax = axes[1]
    t = np.linspace(0, 1, 80)
    shaky = 0.35 * np.sin(2 * np.pi * 9 * t) + 0.12 * np.sin(2 * np.pi * 3.3 * t)
    smooth = 0.08 * np.sin(2 * np.pi * 1.2 * t)
    ax.plot(t, shaky, color=WAVE, lw=1.8, label="晃动")
    ax.plot(t, smooth, color=TEAL, lw=2.4, label="稳住后")
    finish(ax)
    ax.set_xlabel("时间")
    ax.set_ylabel("画面偏移")
    ax.set_title("防抖要解决的事")
    ax.legend(frameon=False, fontsize=10)

    save_fig(fig, "ch00_goals.png")


def plot_mine():
    scene = make_scene()
    left, _, _ = crop_view(scene, 40, 50)
    right, _, _ = crop_view(scene, 40 + 140 + OVERLAP_SHIFT_X, 50 + OVERLAP_SHIFT_Y)
    bad = overlay_misaligned(left, right, 140 + OVERLAP_SHIFT_X, OVERLAP_SHIFT_Y)
    fig, ax = plt.subplots(figsize=(7.2, 4.0), layout="constrained")
    ax.imshow(bad[:, :, ::-1])
    ax.set_title(
        f"你的错位参数：额外偏移 dx={OVERLAP_SHIFT_X}，dy={OVERLAP_SHIFT_Y}"
    )
    ax.axis("off")
    save_fig(fig, "ch00_my_overlap.png")

    plot_shake(
        amplitude=SHAKE_AMPLITUDE,
        name="ch00_my_shake.png",
        title_prefix=f"幅度={SHAKE_AMPLITUDE}　",
    )


def write_demo_video():
    """写一段很短的晃动裁切视频，确认 VideoWriter 可用。"""
    scene = make_scene()
    _, raw_x, raw_y, *_ = shaky_path(n=30, amplitude=16)
    IMG.mkdir(parents=True, exist_ok=True)
    path = IMG / "ch00_shake_demo.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, 12.0, (320, 240))
    if not writer.isOpened():
        print("提示：本机未能打开 mp4 写入，跳过演示视频（图片已足够完成本篇）。")
        return
    for x, y in zip(raw_x, raw_y):
        frame, _, _ = crop_view(scene, x - 160, y - 120, 320, 240)
        writer.write(frame)
    writer.release()
    print(f"已保存 {path}")


def self_check():
    scene = make_scene()
    if scene.shape != (360, 640, 3):
        raise SystemExit(f"场景尺寸不对：{scene.shape}")
    crop, left, top = crop_view(scene, 10, 20, 100, 80)
    if crop.shape != (80, 100, 3) or left != 10 or top != 20:
        raise SystemExit("裁切自检失败")
    # OpenCV 读回刚刚要写的图之前，先检查 BGR 往返
    probe = np.zeros((16, 16, 3), dtype=np.uint8)
    probe[:] = (40, 80, 160)
    IMG.mkdir(parents=True, exist_ok=True)
    probe_path = IMG / "_probe.png"
    cv2.imwrite(str(probe_path), probe)
    back = cv2.imread(str(probe_path))
    probe_path.unlink(missing_ok=True)
    if back is None or back.shape != (16, 16, 3):
        raise SystemExit("OpenCV 读写自检失败")
    print("自检通过：场景可裁切，OpenCV 能读写图片。")


def main():
    use_style()
    self_check()
    plot_goals()
    plot_two_views_and_overlay()
    plot_shake(amplitude=18, name="ch00_shake_path.png")
    make_shake_strip()
    write_demo_video()
    plot_mine()
    print("课文对照图已写入 assets/figures/ 与 assets/images/")
    print("你的图：ch00_my_overlap.png、ch00_my_shake.png（只跟随顶部参数）")


if __name__ == "__main__":
    main()
