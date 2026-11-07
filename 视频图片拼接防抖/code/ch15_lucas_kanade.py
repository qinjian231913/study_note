"""第 15 篇：稀疏光流 Lucas-Kanade。

在「视频图片拼接防抖」文件夹里运行：

    MPLCONFIGDIR=/tmp/mpl-cache .venv/bin/python code/ch15_lucas_kanade.py

图会保存到 assets/figures/ 与 assets/images/。
顶部「改这里」只影响 ch15_my_*.png；课文对照图每次按固定参数重画。
"""

from pathlib import Path

SHOW_WINDOW = False

import matplotlib

if not SHOW_WINDOW:
    matplotlib.use("Agg")

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from plot_style import INK, MUTED, TEAL, WAVE, use_style

WAVE_BGR = (40, 77, 194)
TEAL_BGR = (102, 107, 14)
INK_BGR = (21, 25, 28)

# ===== 改这里 =====
# 只影响 ch15_my_synth.png / ch15_my_arrows.png
MY_TRUE_SHIFT = (3.0, -1.5)  # 合成图的真值平移 (dx, dy)，像素
MY_WIN_SIZE = 15  # 奇数；LK 窗口边长。改大更稳，太大会混进不同运动
MY_ARROW_SCALE = 4.0  # 画箭头时把位移放大多少倍，方便看
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


# ---------------------------------------------------------------------------
# 核心：手写单层 Lucas-Kanade（与课文推导一一对应）
# ---------------------------------------------------------------------------


def spatial_grad(gray):
    """中心差分估 Ix, Iy，尺度与「每像素灰度变化」一致。

    OpenCV 的 Sobel 未归一化时，对线性斜坡响应约为 8 倍真实导数；
    若 It 用 I1−I0，尺度对不上，估出的 (u,v) 会被严重缩小。
    """
    g = gray.astype(np.float64)
    Ix = np.zeros_like(g)
    Iy = np.zeros_like(g)
    Ix[:, 1:-1] = (g[:, 2:] - g[:, :-2]) * 0.5
    Iy[1:-1, :] = (g[2:, :] - g[:-2, :]) * 0.5
    return Ix, Iy


def sample_window(img, cx, cy, half, subpixel=False):
    """以 (cx,cy) 为中心取 (2*half+1) 方窗。

    subpixel=True 时用双线性插值（迭代时在 I1 上取非整数中心）。
    """
    h, w = img.shape
    size = 2 * half + 1
    if not subpixel:
        ix, iy = int(round(cx)), int(round(cy))
        x0, x1 = ix - half, ix + half + 1
        y0, y1 = iy - half, iy + half + 1
        if x0 < 0 or y0 < 0 or x1 > w or y1 > h:
            return None
        return img[y0:y1, x0:x1].astype(np.float64)

    # getRectSubPix 的中心是浮点；要求整窗在图内
    if cx - half < 0 or cy - half < 0 or cx + half > w - 1 or cy + half > h - 1:
        return None
    patch = cv2.getRectSubPix(img.astype(np.float32), (size, size), (float(cx), float(cy)))
    return patch.astype(np.float64)


def lucas_kanade_at(I0, I1, x, y, win_size=15, iters=8):
    """在点 (x, y) 附近用窗口内同一平移假设解 (u, v)。

    亮度守恒：I0(x,y) ≈ I1(x+u, y+v)。
    一阶展开：Ix·du + Iy·dv ≈ −(I1_warped − I0)。
    单次泰勒只对很小位移准；与 OpenCV PyrLK（maxLevel=0）一样迭代累加。

    返回 dict：u, v, G（2×2 结构张量）, rhs, n_pixels, ok。
    """
    assert win_size % 2 == 1, "窗口边长必须是奇数"
    half = win_size // 2
    h, w = I0.shape
    cx, cy = float(x), float(y)
    if cx - half < 1 or cy - half < 1 or cx + half >= w - 1 or cy + half >= h - 1:
        return {"ok": False, "u": 0.0, "v": 0.0, "G": None, "rhs": None, "n": 0}

    I0f = I0.astype(np.float64)
    I1f = I1.astype(np.float64)
    Ix_full, Iy_full = spatial_grad(I0f)
    Ix = sample_window(Ix_full, cx, cy, half)
    Iy = sample_window(Iy_full, cx, cy, half)
    patch0 = sample_window(I0f, cx, cy, half)
    if Ix is None or Iy is None or patch0 is None:
        return {"ok": False, "u": 0.0, "v": 0.0, "G": None, "rhs": None, "n": 0}

    A11 = float(np.sum(Ix * Ix))
    A12 = float(np.sum(Ix * Iy))
    A22 = float(np.sum(Iy * Iy))
    G = np.array([[A11, A12], [A12, A22]], dtype=np.float64)
    det = A11 * A22 - A12 * A12
    eigs = np.linalg.eigvalsh(G)
    # 最小特征值太小 / 条件数过大 → 孔径或平坦
    cond_bad = eigs[1] > 1e-9 and (eigs[0] / eigs[1] < 1e-3)
    if eigs[0] < 1e-2 or cond_bad:
        return {
            "ok": False,
            "u": 0.0,
            "v": 0.0,
            "G": G,
            "rhs": None,
            "n": win_size * win_size,
            "det": det,
        }

    u_acc, v_acc = 0.0, 0.0
    last_rhs = None
    for _ in range(iters):
        # 在第二帧里取「已经挪到 (cx+u, cy+v)」的窗口，与第一帧模板比
        patch1 = sample_window(I1f, cx + u_acc, cy + v_acc, half, subpixel=True)
        if patch1 is None:
            break
        It = patch1 - patch0  # 还没对齐完的亮度差
        b1 = float(-np.sum(Ix * It))
        b2 = float(-np.sum(Iy * It))
        last_rhs = np.array([b1, b2], dtype=np.float64)
        du, dv = np.linalg.solve(G, last_rhs)
        u_acc += float(du)
        v_acc += float(dv)
        if abs(du) < 1e-4 and abs(dv) < 1e-4:
            break

    return {
        "ok": True,
        "u": u_acc,
        "v": v_acc,
        "G": G,
        "rhs": last_rhs,
        "n": win_size * win_size,
        "det": det,
    }


def make_corner_patch(size=80, bright=220, dark=40):
    """造一块「角点」：左上亮、其余暗，梯度在两方向都有。"""
    img = np.full((size, size), dark, dtype=np.float64)
    img[: size // 2, : size // 2] = bright
    # 轻微高斯，避免纯阶跃带来的离散伪影
    img = cv2.GaussianBlur(img, (5, 5), 0.8)
    return img


def make_edge_only(size=80):
    """只有竖直边：水平梯度强、竖直梯度弱 → 孔径问题。"""
    img = np.full((size, size), 40.0, dtype=np.float64)
    img[:, : size // 2] = 220.0
    img = cv2.GaussianBlur(img, (5, 5), 0.8)
    return img


def make_flat(size=80):
    return np.full((size, size), 128.0, dtype=np.float64)


def shift_image(img, dx, dy):
    """亚像素平移第二帧。warpAffine 约定：输出(x,y) = 输入(x-dx, y-dy)，
    即图案向 (dx, dy) 方向移动。"""
    h, w = img.shape[:2]
    M = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)
    return cv2.warpAffine(
        img.astype(np.float64),
        M,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def make_scene_pair(shift=(4.0, 2.0), size=(240, 180)):
    """合成「街景」两帧：整图同一平移，便于对照真值。"""
    w, h = size
    scene = np.full((h, w), 210, dtype=np.uint8)
    # 地面
    scene[h // 2 :, :] = 170
    # 几栋楼
    for x0, bw, bh, val in (
        (30, 28, 70, 90),
        (80, 22, 55, 70),
        (140, 35, 80, 100),
        (190, 26, 60, 80),
    ):
        top = h // 2 - bh
        scene[top : h // 2, x0 : x0 + bw] = val
    # 路面虚线
    y = h // 2 + 28
    for x0 in range(10, w - 10, 28):
        scene[y : y + 3, x0 : x0 + 16] = 245
    # 角点友好的小方块
    cv2.rectangle(scene, (55, 40), (72, 57), 40, -1)
    cv2.rectangle(scene, (56, 41), (63, 48), 230, -1)

    I0 = scene.astype(np.float64)
    I1 = shift_image(I0, shift[0], shift[1])
    return I0, I1


# ---------------------------------------------------------------------------
# 作图
# ---------------------------------------------------------------------------


def plot_window_assumption():
    """图：窗口内假设同一平移 (u,v)。"""
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0), layout="constrained")

    # 左：窗口示意
    ax = axes[0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("小窗口里：所有点共用同一个 (u, v)")
    ax.add_patch(Rectangle((2.2, 1.8), 5.2, 4.4, fill=False, ec=TEAL, lw=2.2))
    rng = np.random.default_rng(0)
    pts = rng.uniform([2.6, 2.2], [6.8, 5.6], size=(9, 2))
    for x, y in pts:
        ax.scatter([x], [y], c=INK, s=28, zorder=3)
        ax.annotate(
            "",
            xy=(x + 1.4, y + 0.55),
            xytext=(x, y),
            arrowprops=dict(arrowstyle="->", color=WAVE, lw=1.6),
        )
    ax.text(2.2, 6.6, "窗口 W", color=TEAL, fontsize=12)
    ax.text(5.0, 0.6, "箭头都一样长、一样方向", color=MUTED, fontsize=11, ha="center")

    # 右：对比「若窗口太大」
    ax = axes[1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("若窗口跨过两种运动：假设碎掉")
    ax.add_patch(Rectangle((1.2, 1.5), 7.2, 5.0, fill=False, ec=WAVE, lw=2.0, ls="--"))
    ax.plot([5, 5], [1.5, 6.5], color=MUTED, lw=1.2, ls=":")
    ax.text(2.6, 6.7, "左半：向右", color=TEAL, fontsize=11)
    ax.text(6.2, 6.7, "右半：向左", color=WAVE, fontsize=11)
    for x, y in [(2.5, 3), (3.2, 4.5), (3.8, 2.5)]:
        ax.annotate("", xy=(x + 1.2, y), xytext=(x, y), arrowprops=dict(arrowstyle="->", color=TEAL, lw=1.5))
    for x, y in [(6.2, 3), (7.0, 4.5), (7.5, 2.5)]:
        ax.annotate("", xy=(x - 1.2, y), xytext=(x, y), arrowprops=dict(arrowstyle="->", color=WAVE, lw=1.5))
    ax.text(5.0, 0.55, "一个 (u,v) 无法同时伺候两边", color=MUTED, fontsize=11, ha="center")

    save_fig(fig, "ch15_window_assumption.png")


def plot_equation_stack():
    """把窗口内每个像素的约束叠成超定方程。"""
    fig, ax = plt.subplots(figsize=(9.2, 4.6), layout="constrained")
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.set_title("亮度守恒 → 窗口内叠成「方程太多、未知数只有 2 个」")

    ax.text(0.3, 6.2, r"每个像素一条：$I_x\,u + I_y\,v = -I_t$", fontsize=14)
    ax.text(0.3, 5.3, "窗口有 n 个像素 → n 条方程，未知数只有 u、v 两个", fontsize=13)
    ax.text(0.3, 4.3, r"$A\,p = b$", fontsize=16, color=TEAL)

    matrix_txt = (
        "A 是 n×2：\n"
        "  [ Ix(1)  Iy(1) ]      [ u ]      [ -It(1) ]\n"
        "  [ Ix(2)  Iy(2) ]  ·   [ v ]  =  [ -It(2) ]\n"
        "  [  ...    ...  ]                [  ...   ]\n"
        "  [ Ix(n)  Iy(n) ]                [ -It(n) ]"
    )
    ax.text(0.3, 2.2, matrix_txt, fontsize=12, va="center")
    ax.text(0.3, 0.5, "n ≫ 2：超定。一般没有精确解，改求「残差平方和最小」的 (u,v)。", fontsize=12, color=MUTED)
    save_fig(fig, "ch15_equation_stack.png")


def plot_normal_equations():
    """法方程 / 最小二乘白话图。"""
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.0), layout="constrained")

    ax = axes[0]
    ax.set_title("最小二乘：找最贴近所有直线交汇的点")
    xs = np.linspace(-1.5, 3.5, 80)
    lines = [
        (1.0, 0.2, 1.0),
        (0.9, 0.4, 1.1),
        (1.1, -0.1, 0.95),
        (0.85, 0.5, 1.15),
        (1.05, 0.1, 1.02),
    ]
    for a, b, c in lines:
        if abs(b) < 1e-6:
            continue
        ys = (c - a * xs) / b
        ax.plot(xs, ys, color=MUTED, alpha=0.55, lw=1.2)
    ax.scatter([1.0], [1.0], s=90, c=WAVE, zorder=5, label="最小二乘解 (u,v)")
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(-0.5, 2.5)
    ax.set_xlabel("u")
    ax.set_ylabel("v")
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, alpha=0.35)

    ax = axes[1]
    ax.axis("off")
    ax.set_title("法方程：不用枚举，直接 2×2 求解")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.text(0.3, 7.0, r"最小化 $\|Ap - b\|^2$，求导置零得：", fontsize=13)
    ax.text(0.3, 5.8, r"$A^{T} A\, p = A^{T} b$", fontsize=16, color=TEAL)
    ax.text(0.3, 4.6, "左边的 2×2 俗称「结构张量」G：", fontsize=12)
    g_txt = (
        "G = [  Σ Ix²     Σ Ix Iy ]\n"
        "    [  Σ Ix Iy   Σ Iy²   ]\n\n"
        "Aᵀb = [ −Σ Ix It ]\n"
        "      [ −Σ Iy It ]"
    )
    ax.text(0.3, 2.2, g_txt, fontsize=12, va="center")
    ax.text(0.3, 0.4, "G 可逆 ↔ 窗口里两个方向都有纹理（角点友好）。", fontsize=12, color=MUTED)
    save_fig(fig, "ch15_normal_eq.png")


def plot_when_breaks():
    """何时成立、何时碎。"""
    cases = [
        ("角点 / 纹理丰富", make_corner_patch(), True, "G 满秩，(u,v) 稳"),
        ("只有一条边", make_edge_only(), False, "孔径：沿边方向不定"),
        ("几乎平坦", make_flat(), False, "梯度≈0，方程退化"),
    ]
    true_shift = (2.5, 1.0)
    fig, axes = plt.subplots(2, 3, figsize=(10.2, 6.2), layout="constrained")
    for i, (title, base, expect_ok, tip) in enumerate(cases):
        I0 = base
        I1 = shift_image(I0, true_shift[0], true_shift[1])
        c = I0.shape[0] // 2
        res = lucas_kanade_at(I0, I1, c, c, win_size=15)

        axes[0, i].imshow(I0, cmap="gray", vmin=0, vmax=255)
        axes[0, i].set_title(title)
        axes[0, i].axis("off")
        # 画窗口
        half = 7
        axes[0, i].add_patch(
            Rectangle((c - half - 0.5, c - half - 0.5), 15, 15, fill=False, ec=WAVE, lw=1.5)
        )

        axes[1, i].axis("off")
        axes[1, i].set_xlim(0, 10)
        axes[1, i].set_ylim(0, 6)
        status = "可解" if res["ok"] else "不可靠 / 失败"
        color = TEAL if res["ok"] == expect_ok else WAVE
        axes[1, i].text(5, 4.5, status, ha="center", fontsize=14, color=color, fontweight="bold")
        if res["ok"]:
            axes[1, i].text(
                5,
                3.0,
                f"估得 ({res['u']:.2f}, {res['v']:.2f})\n真值 ({true_shift[0]}, {true_shift[1]})",
                ha="center",
                fontsize=12,
            )
        else:
            axes[1, i].text(5, 3.0, "结构张量接近奇异\n（det(G) 太小）", ha="center", fontsize=12)
        axes[1, i].text(5, 1.0, tip, ha="center", fontsize=11, color=MUTED)

    fig.suptitle("同一真值平移下：纹理决定 LK 能不能钉住 (u, v)", fontsize=13)
    save_fig(fig, "ch15_when_breaks.png")


def plot_synth_hand_compare(true_shift=(2.0, 1.0), win_size=15, out_name="ch15_synth_compare.png"):
    """合成平移：手写 LK vs OpenCV 单层 PyrLK vs 真值。"""
    I0 = make_corner_patch(96)
    I1 = shift_image(I0, true_shift[0], true_shift[1])
    c = I0.shape[0] // 2
    hand = lucas_kanade_at(I0, I1, c, c, win_size=win_size)

    # OpenCV 单层：maxLevel=0
    pts = np.array([[[float(c), float(c)]]], dtype=np.float32)
    next_pts, status, err = cv2.calcOpticalFlowPyrLK(
        I0.astype(np.uint8),
        I1.astype(np.uint8),
        pts,
        None,
        winSize=(win_size, win_size),
        maxLevel=0,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )
    if status[0, 0]:
        ou = float(next_pts[0, 0, 0] - c)
        ov = float(next_pts[0, 0, 1] - c)
    else:
        ou, ov = float("nan"), float("nan")

    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.8), layout="constrained")
    axes[0].imshow(I0, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title("第 t 帧（角点）")
    axes[0].plot(c, c, "o", color=WAVE, ms=8)
    half = win_size // 2
    axes[0].add_patch(
        Rectangle((c - half - 0.5, c - half - 0.5), win_size, win_size, fill=False, ec=TEAL, lw=1.5)
    )
    axes[0].axis("off")

    axes[1].imshow(I1, cmap="gray", vmin=0, vmax=255)
    axes[1].set_title(f"第 t+1 帧（平移 {true_shift}）")
    axes[1].plot(c + true_shift[0], c + true_shift[1], "s", color=TEAL, ms=8)
    axes[1].axis("off")

    axes[2].axis("off")
    axes[2].set_xlim(0, 10)
    axes[2].set_ylim(0, 8)
    axes[2].set_title("数值对照")
    lines = [
        f"真值 (u, v) = ({true_shift[0]:.2f}, {true_shift[1]:.2f})",
        f"手写法方程  = ({hand['u']:.3f}, {hand['v']:.3f})" if hand["ok"] else "手写：失败",
        f"OpenCV 单层 = ({ou:.3f}, {ov:.3f})",
        f"窗口 = {win_size}×{win_size}，方程条数 n = {win_size * win_size}",
    ]
    if hand["ok"] and hand["G"] is not None:
        G = hand["G"]
        lines.append(f"ΣIx²={G[0,0]:.1f}  ΣIxIy={G[0,1]:.1f}  ΣIy²={G[1,1]:.1f}")
    y = 6.5
    for line in lines:
        axes[2].text(0.4, y, line, fontsize=12)
        y -= 1.1
    save_fig(fig, out_name)

    # 另存一张「手算过程」文字图（课文里引用）
    if out_name == "ch15_synth_compare.png" and hand["ok"]:
        fig2, ax = plt.subplots(figsize=(8.8, 4.2), layout="constrained")
        ax.axis("off")
        ax.set_title("手算路径（与脚本 lucas_kanade_at 一致）")
        G = hand["G"]
        rhs = hand["rhs"]
        text = (
            "1) 在窗口内算 Ix, Iy（中心差分）与亮度差 It\n"
            "2) 累加结构张量 G = [[ΣIx², ΣIxIy],[ΣIxIy, ΣIy²]]\n"
            f"   本例 G =\n"
            f"      [[{G[0,0]:10.2f}, {G[0,1]:10.2f}],\n"
            f"       [{G[1,0]:10.2f}, {G[1,1]:10.2f}]]\n"
            f"3) 迭代累加：每次解 G [du,dv] = Aᵀb，(u,v)+= (du,dv)\n"
            f"4) 收敛到 (u,v)=({hand['u']:.4f}, {hand['v']:.4f})"
            + (f"；末次 Aᵀb=[{rhs[0]:.2f}, {rhs[1]:.2f}]" if rhs is not None else "")
        )
        ax.text(0.04, 0.55, text, fontsize=12, transform=ax.transAxes, va="center")
        save_fig(fig2, "ch15_hand_steps.png")

    return hand, (ou, ov)


def draw_flow_arrows_bgr(gray_bgr_or_gray, pts0, pts1, status, scale=3.0):
    """在灰度图上画稀疏光流箭头，返回 BGR。"""
    if gray_bgr_or_gray.ndim == 2:
        vis = cv2.cvtColor(gray_bgr_or_gray.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    else:
        vis = gray_bgr_or_gray.copy()
    for i, st in enumerate(status):
        if not st:
            continue
        x0, y0 = pts0[i].ravel()
        x1, y1 = pts1[i].ravel()
        # 放大位移方便看
        xe = x0 + scale * (x1 - x0)
        ye = y0 + scale * (y1 - y0)
        p0 = (int(round(x0)), int(round(y0)))
        p1 = (int(round(xe)), int(round(ye)))
        cv2.arrowedLine(vis, p0, p1, WAVE_BGR, 2, tipLength=0.35)
        cv2.circle(vis, p0, 3, TEAL_BGR, -1)
    return vis


def plot_arrows_demo(shift=(4.0, 2.0), win_size=21, arrow_scale=3.5, out_name="ch15_arrows.png"):
    """在合成街景上跑稀疏 LK，画箭头。"""
    I0, I1 = make_scene_pair(shift=shift)
    g0 = I0.astype(np.uint8)
    g1 = I1.astype(np.uint8)

    # 用 goodFeaturesToTrack 选角点（稀疏）
    pts = cv2.goodFeaturesToTrack(g0, maxCorners=40, qualityLevel=0.05, minDistance=12)
    if pts is None:
        raise SystemExit("未找到特征点，请检查合成场景")

    next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
        g0,
        g1,
        pts,
        None,
        winSize=(win_size, win_size),
        maxLevel=0,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )
    status = status.reshape(-1)
    pts0 = pts.reshape(-1, 2)
    pts1 = next_pts.reshape(-1, 2)

    vis0 = draw_flow_arrows_bgr(g0, pts0, pts1, status, scale=arrow_scale)
    # 并排：帧0+箭头 | 帧1
    g1_bgr = cv2.cvtColor(g1, cv2.COLOR_GRAY2BGR)
    gap = 10
    h = g0.shape[0]
    canvas = np.full((h, g0.shape[1] + gap + g1.shape[1], 3), 251, dtype=np.uint8)
    canvas[:, : g0.shape[1]] = vis0
    canvas[:, g0.shape[1] + gap :] = g1_bgr
    cv2.putText(canvas, "t + arrows", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, INK_BGR, 1, cv2.LINE_AA)
    cv2.putText(
        canvas,
        "t+1",
        (g0.shape[1] + gap + 8, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        INK_BGR,
        1,
        cv2.LINE_AA,
    )
    save_bgr(out_name.replace(".png", "_opencv.png") if out_name.startswith("ch15_my") else "ch15_arrows_opencv.png", canvas)

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8), layout="constrained")
    axes[0].imshow(vis0[:, :, ::-1])
    axes[0].set_title(f"稀疏光流箭头（位移×{arrow_scale}）")
    axes[0].axis("off")
    axes[1].imshow(g1, cmap="gray")
    axes[1].set_title(f"下一帧（真值平移 {shift}）")
    axes[1].axis("off")
    save_fig(fig, out_name)

    # 统计平均位移，方便自检
    good = status.astype(bool)
    du = pts1[good, 0] - pts0[good, 0]
    dv = pts1[good, 1] - pts0[good, 1]
    return float(np.mean(du)), float(np.mean(dv)), int(np.sum(good))


def plot_opencv_intuition():
    """单层 calcOpticalFlowPyrLK 在做什么（流程示意）。"""
    fig, ax = plt.subplots(figsize=(9.0, 4.6), layout="constrained")
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.set_title("OpenCV calcOpticalFlowPyrLK（本篇只看 maxLevel=0 单层）")
    boxes = [
        (0.4, 5.0, "输入：prev 灰图\n+ 特征点坐标"),
        (3.5, 5.0, "每个点取窗口\n算 Ix,Iy,It"),
        (6.6, 5.0, "解法方程得\n(u,v)，可迭代"),
        (3.5, 1.8, "输出：nextPts\nstatus / err"),
    ]
    for x, y, t in boxes:
        ax.add_patch(Rectangle((x, y), 2.8, 1.6, facecolor="#e5f3f1", edgecolor=TEAL, lw=1.6))
        ax.text(x + 1.4, y + 0.8, t, ha="center", va="center", fontsize=11)
    # arrows
    ax.annotate("", xy=(3.4, 5.8), xytext=(3.2, 5.8), arrowprops=dict(arrowstyle="->", color=WAVE, lw=2))
    ax.annotate("", xy=(6.5, 5.8), xytext=(6.3, 5.8), arrowprops=dict(arrowstyle="->", color=WAVE, lw=2))
    ax.annotate("", xy=(4.9, 3.4), xytext=(7.9, 5.0), arrowprops=dict(arrowstyle="->", color=WAVE, lw=2))
    ax.text(
        0.4,
        0.6,
        "winSize → 窗口；maxLevel=0 → 不用金字塔（大位移下一篇再讲）；\n"
        "criteria → 迭代停止条件。稀疏：只在给定点上估，不是每个像素。",
        fontsize=12,
        color=MUTED,
    )
    save_fig(fig, "ch15_opencv_flow.png")


def plot_mine():
    """跟随「改这里」参数的两张图。"""
    plot_synth_hand_compare(
        true_shift=MY_TRUE_SHIFT,
        win_size=MY_WIN_SIZE if MY_WIN_SIZE % 2 == 1 else MY_WIN_SIZE + 1,
        out_name="ch15_my_synth.png",
    )
    plot_arrows_demo(
        shift=MY_TRUE_SHIFT,
        win_size=MY_WIN_SIZE if MY_WIN_SIZE % 2 == 1 else MY_WIN_SIZE + 1,
        arrow_scale=MY_ARROW_SCALE,
        out_name="ch15_my_arrows.png",
    )


def self_check():
    # 1) 角点 + 小平移：手写 LK 应接近真值
    I0 = make_corner_patch(80)
    true = (2.0, 1.0)
    I1 = shift_image(I0, true[0], true[1])
    c = 40
    res = lucas_kanade_at(I0, I1, c, c, win_size=15)
    assert res["ok"], "角点场景应可解"
    assert abs(res["u"] - true[0]) < 0.35, f"u 偏差过大: {res['u']}"
    assert abs(res["v"] - true[1]) < 0.35, f"v 偏差过大: {res['v']}"

    # 2) 法方程与显式最小二乘一致
    Ix = np.array([1.0, 0.0, 1.0, 2.0])
    Iy = np.array([0.0, 1.0, 1.0, 0.5])
    It = np.array([-2.0, -1.0, -3.0, -4.5])
    A = np.column_stack([Ix, Iy])
    b = -It
    G = A.T @ A
    rhs = A.T @ b
    uv_ne = np.linalg.solve(G, rhs)
    uv_lstsq = np.linalg.lstsq(A, b, rcond=None)[0]
    assert np.allclose(uv_ne, uv_lstsq, atol=1e-9)

    # 3) 平坦区应失败
    flat = make_flat(60)
    flat1 = shift_image(flat, 2, 1)
    bad = lucas_kanade_at(flat, flat1, 30, 30, win_size=11)
    assert not bad["ok"], "平坦区应判定不可靠"

    # 4) OpenCV 单层与手写同量级
    pts = np.array([[[float(c), float(c)]]], dtype=np.float32)
    nxt, st, _ = cv2.calcOpticalFlowPyrLK(
        I0.astype(np.uint8),
        I1.astype(np.uint8),
        pts,
        None,
        winSize=(15, 15),
        maxLevel=0,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 40, 0.01),
    )
    assert st[0, 0] == 1
    ou = float(nxt[0, 0, 0] - c)
    ov = float(nxt[0, 0, 1] - c)
    assert abs(ou - true[0]) < 0.5 and abs(ov - true[1]) < 0.5

    # 5) 结构张量对称且对角非负
    assert abs(res["G"][0, 1] - res["G"][1, 0]) < 1e-9
    assert res["G"][0, 0] >= 0 and res["G"][1, 1] >= 0

    print("自检通过：手写 LK、法方程=最小二乘、平坦失败、OpenCV 单层量级正确。")


def main():
    use_style()
    self_check()
    plot_window_assumption()
    plot_equation_stack()
    plot_normal_equations()
    plot_when_breaks()
    plot_opencv_intuition()
    plot_synth_hand_compare(true_shift=(2.0, 1.0), win_size=15)
    mean_u, mean_v, n_good = plot_arrows_demo(shift=(4.0, 2.0), win_size=21, arrow_scale=3.5)
    print(f"街景稀疏跟踪：成功 {n_good} 点，平均位移 ≈ ({mean_u:.2f}, {mean_v:.2f})")
    plot_mine()
    print("课文对照图已写入 assets/figures/ 与 assets/images/")
    print("你的图：ch15_my_synth.png、ch15_my_arrows.png")


if __name__ == "__main__":
    main()
