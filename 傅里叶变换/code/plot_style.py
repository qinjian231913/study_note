"""各篇脚本共用的作图颜色和字体。

一般不用单独运行本文件。直接运行某一课的脚本即可，例如：

    python code/ch00_sine.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager

INK = "#1c1915"
MUTED = "#5c564c"
WAVE = "#c24d28"
TEAL = "#0e6b66"
GRID = "#e6dccb"
PAPER = "#fbf8f2"


def use_style():
    """让图的底色、字体和网页一致。优先使用微软雅黑，避免中文变成方框。"""
    chosen = None
    for path in (
        Path(r"C:\Windows\Fonts\msyh.ttc"),
        Path(r"C:\Windows\Fonts\msyh.ttf"),
        Path(r"C:\Windows\Fonts\simhei.ttf"),
        Path(r"C:\Windows\Fonts\Deng.ttf"),
    ):
        if not path.exists():
            continue
        try:
            font_manager.fontManager.addfont(str(path))
            chosen = font_manager.FontProperties(fname=str(path)).get_name()
            break
        except (OSError, RuntimeError, ValueError):
            continue

    if chosen:
        plt.rcParams["font.family"] = chosen
        plt.rcParams["font.sans-serif"] = [chosen, "Microsoft YaHei", "SimHei"]
    else:
        plt.rcParams["font.family"] = "sans-serif"
        plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]

    plt.rcParams.update(
        {
            "axes.unicode_minus": False,
            "figure.facecolor": PAPER,
            "savefig.facecolor": PAPER,
            "axes.facecolor": "#fffdf8",
            "axes.edgecolor": "#2a241c",
            "axes.labelcolor": INK,
            "axes.titlecolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "text.color": INK,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "axes.linewidth": 0.8,
        }
    )
    print(f"作图字体：{chosen or '系统默认（若中文是方框，请确认已安装微软雅黑）'}")


def finish(ax):
    """去掉上、右边框，画出网格和 y = 0 的中线。"""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, zorder=0)
    ax.axhline(0, color=MUTED, lw=0.8, zorder=1)
    ax.set_axisbelow(True)


if __name__ == "__main__":
    print("这是共用样式。请运行： python code/ch00_sine.py")
