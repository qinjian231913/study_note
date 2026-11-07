# 章节撰写规范（给写作 Agent）

读者是**计算机视觉零基础小白**。默认会一点 Python，不懂光流/单应/IMU。宁可写长、写碎，也不要跳步骤。公式必须配白话。

## 路径与已有资源

- 系列根目录：`/Users/qinjian/project/study_note/视频图片拼接防抖/`
- 样式：只用已有 `assets/style.css`（勿改设计体系）
- 共用作图：`code/plot_style.py`（颜色 WAVE=`#c24d28`，TEAL=`#0e6b66`）
- 环境：优先 `源目录/.venv`：`MPLCONFIGDIR=/tmp/mpl-cache .venv/bin/python code/chXX_*.py`
- 模板范文（必须先读再仿）：
  - `00-intro.html`、`01-coords.html`、`02-affine.html`
  - `code/ch00_intro.py`、`code/ch01_coords.py`、`code/ch02_affine.py`
- 大纲与章节目标：`index.html` 里对应 `chNN` 的简介

## 你只准改这些文件

1. 本篇 HTML：`NN-slug.html`（见任务指定文件名）
2. 本篇脚本：`code/chNN_*.py`
3. 本篇生成的 `assets/figures/chNN_*.png`、`assets/images/chNN_*`（如有）
4. **不要改** `index.html`、`assets/style.css`、其他篇章、`.venv`

上一篇 HTML 里若有「（待写）」的下一篇链接，可改成去掉「（待写）」；不要大改上一篇正文。

## HTML 结构（必须齐全）

```
header.site（brand + nav：目录、上一篇、当前、下一篇）
kicker / h1 / lede / meta pills
nav.toc（本篇目录，锚点）
多个 h2 小节：现象→概念→图→公式白话→交互→脚本→练习→自测
callout.note / callout.run / callout.try
figure + figcaption（图 N-k　……）
playground（至少一处可交互，canvas 或滑块）
details.quiz 自测 ≥3 题
下一篇预告 + nav.crumbs
```

文风：简体中文；短句；「先看见再公式」；类比生活场景；重点篇（光流/RS/陀螺）交互与失败案例更多。

颜色（canvas/脚本）：INK `#1c1915`，MUTED `#5c564c`，WAVE `#c24d28`，TEAL `#0e6b66`。

OpenCV 画图用 BGR；给 Matplotlib 显示前 `[:, :, ::-1]`。

## Python 脚本约定

- 文件头 docstring 写清如何运行
- 顶部 `# ===== 改这里 =====` 可调参数，只影响 `chNN_my_*.png`
- `SHOW_WINDOW = False` + Agg backend
- `self_check()` + 打印「自检通过」
- 生成全部课文对照图后退出

## 交付时在最终回复写明

- 创建的 HTML / py 路径
- 生成了哪些图
- 是否脚本跑通
- 本篇核心三句话小结
