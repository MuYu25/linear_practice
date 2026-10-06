#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
用矩阵运算对图片做「旋转」和「拉伸」。

线性代数的核心思想
------------------
一张图片就是一堆像素点，每个像素的坐标 (x, y) 可以看成一个列向量 p = [x, y]^T。
所谓「旋转」「拉伸」这些几何操作，本质上都是把每个坐标向量 p 乘上一个 2x2 矩阵 M：

        p' = M @ p          （p' 是变换后的新坐标）

    * 旋转矩阵  R = [[cosθ, -sinθ],
                     [sinθ,  cosθ]]     —— 正交矩阵，R^T R = I，det(R) = 1
    * 缩放矩阵  S = [[sx,  0],
                     [0,  sy]]          —— 对角矩阵，det(S) = sx * sy

两个操作叠加，就是两个矩阵相乘：M = S @ R（先旋转、再拉伸）。
注意矩阵乘法不满足交换律，S @ R != R @ S，效果也不一样。

本脚本**不使用** PIL 现成的 rotate()/resize()，而是自己实现
「坐标乘矩阵 + 反向映射采样」，把线性代数的过程完整跑一遍。

运行：  python3 main.py
"""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ----------------------------------------------------------------------------
# 可调参数
# ----------------------------------------------------------------------------
SRC = Path("/home/m1928a1/Pictures/wallpaper/photo/wallhaven-kxqm26.jpg")

THETA_DEG = 30.0        # 旋转角度（度）
STRETCH_X = 1.5         # 水平方向拉伸倍数（>1 变宽）
STRETCH_Y = 0.8         # 垂直方向拉伸倍数（<1 变矮）

WORK_SIDE = 1600        # 输出图的最大边长（原图 8078x4778 太大，先降采样；
                        # 想看全分辨率就把它改成 None）
PANEL_SIDE = 700        # 拼图里每一格的边长
BG_COLOR = (255, 255, 255)   # 变换后露出来的背景色

HERE = Path(__file__).resolve().parent
BACKUP_DIR = HERE / "backup"
OUT_DIR = HERE / "output"


# ----------------------------------------------------------------------------
# 1. 矩阵：把「旋转」「拉伸」写成 2x2 矩阵
# ----------------------------------------------------------------------------
def rotation_matrix(theta_deg: float) -> np.ndarray:
    """标准旋转矩阵 R = [[cosθ, -sinθ], [sinθ, cosθ]]。

    注意：图像坐标系的 y 轴是**向下**的（原点在左上角），
    所以这个矩阵在屏幕上看是顺时针旋转 θ 度，在数学坐标系里是逆时针。
    """
    t = np.deg2rad(theta_deg)
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s],
                     [s,  c]], dtype=np.float64)


def stretch_matrix(sx: float, sy: float) -> np.ndarray:
    """缩放（拉伸）矩阵 S = [[sx, 0], [0, sy]]，一个对角矩阵。

    sx、sy 就是它的两个特征值，特征向量分别是 e1=(1,0) 和 e2=(0,1)：
    也就是说，x 方向和 y 方向的基向量只被「拉长」，方向不变。
    """
    return np.array([[sx, 0.0],
                     [0.0, sy]], dtype=np.float64)


# ----------------------------------------------------------------------------
# 2. 核心：把矩阵作用到整张图片上
# ----------------------------------------------------------------------------
def _bilinear(img: np.ndarray, xs: np.ndarray, ys: np.ndarray, bg) -> np.ndarray:
    """在浮点坐标 (xs, ys) 处做双线性插值采样。

    坐标可能不是整数（旋转后必然如此），所以取周围 4 个像素加权平均：
        value = (1-wy) * [(1-wx)*c00 + wx*c01] + wy * [(1-wx)*c10 + wx*c11]
    落在图像外的点直接填背景色。
    """
    h, w = img.shape[:2]

    x0 = np.floor(xs).astype(np.int32)
    y0 = np.floor(ys).astype(np.int32)
    wx = (xs - x0).astype(np.float32)[..., None]     # 小数部分就是插值权重
    wy = (ys - y0).astype(np.float32)[..., None]

    # 夹到合法范围，避免越界索引（越界的点后面会被背景色覆盖）
    x0c, y0c = np.clip(x0, 0, w - 1), np.clip(y0, 0, h - 1)
    x1c, y1c = np.clip(x0 + 1, 0, w - 1), np.clip(y0 + 1, 0, h - 1)

    src = img.astype(np.float32)
    c00, c01 = src[y0c, x0c], src[y0c, x1c]
    c10, c11 = src[y1c, x0c], src[y1c, x1c]

    top = c00 * (1.0 - wx) + c01 * wx
    bot = c10 * (1.0 - wx) + c11 * wx
    out = top * (1.0 - wy) + bot * wy

    inside = (xs >= 0) & (xs <= w - 1) & (ys >= 0) & (ys <= h - 1)
    if not inside.all():
        out[~inside] = np.asarray(bg, dtype=np.float32)

    return np.clip(out, 0, 255).astype(np.uint8)


def warp(img: np.ndarray, M: np.ndarray, bg=BG_COLOR, block_rows: int = 256) -> np.ndarray:
    """把 2x2 矩阵 M 作用到整张图片：对每个像素 p, 计算 p' = M @ p。

    这里用的是**反向映射**（inverse mapping）：
    我们不遍历原图去「推」像素（那样输出位置会有空洞），
    而是遍历输出图的每个位置 p'，用逆矩阵 M^-1 反推出它来自原图的哪个位置 p：

        p = M^-1 @ p'

    这样输出图每个像素都恰好被访问一次，不会漏点。顺便用到了逆矩阵。

    输出画布大小由原图 4 个角点变换后的包围盒决定，四周多出来的部分填背景色。
    """
    h, w = img.shape[:2]
    out_w, out_h, lo = output_canvas(h, w, M)

    out = np.empty((out_h, out_w, 3), dtype=np.uint8)
    xs_out = np.arange(out_w, dtype=np.float64) + lo[0]

    # 逐行块处理，避免一次性开辟几十 MB 的坐标网格
    for y0 in range(0, out_h, block_rows):
        y1 = min(y0 + block_rows, out_h)
        ys_out = np.arange(y0, y1, dtype=np.float64) + lo[1]
        src_x, src_y = inverse_coords(xs_out, ys_out, M)

        out[y0:y1] = _bilinear(img, src_x, src_y, bg)

    return out


def output_canvas(h: int, w: int, M: np.ndarray) -> tuple[int, int, np.ndarray]:
    """算出变换后图像需要多大的画布，以及画布原点对应的偏移量 lo。

    做法：把原图四个角点正变换过去 p' = M @ p，取这 4 个点的包围盒。
    """
    corners = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float64)
    moved = corners @ M.T                       # 每一行是一个点，p' = M @ p

    # 浮点修正：cos(90°) 算出来是 6.1e-17 而不是 0，本该是 0 的坐标会变成 1.8e-16，
    # 被 ceil() 一放大就凭空多出一整行/列空白。所以先把「几乎是整数」的值吸附成整数。
    moved = np.where(np.abs(moved - np.round(moved)) < 1e-9, np.round(moved), moved)

    lo = np.floor(moved.min(axis=0))
    hi = np.ceil(moved.max(axis=0))
    return int(hi[0] - lo[0]) + 1, int(hi[1] - lo[1]) + 1, lo


def inverse_coords(xs_out: np.ndarray, ys_out: np.ndarray,
                   M: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """反向映射：输出像素坐标 p' -> 它在原图里的浮点坐标 p = M^-1 @ p'。

    这是整个变换的关键一步：不遍历原图去「推」像素（会留空洞），
    而是遍历输出图去「拉」像素（每个输出像素恰好被访问一次）。
    """
    Minv = np.linalg.inv(M)
    gx, gy = np.meshgrid(xs_out, ys_out)
    # 展开写成标量形式，省去矩阵构造的开销
    src_x = Minv[0, 0] * gx + Minv[0, 1] * gy
    src_y = Minv[1, 0] * gx + Minv[1, 1] * gy

    # 浮点修正（和 output_canvas 里同一个坑的反面）：
    # 旋转 90° 时 cos 算出来是 6.1e-17 而不是 0，本该正好落在 0 号像素上的点会变成 -1.8e-16，
    # 后面「是否在图像内」的判断会把这一整圈边界像素误判成图像外而丢掉。
    # 先把「几乎是整数」的坐标吸附成整数，再让插值和边界判断去用。
    src_x = np.where(np.abs(src_x - np.round(src_x)) < 1e-9, np.round(src_x), src_x)
    src_y = np.where(np.abs(src_y - np.round(src_y)) < 1e-9, np.round(src_y), src_y)
    return src_x, src_y


# ----------------------------------------------------------------------------
# 3. 备份 / 读写 / 拼图的小工具
# ----------------------------------------------------------------------------
def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def backup_image(src: Path) -> Path:
    """把原图原样复制一份到 backup/，并校验哈希确保备份没有损坏。"""
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    dst = BACKUP_DIR / f"{src.stem}_backup{src.suffix}"
    shutil.copy2(src, dst)                     # copy2 连修改时间一起保留

    if sha256_of(src) != sha256_of(dst):
        raise RuntimeError("备份校验失败：备份文件与原图不一致！")
    print(f"[备份] {src}")
    print(f"     -> {dst}  (sha256 校验通过)")
    return dst


def load_working_copy(src: Path, max_side: int | None = WORK_SIDE) -> np.ndarray:
    """读入原图，必要时等比缩小成方便处理的副本，返回 numpy 数组 (H, W, 3)。"""
    im = Image.open(src).convert("RGB")
    if max_side is not None and max(im.size) > max_side:
        scale = max_side / max(im.size)
        new_size = (round(im.width * scale), round(im.height * scale))
        im = im.resize(new_size, Image.LANCZOS)
    return np.asarray(im, dtype=np.uint8)


def get_font(size: int) -> ImageFont.ImageFont:
    """找一款能显示中文的字体来画标题（找不到就退回 Pillow 自带字体）。"""
    candidates = [
        ("/usr/share/fonts/adobe-source-han-sans/SourceHanSansCN-Regular.otf", 0),
        ("/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc", 2),   # index 2 = SC
        ("/usr/share/fonts/TTF/DejaVuSans.ttf", 0),
    ]
    for path, index in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size, index=index)
    return ImageFont.load_default(size=size)


def make_collage(panels: list[tuple[str, np.ndarray]], path: Path,
                 side: int = PANEL_SIDE, cols: int = 3) -> Path:
    """把若干张图 (标题, 图片数组) 排成网格，存成一张大图方便一眼对比。"""
    font = get_font(26)
    pad, title_h, gap = 16, 44, 14
    cell_w = side + pad * 2
    cell_h = side + title_h + pad * 2

    rows = (len(panels) + cols - 1) // cols
    canvas = Image.new("RGB", (cell_w * cols + gap * (cols - 1),
                               cell_h * rows + gap * (rows - 1)), (28, 28, 32))
    draw = ImageDraw.Draw(canvas)

    for i, (title, arr) in enumerate(panels):
        r, c = divmod(i, cols)
        ox = c * (cell_w + gap)
        oy = r * (cell_h + gap)
        draw.rectangle([ox, oy, ox + cell_w - 1, oy + cell_h - 1], fill=(44, 44, 52))
        draw.text((ox + pad, oy + pad), title, fill=(240, 240, 240), font=font)

        thumb = Image.fromarray(arr)
        thumb.thumbnail((side, side), Image.LANCZOS)
        # 居中粘贴
        px = ox + (cell_w - thumb.width) // 2
        py = oy + title_h + pad + (side - thumb.height) // 2
        canvas.paste(thumb, (px, py))

    canvas.save(path)
    print(f"[效果图] -> {path}   ({canvas.width}x{canvas.height})")
    return path


def print_matrix(name: str, M: np.ndarray, fmt: str = "{: .4f}") -> None:
    print(f"{name} =")
    for row in M:
        print("    [ " + "  ".join(fmt.format(v) for v in row) + " ]")


# ----------------------------------------------------------------------------
# 4. 主流程
# ----------------------------------------------------------------------------
def main() -> None:
    if not SRC.exists():
        raise SystemExit(f"找不到原图：{SRC}")

    # --- 第 0 步：备份 -------------------------------------------------------
    backup_image(SRC)
    img = load_working_copy(SRC)
    h, w = img.shape[:2]
    print(f"\n[载入] 工作副本 {w}x{h}（原图 {Image.open(SRC).size}，已降采样便于处理）")

    # --- 第 1 步：构造矩阵 ---------------------------------------------------
    R = rotation_matrix(THETA_DEG)
    S = stretch_matrix(STRETCH_X, STRETCH_Y)
    M_SR = S @ R          # 先旋转，再拉伸
    M_RS = R @ S          # 先拉伸，再旋转
    M_inv = np.linalg.inv(M_SR)

    print("\n================ 线性代数：矩阵 ================")
    print_matrix(f"旋转矩阵 R (θ={THETA_DEG}°)", R)
    print(f"    正交性检查 R^T R =\n{np.round(R.T @ R, 6)}   (单位阵 I)")
    print(f"    det(R) = {np.linalg.det(R):.4f}   <- 行列式为 1，面积不变")
    print(f"    特征值 = {np.round(np.linalg.eigvals(R), 4)}   <- 复数 e^(±iθ)，实数域内没有特征向量")
    print()
    print_matrix(f"拉伸矩阵 S (sx={STRETCH_X}, sy={STRETCH_Y})", S)
    print(f"    det(S) = {np.linalg.det(S):.4f} = sx * sy   <- 面积被放大/缩小这个倍数")
    print(f"    特征值 = {np.round(np.linalg.eigvals(S), 4)}   <- 就是对角元，特征向量是 x/y 轴")
    print()
    print_matrix("组合矩阵 M = S @ R  (先旋转，后拉伸)", M_SR)
    print_matrix("组合矩阵 R @ S      (先拉伸，后旋转)", M_RS)
    print(f"    S@R 与 R@S 是否相等？ {np.allclose(M_SR, M_RS)}"
          f"   <- 矩阵乘法不满足交换律，两者效果不同")

    # --- 第 2 步：把矩阵作用到图片上 -----------------------------------------
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("\n================ 图像变换 ================")

    results: list[tuple[str, np.ndarray]] = [("原图 Original", img)]
    Image.fromarray(img).save(OUT_DIR / "original.jpg", quality=92)

    # (输出文件名, 图中标题, 作用在哪个矩阵上)
    jobs = [
        ("rotated",         f"旋转 {THETA_DEG:.0f}°  (R)",           R),
        ("stretched",       f"拉伸 sx={STRETCH_X}, sy={STRETCH_Y}  (S)", S),
        ("combined_SR",     "先旋转后拉伸  (M = S @ R)",            M_SR),
        ("combined_RS",     "先拉伸后旋转  (R @ S)",                M_RS),
    ]

    for name, title, mat in jobs:
        out = warp(img, mat)
        Image.fromarray(out).save(OUT_DIR / f"{name}.jpg", quality=92)
        print(f"[输出] {name}.jpg   {out.shape[1]}x{out.shape[0]}   <- {title}")
        results.append((title, out))

    # 再演示逆矩阵：把 M⁻¹ 作用在「已经被 M 变换过」的图上，应该还原成原图
    rotated_stretched = warp(img, M_SR)
    restored = warp(rotated_stretched, M_inv)
    Image.fromarray(restored).save(OUT_DIR / "inverse_restore.jpg", quality=92)
    print(f"[输出] inverse_restore.jpg   {restored.shape[1]}x{restored.shape[0]}"
          f"   <- 逆矩阵还原  (M⁻¹ @ M = I)")
    results.append(("逆矩阵还原  (M^-1 作用在 M 的结果上)", restored))

    # --- 第 3 步：拼成一张效果图 --------------------------------------------
    print()
    make_collage(results, OUT_DIR / "collage.png")

    print("\n全部完成。效果图：", OUT_DIR / "collage.png")


if __name__ == "__main__":
    main()
