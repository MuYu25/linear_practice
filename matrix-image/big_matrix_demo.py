#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
为什么一个 2x2 矩阵就能变换一整张图片？—— 把那个「图片大小的大矩阵」真的造出来。

疑问：图片本身不就是个 H x W 的矩阵吗，怎么用一个 2x2 就够了？
答案：两个矩阵作用的东西不一样——
      2x2 矩阵作用在「坐标」上，图片矩阵是「坐标处的取值」。
      把前者展开到后者的大小，就能得到一个 N x N 的大矩阵（N = 像素总数）。

本脚本把大矩阵显式造出来验证这一点，并诚实地算一下它有多大。

运行：  python3 big_matrix_demo.py
"""

from __future__ import annotations

import numpy as np

from main import inverse_coords, load_working_copy, output_canvas, rotation_matrix, warp

SRC = "/home/m1928a1/Pictures/wallpaper/photo/wallhaven-kxqm26.jpg"


# ----------------------------------------------------------------------------
# 把 2x2 的坐标变换「展开」成 N x N 的权重矩阵（COO 稀疏格式）
# ----------------------------------------------------------------------------
def weight_matrix(h: int, w: int, M: np.ndarray):
    """构造大矩阵 W，使得对灰度图 f 有：  g = W @ f   （f 拉直成长度 N 的向量）

    W[i, j] 的含义：输出图的第 i 个像素，从原图的第 j 个像素身上取了多少权重。
    权重就是双线性插值的 4 个系数，所以每一行**最多只有 4 个非零元**。
    """
    out_w, out_h, lo = output_canvas(h, w, M)
    xs_out = np.arange(out_w, dtype=np.float64) + lo[0]
    ys_out = np.arange(out_h, dtype=np.float64) + lo[1]
    src_x, src_y = inverse_coords(xs_out, ys_out, M)   # p = M^-1 @ p'

    x0 = np.floor(src_x).astype(np.int64)
    y0 = np.floor(src_y).astype(np.int64)
    wx, wy = src_x - x0, src_y - y0
    inside = (src_x >= 0) & (src_x <= w - 1) & (src_y >= 0) & (src_y <= h - 1)

    out_idx = np.arange(out_h * out_w).reshape(out_h, out_w)

    rows, cols, vals = [], [], []
    for dy, weight_y in ((0, 1.0 - wy), (1, wy)):        # 上下两个邻居
        for dx, weight_x in ((0, 1.0 - wx), (1, wx)):    # 左右两个邻居
            jx = np.clip(x0 + dx, 0, w - 1)
            jy = np.clip(y0 + dy, 0, h - 1)
            weight = weight_x * weight_y * inside        # 图像外的点权重为 0
            keep = weight != 0
            rows.append(out_idx[keep])
            cols.append((jy * w + jx)[keep])
            vals.append(weight[keep])

    return (np.concatenate(rows), np.concatenate(cols), np.concatenate(vals),
            (out_h, out_w))


def apply_big_matrix(coo, flat, bg: float = 0.0) -> np.ndarray:
    """算 W @ flat —— 用散列相加实现（等价于稀疏矩阵乘向量），不需要 scipy。"""
    rows, cols, vals, (out_h, out_w) = coo
    n_out = out_h * out_w

    acc = np.zeros(n_out)
    np.add.at(acc, rows, vals * flat[cols])

    row_sum = np.zeros(n_out)                 # 每行权重之和：内部点为 1，图像外为 0
    np.add.at(row_sum, rows, vals)

    return bg * (1.0 - row_sum) + acc         # 落在图像外的位置就显示背景色


def show_grid(arr: np.ndarray, title: str) -> None:
    print(f"{title}")
    for row in arr:
        print("    " + "  ".join(f"{v:5.0f}" for v in row))
    print()


# ----------------------------------------------------------------------------
def main() -> None:
    R90 = rotation_matrix(90.0)

    # ========== 第 1 部分：4x4 的玩具图，把 16x16 大矩阵打印出来 ==========
    print("=" * 74)
    print("第 1 部分：4x4 的小图 -> 16x16 的大矩阵（旋转 90°）")
    print("=" * 74)

    toy = np.arange(1, 17, dtype=np.float64).reshape(4, 4)   # 用 1~16 标记每个像素
    show_grid(toy, "原图 f（每个数字是一个像素）:")

    coo = weight_matrix(4, 4, R90)
    rows, cols, vals, (oh, ow) = coo

    W_dense = np.zeros((oh * ow, 4 * 4))
    W_dense[rows, cols] = vals
    print(f"大矩阵 W，形状 {W_dense.shape[0]}x{W_dense.shape[1]}"
          f"（= 输出像素数 x 输入像素数）:")
    for row in W_dense:
        print("    [" + " ".join(f"{v:.0f}" for v in row) + "]")
    print()

    out_flat = apply_big_matrix(coo, toy.ravel())
    show_grid(out_flat.reshape(oh, ow), "W @ f 的结果（= 把图旋转 90°）:")
    print(f"    W 的每一行/每一列都恰好只有一个 1 —— 这叫**置换矩阵**（permutation matrix），")
    print(f"    它的作用就是「把像素搬个位置」。旋转 90° 是置换矩阵，旋转 30° 就不是了（见第 2 部分）。")

    # ========== 第 2 部分：真实图片上验证「大矩阵 == warp()」 ==========
    print()
    print("=" * 74)
    print("第 2 部分：真实图片，旋转 30°，验证 大矩阵 与 warp() 结果完全一致")
    print("=" * 74)

    img = load_working_copy(SRC, max_side=200)
    h, w = img.shape[:2]
    # 转灰度：单通道，才是标准的「矩阵乘向量」。两端用**完全相同**的输入，否则比出来的是取整误差
    gray = (img.astype(np.float64) @ [0.299, 0.587, 0.114]).astype(np.uint8)

    theta = 30.0
    M = rotation_matrix(theta)
    coo = weight_matrix(h, w, M)
    rows, cols, vals, (out_h, out_w) = coo
    n_out = out_h * out_w

    mine = apply_big_matrix(coo, gray.ravel().astype(np.float64), bg=0.0).reshape(out_h, out_w)
    from_warp = warp(np.repeat(gray[..., None], 3, axis=2), M, bg=(0, 0, 0))[:, :, 0].astype(np.float64)

    err = np.abs(mine - from_warp)
    nonzero_rows = (np.bincount(rows, minlength=n_out) > 0).sum()

    print(f"\n输入图        : {w} x {h}  ->  拉直成向量 (N = {w * h})")
    print(f"输出图        : {out_w} x {out_h}  ->  拉直成向量 (N = {n_out})")
    print(f"大矩阵 W      : {n_out} x {w * h}")
    print(f"非零元个数    : {len(vals):,}  ({len(vals) / (n_out * w * h) * 100:.4f}% 非零 —— 极度稀疏)")
    print(f"            每行最多 4 个，正是双线性插值的 4 邻域权重")
    print(f"            （有 {n_out - nonzero_rows:,} 行整行为 0：那些输出像素落在图像之外，显示背景）")
    # 两个结果的差异应该只来自「最后把浮点值截断成 uint8」时的临界取舍。
    # 验证方法：看不一致的那些像素，真值是不是正好压在整数上。
    mine_u8 = np.floor(np.clip(mine, 0, 255))
    mismatch = mine_u8 != from_warp
    frac_off = np.abs(mine[mismatch] - np.round(mine[mismatch]))

    print(f"\n最大逐像素误差: {err.max():.6f}   <- 只差在最后把浮点结果截断成 uint8（0~1 灰阶）")
    print(f"            完全相同(取整后)的像素占比: {(~mismatch).mean() * 100:.3f}%")
    print(f"            不一致的 {mismatch.sum()} 个像素中，真值离整数最远只差 "
          f"{frac_off.max():.2e}"
          f"（{np.mean(frac_off < 1e-9) * 100:.1f}% 正好压在整数上）")
    print(f"            -> 纯属浮点临界值的取舍，数学上 W @ f 和 warp() 是同一件事")

    dense_bytes = (n_out * w * h) * 4
    sparse_bytes = len(vals) * 12
    print(f"\n  稠密存储这个矩阵需要 : {dense_bytes / 1e9:.1f} GB "
          f"（图才缩到 {w} 像素宽就已经这么大）")
    print(f"  稀疏存储（只存非零） : {sparse_bytes / 1e6:.2f} MB")

    # ========== 第 3 部分：外推到你的原图 ==========
    print()
    print("=" * 74)
    print("第 3 部分：换成你的原图（8078 x 4778）呢？")
    print("=" * 74)

    n_src = 8078 * 4778
    print(f"  像素总数 N = {n_src:,}")
    print(f"  完整大矩阵 = N x N = {n_src:,} x {n_src:,}")
    print(f"  稠密 float32 存储 : {n_src ** 2 * 4 / 1e15:.1f} PB  <- 约 6 PB，任何硬盘都装不下")
    print(f"  稀疏存储（4 非零/行）: {4 * n_src * 12 / 1e9:.2f} GB  <- 理论上放得下，但没人会这么干")
    print(f"  而 2x2 矩阵只要      : 4 个浮点数 = 32 字节")
    print()
    print("  结论：那张大矩阵**确实存在**，只是它的信息量极小（99.9999% 是 0），")
    print("        所以我们从不把它写出来，只写那个 2x2 —— 这就是「用小的线性算子")
    print("        描述大的线性变换」。warp() 里的反向映射，等价于在按需读取这些权重。")
    print()
    print("  彩色图的话，大矩阵还要再大 3 倍，而且是块对角形式（三个通道互不混合）：")
    print("      [ W  0  0 ] [ r ]   [ r' ]")
    print("      [ 0  W  0 ] [ g ] = [ g' ]      每个通道独立地用同一个 W 变换")
    print("      [ 0  0  W ] [ b ]   [ b' ]")


if __name__ == "__main__":
    main()
