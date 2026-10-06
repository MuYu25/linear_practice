#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图像压缩的 SVD：丢掉最小的奇异值。

核心问题：做 SVD 压缩，需要把图片看成「高维向量的矩阵」吗？
答案：**不需要**，恰恰相反——SVD 压缩之所以优雅，就是因为你完全不用动它的二维结构。

    一张灰度图本来就是 m x n 的矩阵 A，直接 np.linalg.svd(A) 就行。
    A = U Σ V^T，只保留前 k 个奇异值，就得到秩-k 的近似。

顺带一个漂亮的呼应：SVD 说「任何矩阵 = 旋转 · 拉伸 · 旋转」
    U 和 V 是**正交矩阵**（旋转/反射），Σ 是**对角矩阵**（拉伸）——
    正是前面那个 2x2 旋转/拉伸矩阵的高维版本，只不过这次作用在「颜色值空间」上。

运行：  python3 svd_compress.py
"""

from __future__ import annotations

import numpy as np

from main import OUT_DIR, load_working_copy, make_collage

SRC = "/home/m1928a1/Pictures/wallpaper/photo/wallhaven-kxqm26.jpg"
KS = [5, 10, 30, 60, 120]          # 保留的奇异值个数
WORK_SIDE = 1600


def svd_compress_channel(channel: np.ndarray, k: int):
    """对单个通道做秩-k 截断。channel 是 m x n 的矩阵，直接 SVD，不用拉直。"""
    U, s, Vt = np.linalg.svd(channel, full_matrices=False)   # s 已按从大到小排列
    return (U[:, :k] * s[:k]) @ Vt[:k, :], U[:, :k], s[:k], Vt[:k, :]


def main() -> None:
    img = load_working_copy(SRC, max_side=WORK_SIDE)
    h, w = img.shape[:2]
    print(f"工作图: {w} x {h}（原图 8078 x 4778）")
    print(f"灰度图就是一个 {h} x {w} 的矩阵 A —— 直接对它做 SVD，不需要拉直成向量\n")

    gray = (img.astype(np.float32) @ [0.299, 0.587, 0.114])
    print("=" * 74)
    print("第一步：np.linalg.svd(A)  —— 就这么一行，输入的维度是 2")
    print("=" * 74)
    U, s, Vt = np.linalg.svd(gray, full_matrices=False)
    print(f"  A      : {gray.shape}")
    print(f"  U      : {U.shape}   <- 正交矩阵（列向量两两正交）")
    print(f"  s      : {s.shape}   <- 对角矩阵 Σ 的对角线，已按从大到小排好")
    print(f"  Vt     : {Vt.shape}  <- 正交矩阵")
    print(f"\n  前 8 个奇异值: {np.round(s[:8], 1)}")
    print(f"  最大的 / 最小的 = {s[0]:.1f} / {s[-1]:.4f} = {s[0]/s[-1]:,.0f} 倍")
    print(f"  -> 跨度这么大，说明能量高度集中在前几个奇异值上（自然图像的典型特征）")

    energy = np.cumsum(s ** 2) / np.sum(s ** 2)
    print(f"\n  前 k 个奇异值保留的「能量」(Frobenius 范数平方的占比):")
    for k in [1, 5, 10, 30, 60, 120, 300]:
        print(f"    k = {k:>3}: {energy[k-1] * 100:6.2f}%")

    # ---------------------------------------------------------------
    print()
    print("=" * 74)
    print("第二步：三个通道分别做秩-k 截断，看效果和存储代价")
    print("=" * 74)
    m, n = h, w
    print(f"\n{'k':>5}  {'保留能量':>8}  {'需要存多少个数':>14}  {'相对原图':>9}  "
          f"{'uint8 原始数据':>13}  {'float32 实际':>12}")
    print("-" * 76)
    full_count = m * n
    print(f"{'原图':>5}  {'100.00%':>8}  {full_count:>14,}  {'1.000x':>9}  "
          f"{full_count * 3 / 1e6:>10.1f} MB  {full_count * 3 * 4 / 1e6:>9.1f} MB")
    for k in KS:
        count = k * (m + n + 1)                    # U 的 k 列 + Σ 的 k 个 + V^T 的 k 行
        print(f"{k:>5}  {energy[k-1] * 100:>7.2f}%  {count:>14,}  "
              f"{count / full_count:>8.3f}x  {count * 3 / 1e6:>10.1f} MB  "
              f"{count * 3 * 4 / 1e6:>9.1f} MB")
    print(f"\n  注意最后一列：SVD 的因子是**浮点数**，存起来是 uint8 的 4 倍（float32）。")
    print(f"  报「压缩比」时必须说清是按哪种数据类型算的，否则容易自我感觉良好。")
    print(f"\n  盈亏平衡点: k < m*n/(m+n) = {full_count / (m + n):.0f} 时才开始省空间")

    panels = [("原图 Original", img)]
    for k in KS:
        chans = []
        for c in range(3):
            rec, _, _, _ = svd_compress_channel(img[:, :, c].astype(np.float32), k)
            chans.append(np.clip(rec, 0, 255))
        out = np.stack(chans, axis=2).astype(np.uint8)
        count = k * (m + n + 1)
        panels.append((f"k = {k}   ({energy[k-1]*100:.1f}% 能量, {count/full_count:.3f}x)",
                       out))
        print(f"  [重建] k={k}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    make_collage(panels, OUT_DIR / "svd_collage.png", cols=3)

    print(f"\n{'=' * 74}")
    print("结论")
    print("=" * 74)
    print("  * 不需要高维向量。图像本来就是 m x n 的二维矩阵，SVD 直接吃。")
    print("  * SVD 的三个因子 U · Σ · V^T 就是「旋转 · 拉伸 · 旋转」的高维版：")
    print("      U、V 是正交矩阵（旋转/反射），Σ 是对角矩阵（拉伸）")
    print(f"  * 诚实提醒：这是「数值个数」上的压缩。真要比文件大小，JPEG 利用了人眼")
    print(f"    的感知特性，通常比同观感的 SVD 截断小得多。SVD 的价值主要在")
    print(f"    去噪、特征提取、降维，而不是当编解码器用。")


if __name__ == "__main__":
    main()
