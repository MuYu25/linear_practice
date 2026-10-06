#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
特征值与特征向量：从一张图里「读出」结构。

前面几轮都是拿矩阵去**改**图片。这一轮反过来：图片里本来就藏着矩阵，
求它们的特征分解，就能把图的结构读出来。

两个真实工程任务：
    第 1 部分  RGB 协方差矩阵（3x3） -> 颜色主成分（PCA / KLT 色彩空间变换）
               特征值 = 每个颜色方向上的方差 = 信息量
               特征向量 = 颜色主轴

    第 2 部分  结构张量（2x2） -> 局部边缘方向场
               特征值 = 该方向上的结构强度
               特征向量 = 边缘的法线方向（梯度方向）
               这是 Harris 角点检测、各向异性扩散去噪的基石

运行：  python3 eigen_image.py
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from main import OUT_DIR, _bilinear, load_working_copy, make_collage, rotation_matrix

SRC = "/home/m1928a1/Pictures/wallpaper/photo/wallhaven-kxqm26.jpg"


def box_blur(a: np.ndarray, r: int) -> np.ndarray:
    """积分图实现的 O(1) 均值滤波（不需要 scipy）。

    这里有个数值坑：积分图是靠「两个大数相减」求窗口和的。如果 a 的整体均值很大
    （比如 gx^2 在亮区能到 1e4），cumsum 会一路累加到 1e7 量级，而平坦区的真实窗口和
    接近 0 —— 相减时发生**灾难性抵消**，误差能到 1e-10，结果出现不该有的负数。
    先减掉全局均值，cumsum 就从「单调长坡」变成「随机游走」，量级大幅下降。
    窗口均值在精确算术下不变，所以最后加回来即可。
    """
    k = 2 * r + 1
    m = a.mean()
    padded = np.pad(a - m, r, mode="edge")
    c = np.pad(padded.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    return (c[k:, k:] - c[:-k, k:] - c[k:, :-k] + c[:-k, :-k]) / (k * k) + m


def rotate_patch(patch: np.ndarray, deg: float) -> np.ndarray:
    """把一个小 patch 绕它自己的中心旋转（复用 main.py 里的双线性采样器）。

    main.warp() 是绕原点旋转的，小 patch 需要绕自己的中心转，
    所以这里自己写平移项：先平移到中心 -> 旋转 -> 平移回去。
    """
    h, w = patch.shape[:2]
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    R = rotation_matrix(deg)
    yy, xx = np.mgrid[0:h, 0:w]
    px, py = xx - cx, yy - cy
    src_x = R[0, 0] * px + R[0, 1] * py + cx
    src_y = R[1, 0] * px + R[1, 1] * py + cy
    rgb = np.repeat(patch[..., None], 3, axis=2)          # _bilinear 是按三通道写的
    return _bilinear(rgb, src_x, src_y, (0, 0, 0))[:, :, 0]


def structure_tensor(gray: np.ndarray, r: int = 4, passes: int = 1):
    """结构张量 J = <[gx², gxgy; gxgy, gy²]> —— 对梯度的外积做窗口平均。

    它是半正定的（对角是平方的均值，且满足 Cauchy-Schwarz |Jxy| <= sqrt(Jxx Jyy)），
    但浮点误差会破坏这个结构，所以这里显式修回来 —— 不修就会出现负特征值，
    各向异性算出 58.9 这种不可能的值（按定义它必须在 0~1）。

    passes > 1 时连做几次均值滤波，相当于一个近似高斯的窗口（比单次方形窗口顺滑得多）。
    """
    gy, gx = np.gradient(gray)

    def smooth(a: np.ndarray) -> np.ndarray:
        for _ in range(passes):
            a = box_blur(a, r)
        return a

    return (np.maximum(smooth(gx * gx), 0.0), smooth(gx * gy),
            np.maximum(smooth(gy * gy), 0.0))


def decompose_tensor(Jxx, Jxy, Jyy):
    """对称 2x2 矩阵的解析特征分解，返回 (λ1, λ2, 主轴方向θ, 各向异性)。

    对 [[a,b],[b,c]]：  λ = (a+c)/2 ± sqrt(((a-c)/2)² + b²)
    主导特征向量的方向： θ = ½·atan2(2b, a-c)
    """
    tr, diff = Jxx + Jyy, Jxx - Jyy
    tmp = np.sqrt(np.maximum(diff * diff / 4.0 + Jxy * Jxy, 0.0))
    tmp = np.minimum(tmp, tr / 2.0)                    # 恢复 Cauchy-Schwarz，保证 λ2 >= 0
    l1, l2 = tr / 2.0 + tmp, tr / 2.0 - tmp
    theta = 0.5 * np.arctan2(2.0 * Jxy, diff)
    # 平坦区的 tr 只有数值噪声量级（~1e-10），那里的「方向」没有意义，判为各向同性
    coherence = np.where(tr > tr.max() * 1e-9,
                         (l1 - l2) / np.maximum(tr, 1e-300), 0.0)
    return l1, l2, theta, coherence


def main() -> None:
    img = load_working_copy(SRC, max_side=1600)
    h, w = img.shape[:2]
    X = img.reshape(-1, 3).astype(np.float64)
    N = X.shape[0]

    # ========================================================================
    print("=" * 76)
    print("第 1 部分：RGB 协方差矩阵的 3 个特征向量 = 这张图的颜色主轴")
    print("=" * 76)
    mu = X.mean(axis=0)
    Xc = X - mu
    Sigma = Xc.T @ Xc / N                     # 3x3 协方差矩阵（对称）
    print(f"  协方差矩阵 Σ ({Sigma.shape}) —— 从 {N:,} 个像素统计出来的")
    for row in Sigma:
        print("      [ " + "  ".join(f"{v:>10,.1f}" for v in row) + " ]")

    evals, evecs = np.linalg.eigh(Sigma)      # 对称矩阵用 eigh；返回升序
    order = np.argsort(evals)[::-1]
    evals, evecs = evals[order], evecs[:, order]

    print(f"\n  特征值（= 每个颜色方向上的方差）: {np.round(evals, 1)}")
    print(f"  占比: {np.round(evals / evals.sum() * 100, 2)} %")
    print(f"\n  特征向量（= 颜色主轴，按方差从大到小）:")
    for i in range(3):
        v = evecs[:, i]
        print(f"      第{i + 1}主方向 = ({v[0]: .3f}, {v[1]: .3f}, {v[2]: .3f})   "
              f"方差 {evals[i]:,.1f}  ({evals[i] / evals.sum() * 100:.2f}%)")
    print(f"\n  读出来的工程结论：")
    print(f"    * 第 1 主方向 ≈ 三通道同号 -> 就是「整体明暗」，占了 {evals[0] / evals.sum() * 100:.2f}% 的方差")
    print(f"    * 最小特征值只有 {evals[2]:,.1f}，说明三通道高度线性相关")
    print(f"    * κ(Σ) = {evals[0] / evals[2]:,.1f} —— 和上一轮标定翻车是同一个根源")

    panels = [("原图 Original", img)]
    for k in (1, 2):
        proj = Xc @ evecs[:, :k]                       # 投影到前 k 个主方向
        rec = proj @ evecs[:, :k].T + mu               # 再投影回来
        captured = evals[:k].sum() / evals.sum() * 100
        panels.append((f"只用前 {k} 个颜色主轴重建 ({captured:.2f}% 方差)",
                       np.clip(rec, 0, 255).reshape(h, w, 3).astype(np.uint8)))
        print(f"    [出图] 保留前 {k} 个主方向 -> 覆盖 {captured:.2f}% 的方差")

    # ========================================================================
    print()
    print("=" * 76)
    print("第 2 部分：结构张量的特征分解 -> 局部边缘方向")
    print("=" * 76)
    gray = img.astype(np.float64) @ [0.299, 0.587, 0.114]
    # 方向场：先轻平滑去掉噪点，再用大窗口（16 半径做两次 ≈ 高斯）统计方向，
    # 否则逐像素估计出来的方向场全是噪声，看不出结构
    Jxx, Jxy, Jyy = structure_tensor(box_blur(gray, 2), r=16, passes=2)
    l1, l2, theta, coherence = decompose_tensor(Jxx, Jxy, Jyy)

    print(f"  梯度 -> 结构张量 J，逐像素做 2x2 特征分解（解析公式，向量化）")
    print(f"  大特征值 λ1 : 最大 {l1.max():,.1f}   中位数 {np.median(l1):,.1f}")
    print(f"  各向异性 (λ1-λ2)/(λ1+λ2) : 平均 {coherence.mean():.3f}   最大 {coherence.max():.3f}")
    print(f"\n  两个特征值直接就是分类器：")
    print(f"    λ1 ≈ λ2 ≈ 0   ->  平坦区域（天空、墙面）")
    print(f"    λ1 >> λ2      ->  有明确方向的边缘")
    print(f"    λ1 ≈ λ2 >> 0  ->  角点（两个方向都有结构，Harris 就是看这个）")

    # ---- 用特征向量构造旋转矩阵，把局部坐标系转到特征向量方向上 ----
    score = coherence * (l1 > np.percentile(l1, 99.5))     # 挑「干净且强」的边缘
    ey, ex = np.unravel_index(np.argmax(score), score.shape)
    half, patch_r = 16, 4                          # patch 只有 32x32，窗口要小
    patch = gray[ey - half:ey + half, ex - half:ex + half]

    # 关键：用 **patch 自己** 的特征向量去转（尺度要一致，否则比较没有意义），
    # 只看中心那个像素（转过的 patch 四周会采到图像外的补偿值，边缘不可信）
    def center_eig(p):
        A, B, C = structure_tensor(p, patch_r)
        _, _, th_p, coh_p = decompose_tensor(A, B, C)
        return np.rad2deg(th_p[half, half]), B[half, half], coh_p[half, half]

    th_before, jxy_before, coh_before = center_eig(patch)
    aligned = rotate_patch(patch, th_before)
    th_after, jxy_after, coh_after = center_eig(aligned)

    print(f"\n  拿一个干净边缘做实验：(x={ex}, y={ey})")
    print(f"  特征向量给出 patch 的主方向 θ = {th_before:+.2f}°"
          f"   （注意：这是**梯度**方向 = 边缘的法线）")
    print(f"\n  {'':>16} {'主轴方向 θ':>12} {'交叉项 Jxy':>12} {'各向异性':>10}")
    print("  " + "-" * 54)
    print(f"  {'原始 patch':>16} {th_before:>11.2f}° {jxy_before:>12.2f} {coh_before:>10.4f}")
    print(f"  {f'转过 {th_before:+.2f}°':>16} {th_after:>11.2f}° {jxy_after:>12.2f} {coh_after:>10.4f}")
    print(f"\n  主轴方向 {th_before:+.2f}° -> {th_after:+.2f}°，交叉项 |Jxy| 降了 "
          f"{100 * (1 - abs(jxy_after) / abs(jxy_before)):.1f}% -> 张量变成了**对角矩阵**")
    print(f"\n  这正是特征向量的定义：在它自己的方向上，矩阵没有交叉项（对角化）。")
    print(f"  工程套路：**用特征向量找出重要方向，转到那个方向去处理，再转回来**。")
    print(f"  （局部坐标系的那个旋转矩阵，就是你第一轮写的 R(θ)）")

    # ---- 方向场可视化：色相 = 方向, 饱和度 = 各向异性 ----
    hue = ((theta + np.pi / 2) / np.pi * 179).astype(np.uint8)
    sat = np.clip(coherence * 255, 0, 255).astype(np.uint8)
    val = np.full_like(hue, 255)
    orientation_rgb = np.array(
        Image.fromarray(np.stack([hue, sat, val], -1), "HSV").convert("RGB"))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    panels.append(("结构张量方向场：色相=边缘方向，浓淡=结构强度", orientation_rgb))
    make_collage(panels, OUT_DIR / "eigen_collage.png", cols=2)


if __name__ == "__main__":
    main()
