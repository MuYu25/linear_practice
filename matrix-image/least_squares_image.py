#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
投影与最小二乘：从数据里反求算子（相机色彩标定）。

工程问题
--------
相机拍出来的颜色偏了，你要找一个 3x3 矩阵 M 把它校正回来。但你手上只有
「偏色的图」和「它应该长什么样」，没有那个矩阵本身 —— 这就是最小二乘。

翻译成矩阵语言
--------------
    每个像素给出 3 个方程：   M @ c_i = d_i     (c_i 原始颜色, d_i 目标颜色)
    总方程数 = 3N = 3 x 151 万 = 453 万
    未知数   = 9（M 的 9 个元素）
    => 方程数是未知数的 50 万倍 —— 严重超定，精确解根本不存在（除非数据完美线性）

最小二乘的答案：求使 ||C M^T - D||^2 最小的 M，也就是把 D **投影**到 C 的列空间上。

运行：  python3 least_squares_image.py
"""

from __future__ import annotations

import numpy as np

from main import OUT_DIR, load_working_copy, make_collage

SRC = "/home/m1928a1/Pictures/wallpaper/photo/wallhaven-kxqm26.jpg"


def solve_color_matrix(C: np.ndarray, D: np.ndarray):
    """最小二乘求 M，使 C @ M.T ≈ D。返回 (M, 3x3 的 Gram 矩阵, 3x3 的右端项)。

    法方程： (C^T C) x = C^T D
    展开写就是：  3x3 的矩阵乘 3x3 的未知数 = 3x3 的右端项
    """
    G = C.T @ C            # Gram 矩阵，3x3
    B = C.T @ D            # 右端项，3x3
    Mt = np.linalg.solve(G, B)
    return Mt.T, G, B


def apply_color_matrix(img: np.ndarray, M: np.ndarray, clip: bool = True) -> np.ndarray:
    out = img.reshape(-1, 3).astype(np.float64) @ M.T
    if clip:
        out = np.clip(out, 0, 255)
    return out.reshape(img.shape)


def main() -> None:
    img = load_working_copy(SRC, max_side=1600)
    h, w = img.shape[:2]
    C = img.reshape(-1, 3).astype(np.float64)
    N = C.shape[0]

    # 真值：一个偏暖、有点通道串扰的色彩矩阵（模拟相机偏色）
    M_true = np.array([[1.12, 0.05, -0.04],
                       [0.02, 0.98,  0.03],
                       [-0.03, 0.04, 0.90]])

    print("=" * 76)
    print("第 1 步：造一个工程场景 —— 相机偏色")
    print("=" * 76)
    print(f"  像素数 N = {N:,}   三通道 => 方程数 3N = {3 * N:,}")
    print(f"  未知数（M 的 9 个元素）= 9")
    print(f"  超定倍数 = {3 * N / 9:,.0f} 倍")
    print(f"\n  我们假装不知道真值矩阵，只拿到「原图」和「偏色后的图」：")
    print(f"  M_true（先不告诉你，第 3 步对答案）= ")
    for row in M_true:
        print("      [ " + "  ".join(f"{v: .3f}" for v in row) + " ]")

    # 目标图：不做截断，先让问题严格线性、精确可达
    D = apply_color_matrix(img, M_true, clip=False)

    print()
    print("=" * 76)
    print("第 2 步：最小二乘求解（法方程）")
    print("=" * 76)
    M_hat, G, B = solve_color_matrix(C, D.reshape(-1, 3))
    print(f"  C (原色)      : {C.shape}      <- 453 万个方程")
    print(f"  D (目标色)    : {D.reshape(-1, 3).shape}   <- 每个像素一行三列(r,g,b)")
    print(f"\n  法方程 (C^T C) x = C^T D 里的 C^T C：")
    print(f"    Gram 矩阵 G : {G.shape}      <- 453 万个方程的信息，全部压进这 9 个数")
    for row in G:
        print("      [ " + "  ".join(f"{v:>14,.0f}" for v in row) + " ]")
    print(f"  右端项 C^T D : {B.shape}")
    print(f"\n  工程含义：你不需要把 453 万 x 3 的矩阵传进求解器，")
    print(f"            只需要它的 Gram 矩阵（3x3）和 C^T D（3x3）。这就是「充分统计量」。")

    print()
    print("=" * 76)
    print("第 3 步：对答案")
    print("=" * 76)
    print(f"  恢复出来的 M_hat = ")
    for row in M_hat:
        print("      [ " + "  ".join(f"{v: .6f}" for v in row) + " ]")
    err = np.abs(M_hat - M_true).max()
    print(f"\n  与真值的最大元素误差 = {err:.3e}   <- 机器精度级别，等价于完全恢复")
    print(f"  （目标图严格线性时，9 个未知数被 453 万个方程锁死了）")

    print()
    print("=" * 76)
    print("第 4 步：真实世界没那么干净 —— 截断(clip)是（撞破线性假设）非线性")
    print("=" * 76)
    D_clip = apply_color_matrix(img, M_true, clip=True)      # 相机真的会截断
    M_clip, _, _ = solve_color_matrix(C, D_clip.reshape(-1, 3))
    resid = D_clip.reshape(-1, 3) - C @ M_clip.T              # 每个像素 3 个残差分量
    resid_norm = np.linalg.norm(resid, axis=1)

    print(f"  用「截断后」的目标重建，最大元素误差 = {np.abs(M_clip - M_true).max():.3e}")
    print(f"  残差 ||D - C M^T||:")
    print(f"     最大 {resid_norm.max():.2f}   平均 {resid_norm.mean():.4f}   "
          f"非零像素占比 {(resid_norm > 0.01).mean() * 100:.2f}%")

    # 投影的定义：残差必须与 C 的列空间正交
    # 注意 Gram 矩阵的量级是 4e10，直接打印 0.1 会看着像「非零」，所以要归一化后再看
    orth = C.T @ resid                                       # 3x3，应该全是 0
    scale = np.abs(G).max()
    rel = np.linalg.norm(orth) / (np.linalg.norm(C) * np.linalg.norm(resid))
    print(f"\n  投影的正交性检验  C^T (D - C M^T)，已按 Gram 的量级归一化：")
    for row in orth / scale:
        print("      [ " + "  ".join(f"{v:>12.2e}" for v in row) + " ]")
    print(f"      ||C^T r|| / (||C||·||r||) = {rel:.2e}   <- 机器精度级别的 0")
    print(f"      <- 残差和整个列空间正交 —— 这就是「投影」在 453 万维空间里的含义。")
    print(f"         最小二乘解 = D 在 C 的列空间上的**垂足**，残差就是那段垂线。")

    print()
    print("=" * 76)
    print("第 5 步：条件数 —— 为什么颜色标定在工程上很容易翻车")
    print("=" * 76)
    sv = np.linalg.svd(C, compute_uv=False)
    print(f"  C 的奇异值 : {np.round(sv, 1)}")
    print(f"  κ(C)       = {sv[0] / sv[-1]:,.0f}")
    print(f"  κ(C^T C)   = {np.linalg.cond(G):,.0f}   = κ(C)^2   <- 法方程把病态**平方**了！")
    print(f"\n  原因：这张图的三通道高度相关（整体偏蓝白），第三维信息很弱。")

    D_flat = D.reshape(-1, 3)

    def noise_response(sigma: float, adversarial: bool, seed: int = 0):
        """给目标图加同强度的噪声，比较「随机方向」和「针对最弱方向」两种情形。"""
        dD = np.random.default_rng(seed).normal(0.0, sigma, (N, 3))
        norm = np.linalg.norm(dD)
        if adversarial:
            # 让 C^T dD 正好落在 C^T C 的最小特征方向上 —— 扰动强度一样，方向最毒
            #   dD = C (C^T C)^-1 v_min v_min^T   =>   C^T dD = v_min v_min^T
            v_min = np.linalg.eigh(G)[1][:, 0]
            raw = np.outer(C @ np.linalg.solve(G, v_min), v_min)
            dD = raw * (norm / np.linalg.norm(raw))
        Mn, _, _ = solve_color_matrix(C, D_flat + dD)
        return np.abs(Mn - M_true).max()

    print(f"\n  {'噪声σ':>6}  {'随机噪声的误差':>16}  {'定向噪声的误差':>16}  {'放大倍数':>10}")
    print("  " + "-" * 56)
    for sigma in [0.0, 0.5, 2.0]:
        r = noise_response(sigma, adversarial=False)
        a = noise_response(sigma, adversarial=True)
        ratio = f"{a / r:,.0f}x" if r > 0 else "-"
        print(f"  {sigma:>6.1f}  {r:>16.3e}  {a:>16.3e}  {ratio:>10}")

    print(f"\n  这张表有两层结论，别只看一层：")
    print(f"   * 随机噪声：被 150 万个像素**平均掉**了，误差增长远小于 κ。")
    print(f"     （所以「κ 多大噪声就放大多少倍」是**最坏情况上界**，不是随机噪声的常态）")
    print(f"   * 定向噪声：强度完全相同，只把方向挪到最弱的那一维，误差立刻大几个量级。")
    print(f"   * 真实工程里最危险的是**系统性误差**，它不会被平均掉——")
    print(f"     第 4 步那个 clip 就是系统性误差，平均残差高达 {resid_norm.mean():.2f}。")
    print(f"\n  对策（都是线性代数手段）：")
    print(f"   * 先白平衡/去相关再标定，把 κ(C) 降下来；")
    print(f"   * 用 SVD / 伪逆求解，而不是显式解 法方程（条件数从 κ^2 降回 κ）。")

    # ---- 出图 ----
    restored = np.clip(apply_color_matrix(D_clip, np.linalg.pinv(M_clip)), 0, 255)
    resid_img = np.clip(resid_norm.reshape(h, w) * 8, 0, 255).astype(np.uint8)

    panels = [
        ("原图 Original", img),
        ("目标图：偏色后的图（M_true 未知）", np.clip(D, 0, 255).astype(np.uint8)),
        ("用最小二乘解出的 M 的伪逆校正回来", restored.astype(np.uint8)),
        ("残差图（放大 8 倍，越亮 = 线性模型越失效）",
         np.stack([resid_img] * 3, -1)),
    ]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    make_collage(panels, OUT_DIR / "least_squares_collage.png", cols=2)


if __name__ == "__main__":
    main()
