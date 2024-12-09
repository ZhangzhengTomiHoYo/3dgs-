# 非常的复杂

import math
import numpy as np
from tqdm import tqdm
from loguru import logger
from math import sqrt, ceil
import matplotlib.pyplot as plt

"""e.g.
input:
        p = (1, 2, 3)
        matrix = [
            [1, 0, 0, 1],
            [0, 1, 0, 2],
            [0, 0, 1, 3],
            [0, 0, 0, 1]
        ]
output:
        array([2, 4, 6, 1])

实际上就是一个 1x3 的矩阵乘以一个 4x4 的矩阵
"""
def transformPoint4x4(p, matrix):
    matrix = np.array(matrix).flatten(order="F")
    x, y, z = p
    transformed = np.array(
        [
            matrix[0] * x + matrix[4] * y + matrix[8] * z + matrix[12],
            matrix[1] * x + matrix[5] * y + matrix[9] * z + matrix[13],
            matrix[2] * x + matrix[6] * y + matrix[10] * z + matrix[14],
            matrix[3] * x + matrix[7] * y + matrix[11] * z + matrix[15],
        ]
    )
    return transformed


"""
这个函数 ndc2Pix 将标准设备坐标 (NDC) 转换为像素坐标。函数接受两个参数：v 和 S。

v 是一个标准设备坐标值，通常在 -1 到 1 之间。标准设备坐标是一种归一化的坐标系，常用于计算机图形学中，以便将坐标值限制在一个固定范围内。
S 是屏幕尺寸或分辨率，表示屏幕的宽度或高度，以像素为单位。
函数的计算过程如下：

首先，将 v 加 1.0，使其范围从 [-1, 1] 转换为 [0, 2]。
然后，将结果乘以 S，将归一化的坐标转换为屏幕尺寸范围内的值。
接着，减去 1.0，将范围调整为 [-1, S-1]。
最后，乘以 0.5，将范围缩小到 [0, (S-1)/2]。
这个转换过程的目的是将标准设备坐标转换为像素坐标，以便在屏幕上正确显示图形。通过这种转换，可以将归一化的坐标值映射到实际的屏幕像素位置。
"""
def ndc2Pix(v, S):
    return ((v + 1.0) * S - 1.0) * 0.5


"""
mean 世界坐标系下的点坐标
cov3D 世界坐标系下的协方差矩阵
viewmatrix 世界坐标系到相机坐标系的变换矩阵
"""
def transformPoint4x3(p, matrix):
    matrix = np.array(matrix).flatten(order="F")
    x, y, z = p
    transformed = np.array(
        [
            matrix[0] * x + matrix[4] * y + matrix[8] * z + matrix[12],
            matrix[1] * x + matrix[5] * y + matrix[9] * z + matrix[13],
            matrix[2] * x + matrix[6] * y + matrix[10] * z + matrix[14],
        ]
    )
    return transformed
def computeCov2D(mean, focal_x, focal_y, tan_fovx, tan_fovy, cov3D, viewmatrix):
    # The following models the steps outlined by equations 29
    # and 31 in "EWA Splatting" (Zwicker et al., 2002).
    # Additionally considers aspect / scaling of viewport.
    # Transposes used to account for row-/column-major conventions.

    t = transformPoint4x3(mean, viewmatrix) # 3dgs中心点附件 线性变换 一个雅可比矩阵，所以需要这个点在相机坐标系下的坐标

    limx = 1.3 * tan_fovx
    limy = 1.3 * tan_fovy
    txtz = t[0] / t[2]
    tytz = t[1] / t[2]
    t[0] = min(limx, max(-limx, txtz)) * t[2]
    t[1] = min(limy, max(-limy, tytz)) * t[2]

    J = np.array(
        [
            [focal_x / t[2], 0, -(focal_x * t[0]) / (t[2] * t[2])],
            [0, focal_y / t[2], -(focal_y * t[1]) / (t[2] * t[2])],
            [0, 0, 0], # 这地方不管
        ]
    )
    W = viewmatrix[:3, :3] # 观测矩阵/视图矩阵的前三行
    T = np.dot(J, W) 

    cov = np.dot(T, cov3D)
    cov = np.dot(cov, T.T) # ！！！2维，所以 是 2x2 的协方差矩阵 

    # Apply low-pass filter
    # Every Gaussia should be at least one pixel wide/high
    # Discard 3rd row and column
    cov[0, 0] += 0.3
    cov[1, 1] += 0.3
    return [cov[0, 0], cov[0, 1], cov[1, 1]] # 2x2 的矩阵 对角 所以有效值3个

def getWorld2View2(R, t, translate=np.array([0.0, 0.0, 0.0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)

def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = np.zeros((4, 4))

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

def in_frustum(p_orig, viewmatrix):
    # bring point to screen space
    p_view = transformPoint4x3(p_orig, viewmatrix)

    if p_view[2] <= 0.2:
        return None
    return p_view

# covariance = RS[S^T][R^T]
def computeCov3D(scale, mod, rot):
    # 创建缩放矩阵
    S = np.array(
        [[scale[0] * mod, 0, 0], [0, scale[1] * mod, 0], [0, 0, scale[2] * mod]]
    )

    # 归一化四元数以获得有效的旋转
    # 我们使用旋转矩阵
    R = rot

    # 计算3D世界协方差矩阵Sigma
    M = np.dot(R, S)
    cov3D = np.dot(M, M.T)

    return cov3D

SH_C0 = 0.28209479177387814
SH_C1 = 0.4886025119029199
SH_C2 = [
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396,
]
SH_C3 = [
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435,
]


def computeColorFromSH(deg, pos, campos, sh):
    # The implementation is loosely based on code for
    # "Differentiable Point-Based Radiance Fields for
    # Efficient View Synthesis" by Zhang et al. (2022)

    dir = pos - campos
    dir = dir / np.linalg.norm(dir)

    result = SH_C0 * sh[0]

    if deg > 0:
        x, y, z = dir
        result = result - SH_C1 * y * sh[1] + SH_C1 * z * sh[2] - SH_C1 * x * sh[3]

        if deg > 1:
            xx = x * x
            yy = y * y
            zz = z * z
            xy = x * y
            yz = y * z
            xz = x * z
            result = (
                result
                + SH_C2[0] * xy * sh[4]
                + SH_C2[1] * yz * sh[5]
                + SH_C2[2] * (2.0 * zz - xx - yy) * sh[6]
                + SH_C2[3] * xz * sh[7]
                + SH_C2[4] * (xx - yy) * sh[8]
            )

            if deg > 2:
                result = (
                    result
                    + SH_C3[0] * y * (3.0 * xx - yy) * sh[9]
                    + SH_C3[1] * xy * z * sh[10]
                    + SH_C3[2] * y * (4.0 * zz - xx - yy) * sh[11]
                    + SH_C3[3] * z * (2.0 * zz - 3.0 * xx - 3.0 * yy) * sh[12]
                    + SH_C3[4] * x * (4.0 * zz - xx - yy) * sh[13]
                    + SH_C3[5] * z * (xx - yy) * sh[14]
                    + SH_C3[6] * x * (xx - 3.0 * yy) * sh[15]
                )
    result += 0.5
    return np.clip(result, a_min=0, a_max=1)

if __name__ == '__main__':
    pts = np.array([[2, 0, -2], [0, 2, -2], [-2, 0, -2]])
    n = len(pts)
    shs = np.random.random((n, 16, 3))
    opacities = np.ones((n, 1))
    scales = np.ones((n, 3))
    rotations = np.array([np.eye(3)] * n) # np.eye(3) 创建了一个 3x3 的单位矩阵

    cam_pos = np.array([0, 0, 5])
    R = np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]])
    proj_param = {"znear": 0.01, "zfar": 100, "fovX": 45, "fovY": 45}
    viewmatrix = getWorld2View2(R=R, t=cam_pos)
    projmatrix = getProjectionMatrix(**proj_param)
    # MVP矩阵
    projmatrix = np.dot(projmatrix, viewmatrix)
    tanfovx = math.tan(proj_param["fovX"] * 0.5)
    tanfovy = math.tan(proj_param["fovY"] * 0.5)

    # 原来的 rasterizer.forward
    P = len(pts)
    D = 3
    M = 16
    background = np.array([0, 0, 0])
    width = 700
    height = 700
    means3D = pts
    # shs
    colors_precomp = None
    # opacities
    # scales
    scale_modifier = 1
    # rotations
    cov3d_precomp = None
    # viewmatrix
    # projmatrix
    # cam_pos
    tan_fovx = tanfovx
    tan_fovy = tanfovy
    prefiltered = None

    focal_y = height / (2 * tan_fovy)  # focal of y axis
    focal_x = width / (2 * tan_fovx)

    logger.info("Starting preprocess per 3d gaussian...")

    """原来的self.preprocess"""
    # P
    # D
    # M
    orig_points = means3D
    # scales
    # scale_modifier
    # rotations
    # opacities
    # shs
    # viewmatrix
    # projmatrix
    # cam_pos
    W = width
    H = height
    # focal_x
    # focal_y
    # tan_fovx
    # tan_fovy

    rgbs = []  # 高斯的 RGB 颜色
    cov3Ds = []  # 3D 高斯的协方差
    depths = []  # 视图和投影变换后的 3D 高斯深度
    radii = []  # 2D 高斯的半径
    conic_opacity = []  # 2D 高斯的协方差逆矩阵和不透明度
    points_xy_image = []  # 2D 高斯的均值

    for idx in range(P):
        # 确保点在视锥体内
        p_orig = orig_points[idx]
        p_view = in_frustum(p_orig, viewmatrix)
        if p_view is None:
            continue
        depths.append(p_view[2])

        # 变换点，从世界坐标系到 NDC
        # 注意，projmatrix 已经处理为 MVP 矩阵
        p_hom = transformPoint4x4(p_orig, projmatrix)
        p_w = 1 / (p_hom[3] + 0.0000001)
        p_proj = [p_hom[0] * p_w, p_hom[1] * p_w, p_hom[2] * p_w]

        # 通过缩放和旋转参数计算 3D 协方差
        scale = scales[idx]
        rotation = rotations[idx]
        cov3D = computeCov3D(scale, scale_modifier, rotation)
        cov3Ds.append(cov3D)

        # 计算二维屏幕空间协方差矩阵
        # 基于splatting方法，-> JW Sigma W^T J^T
        cov = computeCov2D(
            p_orig, focal_x, focal_y, tan_fovx, tan_fovy, cov3D, viewmatrix
        )
        
        # 反转协方差矩阵（EWA splatting）
        det = cov[0] * cov[2] - cov[1] * cov[1]
        if det == 0:
            depths.pop()
            cov3Ds.pop()
            continue
        det_inv = 1 / det
        conic = [cov[2] * det_inv, -cov[1] * det_inv, cov[0] * det_inv]
        conic_opacity.append([conic[0], conic[1], conic[2], opacities[idx]])
        
        # 计算半径，通过找到二维协方差的特征值
        # 将点从NDC转换为像素
        mid = 0.5 * (cov[0] + cov[1])
        lambda1 = mid + sqrt(max(0.1, mid * mid - det))
        lambda2 = mid - sqrt(max(0.1, mid * mid - det))
        my_radius = ceil(3 * sqrt(max(lambda1, lambda2)))
        point_image = [ndc2Pix(p_proj[0], W), ndc2Pix(p_proj[1], H)]
        
        radii.append(my_radius)
        points_xy_image.append(point_image)
        
        # 将球谐系数转换为RGB颜色
        sh = shs[idx]
        result = computeColorFromSH(D, p_orig, cam_pos, sh)
        rgbs.append(result)

    preprocessed = dict(
        rgbs=rgbs,
        cov3Ds=cov3Ds,
        depths=depths,
        radii=radii,
        conic_opacity=conic_opacity,
        points_xy_image=points_xy_image,
    )

    # 生成 [depth] 键和相应的高斯索引
    # 按深度对索引进行排序
    depths = preprocessed["depths"]
    point_list = np.argsort(depths)

    # render
    logger.info("Starting render...")


    """原来的self.render"""
    # point_list
    W = width
    H = height
    points_xy_image = preprocessed["points_xy_image"]
    features = preprocessed["rgbs"]
    conic_opacity = preprocessed["conic_opacity"]
    bg_color = background

    out_color = np.zeros((H, W, 3))
    pbar = tqdm(range(H * W))

    # 遍历像素
    for i in range(H):
        for j in range(W):
            pbar.update(1)
            pixf = [i, j]
            C = [0, 0, 0]

            # 遍历高斯
            for idx in point_list:

                # 初始化辅助变量，透射率
                T = 1

                # 使用圆锥矩阵重新采样
                # (参考 "Surface Splatting" by Zwicker et al., 2001)
                xy = points_xy_image[idx]  # 2D 高斯的中心
                d = [
                xy[0] - pixf[0],
                xy[1] - pixf[1],
                ]  # 到像素中心的距离
                con_o = conic_opacity[idx]
                power = (
                -0.5 * (con_o[0] * d[0] * d[0] + con_o[2] * d[1] * d[1])
                - con_o[1] * d[0] * d[1]
                )
                if power > 0:
                    continue

                # 3D Gaussian splatting 论文中的公式 (2)。
                # 计算颜色
                alpha = min(0.99, con_o[3] * np.exp(power))
                if alpha < 1 / 255:
                    continue
                test_T = T * (1 - alpha)
                if test_T < 0.0001:
                    break

                # 3D Gaussian splatting 论文中的公式 (3)。
                color = features[idx]
                for ch in range(3):
                    C[ch] += color[ch] * alpha * T

                T = test_T

            # 获取最终颜色
            for ch in range(3):
                out_color[j, i, ch] = C[ch] + T * bg_color[ch]

    plt.imshow(out_color)
    plt.show()
