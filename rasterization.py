# 2024/12/09 9:54 
# games101 光栅化 学习

import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

if __name__ == '__main__':
    height = 512
    weight = 512
    resolution = np.zeros([height, weight, 3])

    

    point0 = [2,0,-2]
    point1 = [0,2,-2]
    point2 = [0,0,0]

    # points
    pts = [point0, point1, point2]

    # 视口矩阵
    h = height
    w = weight
    viewport = np.array(
        [
            [w / 2, 0, 0, w / 2], 
            [0, h / 2, 0, h / 2], 
            [0, 0, 1, 0], 
            [0, 0, 0, 1]
        ]
    )

    # MVP
    # model transformation
    angle = 0
    angle *= np.pi / 180
    mvp = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0, 0],
            [np.sin(angle), np.cos(angle), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )
    camera_pos = [0, 0, -5]
    view_matrix = np.array(
        [
            [1, 0, 0, -camera_pos[0]],
            [0, 1, 0, -camera_pos[1]],
            [0, 0, 1, -camera_pos[2]],
            [0, 0, 0, 1],
        ]
    )
    mvp = np.dot(view_matrix, mvp)
    fovY = 45
    aspect = 1
    near = 0.1
    far = 50
    fov = fovY
    t2a = np.tan(fov / 2.0)
    proj_matrix = np.array(
        [
            [1 / (aspect * t2a), 0, 0, 0],
            [0, 1 / t2a, 0, 0],
            [0, 0, (near + far) / (near - far), 2 * near * far / (near - far)],
            [0, 0, -1, 0],
        ]
    )
    mvp = np.dot(proj_matrix, mvp)
    
    # 将点坐标齐次化
    pts_2d = []
    for p in pts:
        p = np.array(p + [1]) # 拼接1
        p = np.dot(mvp, p)
        p /= p[3]

        # 
        p = np.dot(viewport, p)[:2]
        pts_2d.append([int(p[0]), int(p[1])])

    vis = 1
    if vis:
        # visualize 3d
        fig = plt.figure()
        pts = np.array(pts)
        x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]

        ax = Axes3D(fig)
        ax.scatter(x, y, z, s=80, marker="^", c="g")
        ax.scatter([camera_pos[0]], [camera_pos[1]], [camera_pos[2]], s=180, marker=7, c="r")
        ax.plot_trisurf(x, y, z, linewidth=0.2, antialiased=True, alpha=0.5)

        # Draw coordinate axes
        ax.quiver(0, 0, 0, 1, 0, 0, color='r', length=1)
        ax.quiver(0, 0, 0, 0, 1, 0, color='g', length=1)
        ax.quiver(0, 0, 0, 0, 0, 1, color='b', length=1)

        # Label coordinate axes
        ax.text(1, 0, 0, 'X', color='r')
        ax.text(0, 1, 0, 'Y', color='g')
        ax.text(0, 0, 1, 'Z', color='b')

        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_zlabel('Z axis')

        # Set the viewing angle for better spatial perception
        ax.view_init(elev=20, azim=30)

        plt.show()

        # visualize 2d
        c = (255, 255, 255)
        for i in range(3):
            for j in range(i + 1, 3):
                cv2.line(resolution, pts_2d[i], pts_2d[j], c, 2)
        cv2.imshow("screen", resolution)
        cv2.waitKey(0)