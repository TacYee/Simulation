import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import contourpy
# from sklearn.gaussian_process import GaussianProcessRegressor
# from sklearn.gaussian_process.kernels import Kernel
from scipy.spatial import ConvexHull

class InverseMultiquadricKernel:
    def __init__(self, c=2.0):
        self.c = c

    def __call__(self, X, Y=None):
        if Y is None:
            Y = X
        dists = np.sum((X[:, np.newaxis, :] - Y[np.newaxis, :, :]) ** 2, axis=2)
        K = 1.0 / np.sqrt(dists + self.c**2)
        return K
    
class GaussianProcessRegressor:
    def __init__(self, kernel, alpha=1e-10):
        self.kernel = kernel
        self.alpha = alpha
        self.X_train = None
        self.y_train = None
        self.K_inv = None

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

        # 计算核矩阵 K(X_train, X_train) + αI
        K = self.kernel(X, X) + self.alpha * np.eye(len(X))
        self.K_inv = np.linalg.inv(K)

    def predict(self, X_test, return_std=False):
        # 计算核矩阵 K(X_tessst, X_train)
        K_trans = self.kernel(X_test, self.X_train)

        # 计算均值
        y_mean = K_trans.dot(self.K_inv).dot(self.y_train)

        if return_std:
            # 计算核矩阵 K(X_test, X_test)
            K_test = self.kernel(X_test, X_test)
            # 计算方差
            y_var = K_test - K_trans.dot(self.K_inv).dot(K_trans.T)
            y_std = np.sqrt(np.diag(y_var))
            return y_mean, y_std
        else:
            return y_mean

class GPISModel:
    def __init__(self, x, y, yaw, laser1, laser2, value1, value2,
                boundary_sample_ratio=1, interior_sample_ratio=1, 
                kernel=None, alpha=1e-2, curvature_threshold=-1):
        x = np.array(x)
        y = np.array(y)
        yaw = np.array(yaw)
        laser1 = np.array(laser1)
        laser2 = np.array(laser2)
        value1 = np.array(value1)
        value2 = np.array(value2)

        # **确定墙壁和内部的点**
        self.x_wall = x[(value1 == 1) | (value2 == 1)]
        self.y_wall = y[(value1 == 1) | (value2 == 1)]
        self.x_inside = x[(value1 == -1) | (value2 == -1)]
        self.y_inside = y[(value1 == -1) | (value2 == -1)]

        # **计算 laser 的 X、Y 坐标**
        laser1x = laser1 * np.cos(yaw - 0.7853981)
        laser1y = laser1 * np.sin(yaw - 0.7853981)
        laser2x = laser2 * np.cos(yaw - 0.7853981)
        laser2y = laser2 * np.sin(yaw - 0.7853981)

        # **如果两个值都为 1，取平均**
        laserx = np.where((value1 == 1) & (value2 == 1), (laser1x + laser2x) / 2, 
                        np.where(value1 == 1, laser1x, 
                                np.where(value2 == 1, laser2x, 0)))
        lasery = np.where((value1 == 1) & (value2 == 1), (laser1y + laser2y) / 2, 
                        np.where(value1 == 1, laser1y, 
                                np.where(value2 == 1, laser2y, 0)))

        # **区分墙壁点和内部点**
        self.laser1x_wall = laserx[(value1 == 1) | (value2 == 1)]
        self.laser1y_wall = lasery[(value1 == 1) | (value2 == 1)]
        self.laser1x_inside = laserx[(value1 == -1) | (value2 == -1)]
        self.laser1y_inside = lasery[(value1 == -1) | (value2 == -1)]
        print(value1)

        print(f"x_wall size: {self.x_wall.size}")
        print(f"y_wall size: {self.y_wall.size}")
        print(f"x_inside size: {self.x_inside.size}")
        print(f"y_inside size: {self.y_inside.size}")
        print(f"laser1x_wall size: {self.laser1x_wall.size}")
        print(f"laser1y_wall size: {self.laser1y_wall.size}")

        self.X_boundary = np.vstack([self.x_wall+self.laser1x_wall, self.y_wall + self.laser1y_wall]).T
        self.y_boundary = np.zeros(len(self.x_wall))
        self.X_interior = np.vstack([self.x_inside + self.laser1x_inside, self.y_inside + self.laser1y_inside]).T
        self.y_interior = -np.ones(len(self.x_inside))
        self.boundary_sample_ratio = boundary_sample_ratio
        self.interior_sample_ratio = interior_sample_ratio
        self.kernel = kernel if kernel else InverseMultiquadricKernel(c=2.25)
        self.alpha = alpha
        self.curvature_threshold = curvature_threshold
        
        self.X_train = None
        self.y_train = None
        self.gp = None
        self.Z = None
        self.sigma = None
        self.contour_points = None
        self.penalized_uncertainty_grid = None
        self.contour_sigma_penalized = None
        self.max_uncertainty_point = None
        self.weights = None
        self.curvature = None
    
    def sample_data(self):
        # 下采样边界点
        num_boundary_samples = int(len(self.X_boundary) * self.boundary_sample_ratio)
        boundary_indices = np.random.choice(len(self.X_boundary), num_boundary_samples, replace=False)
        X_boundary_sampled = self.X_boundary[boundary_indices]
        y_boundary_sampled = self.y_boundary[boundary_indices]
        
        # 下采样内部点
        num_interior_samples = int(len(self.X_interior) * self.interior_sample_ratio)
        interior_indices = np.random.choice(len(self.X_interior), num_interior_samples, replace=False)
        X_interior_sampled = self.X_interior[interior_indices]
        y_interior_sampled = self.y_interior[interior_indices]

        # 合并下采样后的数据
        self.X_train = np.vstack([X_boundary_sampled, X_interior_sampled])
        self.y_train = np.concatenate([y_boundary_sampled, y_interior_sampled])
    
    def train_model(self):
        self.gp = GaussianProcessRegressor(kernel=self.kernel, alpha=self.alpha)
        self.gp.fit(self.X_train, self.y_train)
    
    def predict(self):
        x = np.linspace(-4.5, 4.5, 100)
        y = np.linspace(-4.5, 4.5, 100)
        X, Y = np.meshgrid(x, y)
        X_test = np.vstack([X.ravel(), Y.ravel()]).T
        y_pred, sigma = self.gp.predict(X_test, return_std=True)
        self.Z = y_pred.reshape(X.shape)
        self.sigma = sigma.reshape(X.shape)
        line_segments = self._marching_squares(100, self.Z.ravel(), self.sigma.ravel(), -4.5, 9/99, -4.5, 9/99)
        contour_points_all = self._connect_contour_segments(line_segments, len(line_segments))
        x_vals = [point.x for point in contour_points_all]
        y_vals = [point.y for point in contour_points_all]
        self.contour_points = np.column_stack((x_vals, y_vals))
        self.weights = self.gp.K_inv.dot(self.y_train) 
        self.curvature = self._compute_curvature_kernel(self.contour_points, self.weights, self.X_train, self.kernel)
        print(f"curvatures: {self.curvature}")
        grid_points = np.vstack([X.ravel(), Y.ravel()]).T

        contour_sigma_interp = [point.y_std for point in contour_points_all]

        self.significant_points = self._find_high_curvature_clusters_using_curvature(self.contour_points, self.curvature, self.curvature_threshold)
        print(f"significant_points: {self.significant_points}")
        print(f"significant_points shape: {self.significant_points.shape}")
        penalty = self._potential_function(grid_points, self.significant_points, c=0.2)
        penalty_contour = self._potential_function(self.contour_points, self.significant_points, c=0.2)

        original_uncertainty = sigma.ravel()
        penalized_uncertainty = original_uncertainty + penalty

        self.penalized_uncertainty_grid = penalized_uncertainty.reshape(X.shape)
        self.contour_sigma_penalized = contour_sigma_interp + penalty_contour

    class Point:
        def __init__(self, x, y, y_std):
            self.x = x
            self.y = y
            self.y_std = y_std

    class LineSegment:
        def __init__(self, start, end):
            self.start = start
            self.end = end   
    def _save_ordered_contour_point(self, ordered_contour_points, x, y, y_std):
        ordered_contour_points.append(self.Point(x, y, y_std))             
    def _marching_squares(self, grid_size, y_preds, y_stds, x_min, x_step, y_min, y_step):
        line_segments = []
    
        for i in range(grid_size - 1):
            for j in range(grid_size - 1):
                idx_00 = i * grid_size + j      # 左上角
                idx_01 = i * grid_size + (j + 1)  # 右上角
                idx_10 = (i + 1) * grid_size + j  # 左下角
                idx_11 = (i + 1) * grid_size + (j + 1)  # 右下角

                # 确定每个顶点的状态（正负）
                top_left = y_preds[idx_00] > 0
                top_right = y_preds[idx_01] > 0
                bottom_left = y_preds[idx_10] > 0
                bottom_right = y_preds[idx_11] > 0

                # 计算当前网格单元的索引
                cell_index = (top_left << 3) | (top_right << 2) | (bottom_right << 1) | bottom_left

                # 根据 cell_index 的不同值，处理不同的轮廓
                if cell_index == 1 or cell_index == 14:  # 0001 或 1110
                    t1 = abs(y_preds[idx_00]) / (abs(y_preds[idx_00]) + abs(y_preds[idx_10]))
                    x1 = x_min + j * x_step
                    y1 = y_min + i * y_step + t1 * y_step
                    y_std1 = y_stds[idx_00] + t1 * (y_stds[idx_10] - y_stds[idx_00])

                    t2 = abs(y_preds[idx_10]) / (abs(y_preds[idx_10]) + abs(y_preds[idx_11]))
                    x2 = x_min + j * x_step + t2 * x_step
                    y2 = y_min + (i + 1) * y_step
                    y_std2 = y_stds[idx_10] + t2 * (y_stds[idx_11] - y_stds[idx_10])

                    line_segments.append(self.LineSegment(self.Point(x1, y1, y_std1), self.Point(x2, y2, y_std2)))

                elif cell_index == 2 or cell_index == 13:  # 0010 或 1101
                    t1 = abs(y_preds[idx_01]) / (abs(y_preds[idx_01]) + abs(y_preds[idx_11]))
                    x1 = x_min + (j + 1) * x_step
                    y1 = y_min + i * y_step + t1 * y_step
                    y_std1 = y_stds[idx_01] + t1 * (y_stds[idx_11] - y_stds[idx_01])

                    t2 = abs(y_preds[idx_10]) / (abs(y_preds[idx_10]) + abs(y_preds[idx_11]))
                    x2 = x_min + j * x_step + t2 * x_step
                    y2 = y_min + (i + 1) * y_step
                    y_std2 = y_stds[idx_10] + t2 * (y_stds[idx_11] - y_stds[idx_10])

                    line_segments.append(self.LineSegment(self.Point(x1, y1, y_std1), self.Point(x2, y2, y_std2)))

                elif cell_index == 3 or cell_index == 12:  # 0011 或 1100
                    t1 = abs(y_preds[idx_00]) / (abs(y_preds[idx_00]) + abs(y_preds[idx_10]))
                    x1 = x_min + j * x_step
                    y1 = y_min + i * y_step + t1 * y_step
                    y_std1 = y_stds[idx_00] + t1 * (y_stds[idx_10] - y_stds[idx_00])

                    t2 = abs(y_preds[idx_01]) / (abs(y_preds[idx_01]) + abs(y_preds[idx_11]))
                    x2 = x_min + (j + 1) * x_step
                    y2 = y_min + i * y_step + t2 * y_step
                    y_std2 = y_stds[idx_01] + t2 * (y_stds[idx_11] - y_stds[idx_01])

                    line_segments.append(self.LineSegment(self.Point(x1, y1, y_std1), self.Point(x2, y2, y_std2)))

                elif cell_index == 4 or cell_index == 11:  # 0100 或 1011
                    t1 = abs(y_preds[idx_00]) / (abs(y_preds[idx_00]) + abs(y_preds[idx_01]))
                    x1 = x_min + j * x_step + t1 * x_step
                    y1 = y_min + i * y_step
                    y_std1 = y_stds[idx_01] + t1 * (y_stds[idx_01] - y_stds[idx_00])

                    t2 = abs(y_preds[idx_01]) / (abs(y_preds[idx_01]) + abs(y_preds[idx_11]))
                    x2 = x_min + (j + 1) * x_step
                    y2 = y_min + i * y_step + t2 * y_step
                    y_std2 = y_stds[idx_01] + t2 * (y_stds[idx_11] - y_stds[idx_01])

                    line_segments.append(self.LineSegment(self.Point(x1, y1, y_std1), self.Point(x2, y2, y_std2)))

                elif cell_index == 5 or cell_index == 10:
                    # Case for cellIndex == 5 or 10, which doesn't require handling in your original code
                    pass

                elif cell_index == 6 or cell_index == 9:  # 0110 或 1001
                    t1 = abs(y_preds[idx_00]) / (abs(y_preds[idx_00]) + abs(y_preds[idx_01]))
                    x1 = x_min + j * x_step + t1 * x_step
                    y1 = y_min + i * y_step
                    y_std1 = y_stds[idx_00] + t1 * (y_stds[idx_01] - y_stds[idx_00])

                    t2 = abs(y_preds[idx_10]) / (abs(y_preds[idx_11]) + abs(y_preds[idx_10]))
                    x2 = x_min + j * x_step + t2 * x_step
                    y2 = y_min + (i + 1) * y_step
                    y_std2 = y_stds[idx_10] + t2 * (y_stds[idx_11] - y_stds[idx_10])

                    line_segments.append(self.LineSegment(self.Point(x1, y1, y_std1), self.Point(x2, y2, y_std2)))

                elif cell_index == 7 or cell_index == 8:  # 0111 或 1000
                    t1 = abs(y_preds[idx_00]) / (abs(y_preds[idx_01]) + abs(y_preds[idx_00]))
                    x1 = x_min + j * x_step + t1 * x_step
                    y1 = y_min + i * y_step
                    y_std1 = y_stds[idx_00] + t1 * (y_stds[idx_00] - y_stds[idx_10])

                    t2 = abs(y_preds[idx_00]) / (abs(y_preds[idx_00]) + abs(y_preds[idx_10]))
                    x2 = x_min + j * x_step
                    y2 = y_min + i * y_step + t2 * y_step
                    y_std2 = y_stds[idx_00] + t2 * (y_stds[idx_10] - y_stds[idx_00])

                    line_segments.append(self.LineSegment(self.Point(x1, y1, y_std1), self.Point(x2, y2, y_std2)))

                elif cell_index == 0 or cell_index == 15:
                    # 在这种情况下，网格内部没有交点，不需要处理。
                    pass

                else:
                    # 未处理的情况可以加以记录或忽略
                    pass

        return line_segments
    def _connect_contour_segments(self, line_segments, segment_count):
        visited = [False] * segment_count
        ordered_contour_points = []

        # 从第一条未访问的线段出发
        for i in range(segment_count):
            if visited[i]:
                continue

            # 第一个线段保存起点和终点
            self._save_ordered_contour_point(ordered_contour_points, line_segments[i].start.x, line_segments[i].start.y, line_segments[i].start.y_std)
            self._save_ordered_contour_point(ordered_contour_points, line_segments[i].end.x, line_segments[i].end.y, line_segments[i].end.y_std)
            visited[i] = True

            current_end = line_segments[i].end
            found_start = True

            # 查找相邻线段
            while found_start:
                found_start = False
                for j in range(segment_count):
                    if not visited[j]:
                        if (abs(line_segments[j].start.x - current_end.x) < 1e-6 and 
                            abs(line_segments[j].start.y - current_end.y) < 1e-6):
                            # 如果起点与 current_end 相同，则连接，保存终点
                            self._save_ordered_contour_point(ordered_contour_points, line_segments[j].end.x, line_segments[j].end.y, line_segments[j].end.y_std)
                            current_end = line_segments[j].end
                            visited[j] = True
                            found_start = True
                            break
                        elif (abs(line_segments[j].end.x - current_end.x) < 1e-6 and 
                            abs(line_segments[j].end.y - current_end.y) < 1e-6):
                            # 反向连接，保存起点为新的终点
                            self._save_ordered_contour_point(ordered_contour_points, line_segments[j].start.x, line_segments[j].start.y, line_segments[j].start.y_std)
                            current_end = line_segments[j].start
                            visited[j] = True
                            found_start = True
                            break

        return ordered_contour_points   
    def _potential_function(self, x, significant_points, c):
        """计算势函数值"""
        distances = np.linalg.norm(x[:, np.newaxis, :] - significant_points[np.newaxis, :, :], axis=2)
        P = -np.exp(-distances**2 / (2 * c**2))
        return np.min(P, axis=1)
    
    
    def _find_high_curvature_clusters_using_curvature(self, contour_points, curvatures, curvature_threshold=0.5):
        """根据计算出的曲率值找出曲率大于阈值的连续点簇，并处理封闭图形"""
    
        clusters = []
        current_cluster = []
        n = len(contour_points)
    
        for i in range(n):
            # 如果曲率大于设定的阈值，则将当前点添加到当前簇中
            if curvatures[i] < curvature_threshold:
                current_cluster.append(contour_points[i])
            else:
                # 当曲率小于阈值时，若 current_cluster 不为空，则保存它
                if current_cluster:
                    clusters.append(current_cluster)
                # 开始新的簇
                current_cluster = []
    
        # 处理剩余的簇
        if current_cluster:
            clusters.append(current_cluster)
    
        # 检查第一个点和最后一个点是否可以形成连续簇（封闭图形的情形）
        if clusters and len(clusters) > 1:
            if curvatures[0] < curvature_threshold and curvatures[-1] < curvature_threshold:
                # 合并第一个和最后一个簇
                clusters[0] = clusters[-1] + clusters[0]
                clusters.pop(-1)
    
        # 计算每个簇的质心
        significant_points = []
        for cluster in clusters:
            cluster = np.array(cluster)
            centroid = np.mean(cluster, axis=0)
            significant_points.append(centroid)
    
        return np.array(significant_points)

    def _compute_curvature_kernel(self, contour_points, weights, X_train, kernel):
        """
        在核函数上直接计算 GPIS=0 等值线的曲率
        :param X_test: 测试点集
        :param contour_points: GPIS=0 的等值线点
        :param weights: 高斯过程的权重
        :param X_train: 训练点
        :param kernel: 核函数实例
        :return: 等值线点的曲率值
        """
        c2 = kernel.c**2
        curvatures = []

        for x in contour_points:
            # 计算距离和梯度
            diffs = x - X_train
            d2 = np.sum(diffs**2, axis=1)  # ||x - xi||^2
            d2_c2 = d2 + c2

            # 一阶导数 (gradient)
            grad = np.sum(weights[:, None] * (-diffs / d2_c2[:, None]**(3 / 2)), axis=0)

            # 二阶导数 (Hessian)
            hessian = np.zeros((2, 2))
            for i, diff in enumerate(diffs):
                outer = np.outer(diff, diff)
                hessian += weights[i] * (3 * outer / d2_c2[i]**(5 / 2) - np.eye(2) / d2_c2[i]**(3 / 2))

            # 计算曲率公式
            grad_norm = np.linalg.norm(grad)
            if grad_norm == 0:  # 避免除以零
                curvatures.append(0)
                continue
            tr_hessian = np.trace(hessian)
            numerator = grad @ hessian @ grad - grad_norm**2 * tr_hessian
            curvature = numerator / grad_norm**3
            curvatures.append(curvature)

        return np.array(curvatures)
    
    
    def find_max_uncertainty_point(self):
        max_uncertainty_index = np.argmax(self.contour_sigma_penalized)
        self.max_uncertainty_point = self.contour_points[max_uncertainty_index]
        return self.max_uncertainty_point
    
    def plot_results(self, filename=None):
        x = np.linspace(-4.5, 4.5, 100)
        y = np.linspace(-4.5, 4.5, 100)
        X, Y = np.meshgrid(x, y)

        plt.figure(figsize=(14, 6))

        # 左图：GPIS值
        plt.subplot(1, 2, 1)
        plt.contourf(X, Y, self.Z, levels=np.linspace(self.Z.min(), self.Z.max(), 100), cmap="viridis")
        plt.colorbar(label='GPIS Value')
        plt.scatter(self.X_train[:, 0], self.X_train[:, 1], c=self.y_train, cmap="coolwarm", edgecolor="k", s=3)
        plt.scatter(self.max_uncertainty_point[0], self.max_uncertainty_point[1], color='red', s=100, edgecolor='black', label='Max Uncertainty Point')
        plt.contour(X, Y, self.Z, levels=[0], colors='red')
        plt.scatter(self.significant_points[:, 0], self.significant_points[:, 1], c='white', s=30, label='Significant Curvature Points')
        plt.title("2D GPIS with RBF Kernel")
        plt.xlabel("X")
        plt.ylabel("Y")

        # 右图：施加了惩罚后的不确定性
        plt.subplot(1, 2, 2)
        plt.contourf(X, Y, self.penalized_uncertainty_grid, levels=np.linspace(self.penalized_uncertainty_grid.min(), self.penalized_uncertainty_grid.max(), 100), cmap="viridis")
        plt.colorbar(label='Penalized Uncertainty (Std)')
        plt.scatter(self.X_train[:, 0], self.X_train[:, 1], c=self.y_train, cmap="coolwarm", edgecolor="k", s=3)
        plt.title("Uncertainty (Std) with Penalty")
        plt.xlabel("X")
        plt.ylabel("Y")

        plt.tight_layout()

        if filename:
            plt.savefig(filename)
        plt.show()