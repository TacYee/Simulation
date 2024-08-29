import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import contourpy
# from sklearn.gaussian_process import GaussianProcessRegressor
# from sklearn.gaussian_process.kernels import Kernel
from scipy.spatial import ConvexHull

class InverseMultiquadricKernel:
    def __init__(self, c=1.0):
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
        # 计算核矩阵 K(X_test, X_train)
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
    def __init__(self, x, y, yaw, laser1, value1,
                 boundary_sample_ratio=0.1, interior_sample_ratio=0.05, 
                 kernel=None, alpha=1e-2, angle_threshold_degrees=16):
        x = np.array(x)
        y = np.array(y)
        yaw = np.array(yaw)
        laser1 = np.array(laser1)
        value1 = np.array(value1)
        self.x_wall = x[value1 == 1]
        self.y_wall = y[value1 == 1]
        self.x_inside = x[value1 == -1]
        self.y_inside = y[value1 == -1]
        self.laser1x=laser1*np.cos(yaw-0.7853981)
        self.laser1y=laser1*np.sin(yaw-0.7853981)
        self.laser1x_wall=self.laser1x[value1 == 1]
        self.laser1y_wall=self.laser1y[value1 == 1]
        self.laser1x_inside=self.laser1x[value1 == -1]
        self.laser1y_inside=self.laser1y[value1 == -1]
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
        self.y_interior = np.ones(len(self.x_inside))
        self.boundary_sample_ratio = boundary_sample_ratio
        self.interior_sample_ratio = interior_sample_ratio
        self.kernel = kernel if kernel else InverseMultiquadricKernel(c=2)
        self.alpha = alpha
        self.angle_threshold_degrees = angle_threshold_degrees
        
        self.X_train = None
        self.y_train = None
        self.gp = None
        self.Z = None
        self.sigma = None
        self.contour_points = None
        self.penalized_uncertainty_grid = None
        self.contour_sigma_penalized = None
        self.max_uncertainty_point = None
    
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
        x = np.linspace(-3, 3, 20)
        y = np.linspace(-3, 3, 20)
        X, Y = np.meshgrid(x, y)
        X_test = np.vstack([X.ravel(), Y.ravel()]).T
        y_pred, sigma = self.gp.predict(X_test, return_std=True)
        self.Z = y_pred.reshape(X.shape)
        self.sigma = sigma.reshape(X.shape)
        self.contour_points = self._extract_contour_points(X, Y)

        grid_points = np.vstack([X.ravel(), Y.ravel()]).T

        contour_sigma_interp = griddata(X_test, sigma.ravel(), self.contour_points, method='linear')

        hull_points, self.significant_points = self._find_high_curvature_points(self.contour_points, angle_threshold_degrees=self.angle_threshold_degrees)

        penalty = self._potential_function(grid_points, self.significant_points, c=0.4)
        penalty_contour = self._potential_function(self.contour_points, self.significant_points, c=0.4)

        original_uncertainty = sigma.ravel()
        penalized_uncertainty = original_uncertainty + penalty

        self.penalized_uncertainty_grid = penalized_uncertainty.reshape(X.shape)
        self.contour_sigma_penalized = contour_sigma_interp + penalty_contour
    
    def _extract_contour_points(self, X, Y):
        contour = plt.contour(X, Y, self.Z, levels=[0], colors='red')
        contour_lines = contour.collections[0].get_paths()
        return np.concatenate([line.vertices for line in contour_lines])
    
    def _potential_function(self, x, significant_points, c):
        """计算势函数值"""
        distances = np.linalg.norm(x[:, np.newaxis, :] - significant_points[np.newaxis, :, :], axis=2)
        P = -np.exp(-distances**2 / (2 * c**2))
        return np.min(P, axis=1)
    
    def _find_high_curvature_points(self, points, angle_threshold_degrees=10):
        """找出曲率大于指定角度的点"""
        # 计算凸包
        hull = ConvexHull(points)
        hull_points = points[hull.vertices]
    
        # 转换角度阈值为弧度
        angle_threshold_radians = np.deg2rad(angle_threshold_degrees)
    
        # 计算每个点的曲率（角度）并筛选出曲率大于阈值的点
        high_curvature_points = []
        for i in range(len(hull_points)):
            p1 = hull_points[i - 1]
            p2 = hull_points[i]
            p3 = hull_points[(i + 1) % len(hull_points)]
        
            angle = self._compute_angle(p1, p2, p3)
            if angle > angle_threshold_radians:
                high_curvature_points.append(p2)
    
        return hull_points, np.array(high_curvature_points)
    
    def _compute_angle(self, p1, p2, p3):
        """计算三个点形成的角度，返回弧度值"""
        v1 = p2 - p1
        v2 = p3 - p2
        cosine_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        return angle
    
    def find_max_uncertainty_point(self):
        max_uncertainty_index = np.argmax(self.contour_sigma_penalized)
        self.max_uncertainty_point = self.contour_points[max_uncertainty_index]
        return self.max_uncertainty_point
    
    def plot_results(self, filename=None):
        x = np.linspace(-3, 3, 20)
        y = np.linspace(-3, 3, 20)
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

