import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Kernel
from scipy.spatial import ConvexHull

class InverseMultiquadricKernel(Kernel):
    def __init__(self, c=1.0):
        self.c = c

    def __call__(self, X, Y=None, eval_gradient=False):
        if Y is None:
            Y = X
        dists = np.sum((X[:, np.newaxis, :] - Y[np.newaxis, :, :]) ** 2, axis=2)
        K = 1.0 / np.sqrt(dists + self.c**2)
        
        if eval_gradient:
            K_gradient = -0.5 * (dists + self.c**2)**(-1.5)[:, :, np.newaxis]
            return K, K_gradient
        return K

    def diag(self, X):
        return np.full(X.shape[0], 1 / np.sqrt(self.c**2))

    def is_stationary(self):
        return True

class GPISModel:
    def __init__(self, x, y, yaw, laser1, value1,
                 boundary_sample_ratio=0.1, interior_sample_ratio=0.05, 
                 kernel=None, alpha=1e-2, angle_threshold_degrees=16):
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

        self.X_boundary = np.vstack([self.x_wall+self.laser1x_wall, self.y_wall + self.laser1y_wall]).T
        self.y_boundary = np.zeros(len(self.laser1x_wall))
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
        self.gp = GaussianProcessRegressor(kernel=self.kernel, n_restarts_optimizer=10, alpha=self.alpha)
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
    
    def _extract_contour_points(self, X, Y):
        contour = plt.contour(X, Y, self.Z, levels=[0], colors='red')
        contour_lines = contour.collections[0].get_paths()
        return np.concatenate([line.vertices for line in contour_lines])
    
    def apply_penalty(self, c=0.4):
        grid_points = np.vstack([self.contour_points[:, 0], self.contour_points[:, 1]]).T
        grid_sigma = griddata(self.contour_points, self.sigma.ravel(), grid_points, method='linear')
        penalty = self._potential_function(grid_points, self.contour_points, c)
        self.penalized_uncertainty_grid = grid_sigma + penalty
    
    def _potential_function(self, X, significant_points, c):
        penalty = np.zeros(X.shape[0])
        for point in significant_points:
            dist = np.linalg.norm(X - point, axis=1)
            penalty += c / (dist + 1e-8)
        return penalty
    
    def find_max_uncertainty_point(self):
        max_uncertainty_index = np.argmax(self.penalized_uncertainty_grid)
        self.max_uncertainty_point = self.contour_points[max_uncertainty_index]
    
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

