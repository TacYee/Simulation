import torch
import numpy as np
from scipy.spatial import ConvexHull
def quaternion_multiply(q, r):
        w0, x0, y0, z0 = q.unbind(-1)
        w1, x1, y1, z1 = r.unbind(-1)
        return torch.stack([
            w0*w1 - x0*x1 - y0*y1 - z0*z1,
            w0*x1 + x0*w1 + y0*z1 - z0*y1,
            w0*y1 - x0*z1 + y0*w1 + z0*x1,
            w0*z1 + x0*y1 - y0*x1 + z0*w1
        ], dim=-1)

def quaternion_to_rotation_matrix(quaternion: torch.Tensor) -> torch.Tensor:

    w, x, y, z = torch.unbind(quaternion, dim=-1)
    tx = 2.0 * x
    ty = 2.0 * y
    tz = 2.0 * z
    twx = tx * w
    twy = ty * w
    twz = tz * w
    txx = tx * x
    txy = ty * x
    txz = tz * x
    tyy = ty * y
    tyz = tz * y
    tzz = tz * z

    matrix = torch.stack(
        [
            1 - (tyy + tzz),
            txy - twz,
            txz + twy,
            txy + twz,
            1 - (txx + tzz),
            tyz - twx,
            txz - twy,
            tyz + twx,
            1 - (txx + tyy),
        ],
        dim=-1,
    )
    matrix = matrix.unflatten(matrix.dim() - 1, (3, 3))
    return matrix

def quaternion_to_yaw(rot):
    w, x, y, z = rot[..., 0], rot[..., 1], rot[..., 2], rot[..., 3]
    
    # 计算 yaw
    yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))
    
    return yaw

def process_quaternion(drone_state, rot_z_45):
    _, rot, _, _ = torch.split(drone_state, [3, 4, 3, 3], dim=-1)
    yaw = quaternion_to_yaw(rot)
    rot = torch.nn.functional.normalize(rot, p=2, dim=-1)
    rot = quaternion_multiply(rot, rot_z_45)
    R = quaternion_to_rotation_matrix(rot)
    return R.transpose(-1, -2), yaw

def transform_velocity(velocity, R_transpose):
    if velocity.dim() == 1:
        velocity = velocity.unsqueeze(0)
    return torch.matmul(velocity, R_transpose).squeeze(0)

def apply_control(drone, drone_state, controller, target_vel, action_name):
    action = controller(drone_state, target_vel=target_vel)
    drone.apply_action(action)

def perform_attitude_control(drone, drone_state, controller, yaw, action_name):
    action = controller(drone_state, target_yaw=yaw)
    drone.apply_action(action)

def control_drone(drone, drone_state, depth1_noisy, depth2_noisy, vel_forward, vel_backward, vel_side, rot_z_45, controller, yaw_left, yaw_right, MIN_THRESHOLD, MAX_THRESHOLD, counter= 0):
    R_transpose, current_yaw = process_quaternion(drone_state, rot_z_45)
    
    if MAX_THRESHOLD > depth1_noisy > MIN_THRESHOLD and MAX_THRESHOLD > depth2_noisy > MIN_THRESHOLD:
        CF_world = transform_velocity(vel_side, R_transpose)
        apply_control(drone, drone_state, controller, CF_world, "fly side way")
        counter -= 1
    elif MAX_THRESHOLD > depth1_noisy > MIN_THRESHOLD and depth2_noisy < MIN_THRESHOLD:
        target_yaw = current_yaw + yaw_right
        perform_attitude_control(drone, drone_state, controller, target_yaw, "turn right")
        counter -= 1
    elif MAX_THRESHOLD > depth1_noisy > MIN_THRESHOLD and depth2_noisy > MAX_THRESHOLD:
        target_yaw = current_yaw + yaw_left
        perform_attitude_control(drone, drone_state, controller, target_yaw, "turn left")
        counter -= 1
    elif depth1_noisy < MIN_THRESHOLD and MAX_THRESHOLD > depth2_noisy > MIN_THRESHOLD:
        target_yaw = current_yaw + yaw_left
        perform_attitude_control(drone, drone_state, controller, target_yaw, "turn left")
        counter -= 1
    elif depth1_noisy > MAX_THRESHOLD and MAX_THRESHOLD > depth2_noisy > MIN_THRESHOLD:
        target_yaw = current_yaw + yaw_right
        perform_attitude_control(drone, drone_state, controller, target_yaw, "turn right")
        counter -= 1
    elif depth1_noisy > MAX_THRESHOLD and depth2_noisy > MAX_THRESHOLD:
        forward_world = transform_velocity(vel_forward, R_transpose)
        apply_control(drone, drone_state, controller, forward_world, "fly forward")
    else:
        backward_world = transform_velocity(vel_backward, R_transpose)
        apply_control(drone, drone_state, controller, backward_world, "fly backward")
        counter -= 1
    
    return counter

def compute_angle(p1, p2, p3):
    """计算三个点形成的角度，返回弧度值"""
    v1 = p2 - p1
    v2 = p3 - p2
    cosine_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return angle

def find_high_curvature_points(points, angle_threshold_degrees=10):
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
        
        angle = compute_angle(p1, p2, p3)
        if angle > angle_threshold_radians:
            high_curvature_points.append(p2)
    
    return hull_points, np.array(high_curvature_points)

def potential_function(x, significant_points, c=1.0):
    """计算势函数值"""
    distances = np.linalg.norm(x[:, np.newaxis, :] - significant_points[np.newaxis, :, :], axis=2)
    P = -np.exp(-distances**2 / (2 * c**2))
    return np.min(P, axis=1)

def normalize_angle(rad):
    """ 将角度归一化到 -π 到 π 范围 """
    rad = (rad + np.pi) % (2 * np.pi) - np.pi
    return torch.rad2deg(rad)

from scipy.optimize import minimize
from scipy.interpolate import CubicSpline

import numpy as np
from scipy.optimize import minimize, NonlinearConstraint
from scipy.interpolate import CubicSpline
import torch

class TrajectoryOptimizer:
    def __init__(self, uncertainty_grid, value_grid, 
                 lambda_align=0.0, lambda_align_start=0.0, lambda_smooth=0.0,
                 lambda_u=1.0, lambda_v=0.0,
                 n_control=7, steps=100, find_the_goal=False,
                 v_max=0.0):
        self.uncertainty_grid = uncertainty_grid
        self.value_grid = value_grid
        self.lambda_align = lambda_align
        self.lambda_align_start = lambda_align_start
        self.lambda_smooth = lambda_smooth
        self.lambda_u = lambda_u
        self.lambda_v = lambda_v
        self.n_control = n_control
        self.steps = steps
        self.find_the_goal = find_the_goal
        self.v_max = v_max  # 新增最大允许 value 值作为硬约束

    def bilinear_uncertainty(self, x, grid):
        h, w = grid.shape
        x_img = (x[0] + 5) / 10 * (w - 1)
        y_img = (x[1] + 5) / 10 * (h - 1)
        x_img = np.clip(x_img, 0, w - 2)
        y_img = np.clip(y_img, 0, h - 2)
        x0, x1 = int(np.floor(x_img)), min(int(np.floor(x_img)) + 1, w - 1)
        y0, y1 = int(np.floor(y_img)), min(int(np.floor(y_img)) + 1, h - 1)
        dx, dy = x_img - x0, y_img - y0
        q11, q21 = grid[y0, x0], grid[y0, x1]
        q12, q22 = grid[y1, x0], grid[y1, x1]
        return (q11 * (1 - dx) + q21 * dx) * (1 - dy) + (q12 * (1 - dx) + q22 * dx) * dy

    def compute_trajectory(self, control_points):
        t = np.linspace(0, 1, len(control_points))
        cs_x = CubicSpline(t, control_points[:, 0])
        cs_y = CubicSpline(t, control_points[:, 1])
        t_dense = np.linspace(0, 1, self.steps)
        return np.vstack((cs_x(t_dense), cs_y(t_dense))).T

    def to_numpy(self, x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return x

    def optimize(self, x_start, v_start, x_target, theta, t_o=0.7):
        n_target = np.array([np.cos(theta), np.sin(theta)])
        v_start = np.array([np.cos(self.to_numpy(v_start)), np.sin(self.to_numpy(v_start))])
        control_points = np.linspace(
            self.to_numpy(x_start),
            self.to_numpy(x_target),
            self.n_control
        )
        x0 = control_points[1:-1].flatten()

        def cost_fn(x_opt):
            full_points = np.vstack([x_start, x_opt.reshape(-1, 2), x_target])
            traj = self.compute_trajectory(full_points)

            uncertainties = []
            lengths = []
            for i in range(len(traj) - 1):
                mid = (traj[i] + traj[i + 1]) / 2
                u = self.bilinear_uncertainty(mid, self.uncertainty_grid)
                l = np.linalg.norm(traj[i + 1] - traj[i])
                uncertainties.append(u * l)
                lengths.append(l)

            length_total = np.sum(lengths) + 1e-6
            avg_uncertainty = np.sum(uncertainties) / length_total

            end_dir = traj[-1] - traj[-2]
            end_dir /= np.linalg.norm(end_dir) + 1e-6
            start_dir = traj[1] - traj[0]
            start_dir /= np.linalg.norm(start_dir) + 1e-6
            align_term = 1 - np.dot(end_dir, n_target)
            align_start = 1 - np.dot(start_dir, v_start)

            smoothness_cost = 0
            for i in range(1, len(traj) - 1):
                v1 = traj[i] - traj[i - 1]
                v2 = traj[i + 1] - traj[i]
                if np.linalg.norm(v1) < 1e-6 or np.linalg.norm(v2) < 1e-6:
                    continue
                v1 /= np.linalg.norm(v1)
                v2 /= np.linalg.norm(v2)
                angle_diff = 1 - np.dot(v1, v2)
                smoothness_cost += angle_diff

            cost = (- self.lambda_u * avg_uncertainty +
                    self.lambda_align * align_term +
                    self.lambda_align_start * align_start +
                    self.lambda_smooth * smoothness_cost)
            
            penalty = 0
            if self.find_the_goal == True:
                for p in traj:
                    v = self.bilinear_uncertainty(p, self.value_grid)
                    penalty -= self.lambda_v * v
            cost += penalty/length_total

            return cost

        def constraint_fn(x_opt):
            full_points = x_opt.reshape(-1, 2)
            traj = self.compute_trajectory(full_points)
            values = np.array([self.bilinear_uncertainty(p, self.value_grid) for p in traj])

            if self.find_the_goal:
                # v 必须都 > 0，即 values - epsilon >= 0
                return values - t_o
            else:
                # v 必须都 < 0，即 -(values + epsilon) >= 0
                return -(values - 0.1)

        nonlinear_constraint = NonlinearConstraint(constraint_fn, 0, np.inf)

        bounds = [(-5, 5)] * len(x0)
        result = minimize(cost_fn, x0, method='SLSQP', bounds=bounds,
                          constraints=[nonlinear_constraint],
                          options={'maxiter': 100, 'ftol': 1e-6, 'disp': True})
        print(result)

        optimized_pts = np.vstack([x_start, result.x.reshape(-1, 2), x_target])
        trajectory = self.compute_trajectory(optimized_pts)

        avg_uncertainty = np.mean([self.bilinear_uncertainty(p, self.uncertainty_grid) for p in trajectory])
        print(f"Average Uncertainty Along Trajectory: {avg_uncertainty:.4f}")

        # Downsample
        ds = 0.15
        seg_lengths = np.linalg.norm(np.diff(trajectory, axis=0), axis=1)
        s = np.concatenate([[0], np.cumsum(seg_lengths)])
        total_length = s[-1]
        n_samples = int(np.floor(total_length / ds))
        s_sample = np.linspace(0, total_length, n_samples + 1)

        x_sample = np.interp(s_sample, s, trajectory[:, 0])
        y_sample = np.interp(s_sample, s, trajectory[:, 1])
        trajectory = np.vstack([x_sample, y_sample]).T

        return trajectory

    
    def visualize_trajectory(self, trajectory, x_start, theta_start, x_target, theta_target, arrow_step=10):
        import matplotlib.pyplot as plt
        theta_start = self.to_numpy(theta_start)
        theta_target = self.to_numpy(theta_target)
        v_start = np.array([np.cos(theta_start), np.sin(theta_start)])
        n_target = np.array([np.cos(theta_target), np.sin(theta_target)])
        plt.figure(figsize=(8, 8))
        extent = [-5, 5, -5, 5]  # [xmin, xmax, ymin, ymax]
        plt.imshow(self.uncertainty_grid, cmap='hot', origin='lower', alpha=0.6, extent=extent)
        plt.plot(trajectory[:, 0], trajectory[:, 1], 'b-', linewidth=2, label='Optimized Trajectory')
        plt.scatter(*x_start, c='green', label='Start')
        plt.scatter(*x_target, c='red', label='Target')
        x_start = np.array(x_start).flatten()
        v_start = np.array([np.cos(theta_start), np.sin(theta_start)])
        # 起点方向箭头
            # 起点方向箭头
        plt.arrow(float(x_start[0]), float(x_start[1]),# ✅ 加这个
                float(v_start[0]) * 0.5, float(v_start[1]) * 0.5,
                head_width=0.2, color='green', length_includes_head=True)

        # 终点法向箭头
        plt.arrow(float(x_target[0]), float(x_target[1]),
                float(n_target[0]) * 0.5, float(n_target[1]) * 0.5,
                head_width=0.2, color='red', length_includes_head=True)

        # 在轨迹上绘制朝向箭头
        for i in range(0, len(trajectory) - 1, arrow_step):
            p1 = trajectory[i]
            p2 = trajectory[i + 1]
            dir_vec = p2 - p1
            dir_vec /= np.linalg.norm(dir_vec) + 1e-6
            plt.arrow(float(p1[0]), float(p1[1]),
                    float(dir_vec[0]) * 0.5, float(dir_vec[1]) * 0.5,
                    head_width=0.1, color='blue', alpha=0.7)
        plt.xlim(-5, 5)
        plt.ylim(-5, 5)
        plt.legend()
        plt.grid(True)
        plt.title("Trajectory with Direction Arrows")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.show()
