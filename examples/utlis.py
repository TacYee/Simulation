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
    print(f"{action_name}")

def perform_attitude_control(drone, drone_state, controller, yaw, action_name):
    action = controller(drone_state, target_yaw=yaw)
    drone.apply_action(action)
    print(f"{action_name}")

def control_drone(drone, drone_state, depth1_noisy, depth2_noisy, vel_forward, vel_backward, vel_side, rot_z_45, controller, yaw_left, yaw_right, MIN_THRESHOLD, MAX_THRESHOLD, counter= 0):
    R_transpose, current_yaw = process_quaternion(drone_state, rot_z_45)
    
    if MAX_THRESHOLD > depth1_noisy > MIN_THRESHOLD and MAX_THRESHOLD > depth2_noisy > MIN_THRESHOLD:
        CF_world = transform_velocity(vel_side, R_transpose)
        apply_control(drone, drone_state, controller, CF_world, "fly side way")
        counter -= 1
    elif MAX_THRESHOLD > depth1_noisy > MIN_THRESHOLD and depth2_noisy < MIN_THRESHOLD:
        target_yaw = current_yaw + yaw_right
        perform_attitude_control(drone, drone_state, controller, target_yaw, "turn right")
        print(torch.rad2deg(current_yaw))
        print(torch.rad2deg(target_yaw))
    elif MAX_THRESHOLD > depth1_noisy > MIN_THRESHOLD and depth2_noisy > MAX_THRESHOLD:
        target_yaw = current_yaw + yaw_left
        perform_attitude_control(drone, drone_state, controller, target_yaw, "turn left")
        print(torch.rad2deg(current_yaw))
        print(torch.rad2deg(target_yaw))
    elif depth1_noisy < MIN_THRESHOLD and MAX_THRESHOLD > depth2_noisy > MIN_THRESHOLD:
        target_yaw = current_yaw + yaw_left
        perform_attitude_control(drone, drone_state, controller, target_yaw, "turn left")
        print(torch.rad2deg(current_yaw))
        print(torch.rad2deg(target_yaw))
    elif depth1_noisy > MAX_THRESHOLD and MAX_THRESHOLD > depth2_noisy > MIN_THRESHOLD:
        target_yaw = current_yaw + yaw_right
        perform_attitude_control(drone, drone_state, controller, target_yaw, "turn right")
        print(torch.rad2deg(current_yaw))
        print(torch.rad2deg(target_yaw))
    elif depth1_noisy > MAX_THRESHOLD and depth2_noisy > MAX_THRESHOLD:
        forward_world = transform_velocity(vel_forward, R_transpose)
        apply_control(drone, drone_state, controller, forward_world, "fly forward")
    else:
        backward_world = transform_velocity(vel_backward, R_transpose)
        apply_control(drone, drone_state, controller, backward_world, "fly backward")
    
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