import torch
import pandas as pd
import hydra
from omegaconf import OmegaConf
from omni_drones import CONFIG_PATH, init_simulation_app
import omni
import numpy as np
from utlis import *
from GPIS import GPISModel
import matplotlib.pyplot as plt
from tqdm import tqdm
from enum import Enum, auto
from omni_drones.controllers import LeePositionController

class DroneState(Enum):
    """无人机状态枚举"""
    IDLE = auto()
    FIND_GOAL = auto()
    CF_ACTION = auto()
    BACKWARD = auto()
    CHANGE_DIRECTION = auto()
    FOLLOW_TRAJECTORY = auto()
    EXIT = auto()
    LAND = auto()

class DroneFSM:
    def __init__(self, cfg, sim, drone, lidarInterface, lidarPath1, lidarPath2, end_point=None):
        self.cfg = cfg
        self.sim = sim
        self.drone = drone
        self.lidarInterface = lidarInterface
        self.lidarPath1 = lidarPath1
        self.lidarPath2 = lidarPath2
        
        # 初始化状态
        self.current_state = DroneState.IDLE
        self.prev_state = None
        
        # 控制器
        self.controller = LeePositionController(g=9.81, uav_params=drone.params).to(sim.device)
        
        # 状态变量
        self.state_vars = {
            'goal_counter': 0,
            'CF_action_counter': 0,
            'backward_action_counter': 0,
            'direction_change_counter': 0,
            'direction_changes_completed': 0,
            'Forward_counter': 0,
            'traj_index': 9999999,
            'finish_CF': False,
            'outside': False,
            'back': False,
            'random_yaw': None,
            'before_yaw': None,
            'target_yaw': None,
            'exit_point': None,
            'next_point': None,
            'end_point': end_point,
            'trajectory': [],
            'laser_value1': 0,
            'laser_value2': 0,
            'depth_last': 0.5,
            'depth_now': 0.5,
            'current_yaw': 0,
            'state_x': 0,
            'state_y': 0,
            'state_z': 0,
            'last_trajectory': False
        }
        
        # 数据记录
        self.data_records = {
            'state_xs': [],
            'state_ys': [],
            'state_yaws': [],
            'state_lasers1': [],
            'state_lasers2': [],
            'laser_values1': [],
            'laser_values2': []
        }

        self.data_records_temp = {
            'state_xs': [],
            'state_ys': [],
            'state_yaws': [],
            'state_lasers1': [],
            'state_lasers2': [],
            'laser_values1': [],
            'laser_values2': []
        }
        
        # 常量
        self.MIN_THRESHOLD = 0.4
        self.MAX_THRESHOLD = 0.45
        self.ROOM_X_MIN, self.ROOM_X_MAX = -2.8 * 1.33, 2.8 * 1.33
        self.ROOM_Y_MIN, self.ROOM_Y_MAX = -2.8 * 1.33, 2.8 * 1.33
        
        # 速度向量
        self.vel_forward = torch.tensor([0.2, 0.0, 0.0], device=sim.device)
        self.vel_side = torch.tensor([0, -0.2, 0.0], device=sim.device)
        self.vel_backward = torch.tensor([-0.2, 0.0, 0.0], device=sim.device)
        self.cf_vel_forward = torch.tensor([0.05, 0, 0], device=sim.device)
        self.cf_vel_backward = torch.tensor([-0.05, 0, 0], device=sim.device)
        
        # 偏航控制
        yaw_right_rad = np.deg2rad(-25)
        yaw_left_rad = np.deg2rad(25)
        self.yaw_right = torch.tensor([yaw_right_rad], device=sim.device)
        self.yaw_left = torch.tensor([yaw_left_rad], device=sim.device)
        
        # 旋转四元数
        theta = -45.0  # degrees
        theta_rad = np.radians(theta)
        cos_half_theta = np.cos(theta_rad / 2)
        sin_half_theta = np.sin(theta_rad / 2)
        self.rot_z_45 = torch.tensor([cos_half_theta, 0.0, 0.0, sin_half_theta], device=sim.device)
    
    def transition_to(self, new_state):
        """状态转换"""
        print(f"状态转换: {self.current_state} -> {new_state}")
        print(f"当前状态变量: CF_counter={self.state_vars['CF_action_counter']}, "
            f"Backward_counter={self.state_vars['backward_action_counter']}, "
            f"Dir_changes={self.state_vars['direction_changes_completed']}")
        self.prev_state = self.current_state
        self.current_state = new_state
    
    def get_depth_data(self):
        """获取带噪声的深度数据"""
        depth1 = self.lidarInterface.get_linear_depth_data("/World" + self.lidarPath1)
        depth2 = self.lidarInterface.get_linear_depth_data("/World" + self.lidarPath2)
        
        # 添加高斯噪声
        noise1 = np.random.normal(0.0, 0.01, depth1.shape)
        noise2 = np.random.normal(0.0, 0.01, depth2.shape)
        
        return depth1 + noise1, depth2 + noise2
    
    def update_drone_state(self):
        """更新无人机状态"""
        drone_state = self.drone.get_state()[..., :13].squeeze(0)
        self.state_vars['state_x'] = drone_state[..., 0].item()
        self.state_vars['state_y'] = drone_state[..., 1].item()
        self.state_vars['state_z'] = drone_state[..., 2].item()
        _, self.state_vars['current_yaw'] = process_quaternion(drone_state, self.rot_z_45)
        return drone_state
    
    def record_data(self, depth1, depth2):
        """记录数据"""
        self.data_records_temp['state_xs'].append(self.state_vars['state_x'])
        self.data_records_temp['state_ys'].append(self.state_vars['state_y'])
        self.data_records_temp['state_yaws'].append(self.state_vars['current_yaw'].item())
        self.data_records_temp['state_lasers1'].append(depth1.item())
        self.data_records_temp['state_lasers2'].append(depth2.item())
        self.data_records_temp['laser_values1'].append(self.state_vars['laser_value1'])
        self.data_records_temp['laser_values2'].append(self.state_vars['laser_value2'])
        self.state_vars['laser_value1'] = 2
        self.state_vars['laser_value2'] = 2
    
    def check_room_boundary(self):
        """检查是否超出房间边界"""
        x, y = self.state_vars['state_x'], self.state_vars['state_y']
        return (x < self.ROOM_X_MIN or x > self.ROOM_X_MAX or 
                y < self.ROOM_Y_MIN or y > self.ROOM_Y_MAX)
    
    def handle_boundary_violation(self, drone_state):
        """处理边界违规"""
        self.data_records_temp = {k: [] for k in self.data_records_temp}
        print(f"🚨 无人机超出房间范围 (x={self.state_vars['state_x']}, y={self.state_vars['state_y']})，检查地图不确定性是否符合要求！")
        self.state_vars['outside'] = True
        gpis = GPISModel(
            self.data_records['state_xs'], 
            self.data_records['state_ys'], 
            self.data_records['state_yaws'], 
            self.data_records['state_lasers1'],
            self.data_records['state_lasers2'], 
            self.data_records['laser_values1'], 
            self.data_records['laser_values2'], 
            curvature_threshold=-0.7
        )
        gpis.sample_data()
        gpis.train_model()
        gpis.predict()
        gpis.find_max_uncertainty_point()
        gpis.plot_results(filename='gpis_results.png')
        print(f"uncertainty percentage now is: {gpis.uncertainty_retained_percentage}%")
        
        if gpis.uncertainty_retained_percentage < 25:
            # 使用预先设置的end_point作为目标点
            target_point = self.state_vars['end_point'] if self.state_vars['end_point'] is not None \
                          else self.state_vars['next_point']
            
            uncertainty_grid = np.resize(gpis.penalized_uncertainty_grid, (100, 100))
            value_grid = np.resize(gpis.Z, (100, 100))
            optimizer = TrajectoryOptimizer(
                uncertainty_grid,
                value_grid, 
                lambda_align=0, 
                lambda_align_start=0.5,
                lambda_smooth=0.1,
                lambda_v=100, 
                n_control=3, 
                steps=100,
                find_the_goal=True
            )
            
            self.state_vars['trajectory'] = optimizer.optimize(
                np.array([self.state_vars['state_x'], self.state_vars['state_y']]), 
                self.state_vars['current_yaw']-0.7853981,
                target_point,  # 使用目标点
                0
            )
            optimizer.visualize_trajectory(
                self.state_vars['trajectory'], 
                np.array([self.state_vars['state_x'], self.state_vars['state_y']]), 
                self.state_vars['current_yaw']-0.7853981, 
                target_point,  # 使用目标点
                0
            )
            
            self.state_vars['finish_CF'] = False
            self.state_vars['traj_index'] = 0
            self.state_vars['last_trajectory'] = True 
            self.transition_to(DroneState.EXIT)
        else:
            # 环境未探索完，使用之前生成的轨迹反向导航
            if len(self.state_vars['trajectory']) > 0:
                # 1. 找到当前状态距离轨迹最近的点
                current_pos = np.array([self.state_vars['state_x'], self.state_vars['state_y']])
                distances = [np.linalg.norm(current_pos - point) for point in self.state_vars['trajectory']]
                closest_idx = np.argmin(distances)
                
                # 2. 只保留最近点之后的轨迹部分（反转前）
                remaining_trajectory = self.state_vars['trajectory'][:closest_idx]
                
                # 3. 反转剩余轨迹
                reversed_trajectory = remaining_trajectory[::-1]
                self.state_vars['trajectory'] = reversed_trajectory
                
                self.state_vars['finish_CF'] = False
                self.state_vars['traj_index'] = 0
                self.state_vars['back'] = True
                self.transition_to(DroneState.FOLLOW_TRAJECTORY)
                if not self.state_vars['last_trajectory']:
                    self.state_vars['outside'] = False
                print("reverse traj")
            else:
                self.state_vars['back'] = True
                _, self.state_vars['before_yaw'] = process_quaternion(drone_state, self.rot_z_45)
                self.state_vars['target_yaw'] = self.state_vars['before_yaw'] + torch.pi
                if not self.state_vars['last_trajectory']:
                    self.state_vars['outside'] = False
                self.transition_to(DroneState.CHANGE_DIRECTION)
    
    def save_data(self, filename):
        """保存数据到CSV"""
        df = pd.DataFrame(self.data_records)
        df.to_csv(filename, index=False)
    
    def run_state(self):
        """执行当前状态"""
        depth1, depth2 = self.get_depth_data()
        drone_state = self.update_drone_state()
        
        if self.current_state == DroneState.IDLE:
            self.state_idle(drone_state, depth1, depth2)
        elif self.current_state == DroneState.FIND_GOAL:
            self.state_find_goal(drone_state)
        elif self.current_state == DroneState.CF_ACTION:
            self.state_cf_action(drone_state, depth1, depth2)
        elif self.current_state == DroneState.BACKWARD:
            self.state_backward(drone_state, depth1, depth2)
        elif self.current_state == DroneState.CHANGE_DIRECTION:
            self.state_change_direction(drone_state)
        elif self.current_state == DroneState.FOLLOW_TRAJECTORY:
            self.state_follow_trajectory(drone_state, depth1, depth2)
        elif self.current_state == DroneState.EXIT:
            self.state_follow_trajectory(drone_state, depth1, depth2)
        
        # 记录数据
        self.record_data(depth1, depth2)
        
        # 检查边界
        if self.check_room_boundary() and self.current_state != DroneState.EXIT and not self.state_vars['back']:
            self.handle_boundary_violation(drone_state)
    def state_idle(self, drone_state, depth1, depth2):
        """空闲状态"""
        if (self.MIN_THRESHOLD < depth1 < self.MAX_THRESHOLD and 
            self.MIN_THRESHOLD < depth2 < self.MAX_THRESHOLD):
            # 进入CF动作状态
            self.state_vars['CF_action_counter'] = 200
            self.state_vars['backward_action_counter'] = 150
            self.state_vars['Forward_counter'] = 0
            random_direction_rad = np.deg2rad(-90)
            self.state_vars['random_yaw'] = torch.tensor([random_direction_rad], device=self.sim.device)
            self.transition_to(DroneState.CF_ACTION)
        else:
            if depth1 > 0.48 and depth2 > 0.48 and self.state_vars['Forward_counter'] % 200 == 0:
                if self.state_vars['outside']:
                    if not self.state_vars.get('back'):
                        self.state_vars['laser_value1'] = 1
                        self.state_vars['laser_value2'] = 1
                else:
                    if not self.state_vars.get('back'):
                        self.state_vars['laser_value1'] = -1
                        self.state_vars['laser_value2'] = -1
            self.state_vars['Forward_counter'] += 1
            self.state_vars['Forward_counter'] = control_drone(
                self.drone, drone_state, depth1, depth2, 
                self.vel_forward, self.vel_backward, 
                self.vel_side, self.rot_z_45, self.controller, 
                self.yaw_left, self.yaw_right, 
                self.MIN_THRESHOLD, self.MAX_THRESHOLD, 
                self.state_vars['Forward_counter']
            )

    def state_find_goal(self, drone_state):
        """寻找目标状态"""
        R_transpose, _ = process_quaternion(drone_state, self.rot_z_45)
        goal_world = transform_velocity(self.vel_side, R_transpose)
        apply_control(self.drone, drone_state, self.controller, goal_world, "Find the goal")
        
        self.state_vars['goal_counter'] -= 1
        if self.state_vars['goal_counter'] <= 0:
            self.transition_to(DroneState.IDLE)
            self.state_vars['trajectory'] = []
    
    def state_cf_action(self, drone_state, depth1, depth2):
        if len(self.data_records_temp['state_xs']) > 0:
            for key in self.data_records:
                self.data_records[key].extend(self.data_records_temp[key])
        self.data_records_temp = {k: [] for k in self.data_records_temp}
        """CF动作状态"""
        self.state_vars['depth_last'] = self.state_vars['depth_now']
        self.state_vars['depth_now'] = depth2
        self.state_vars['back'] = False
        self.state_vars['CF_action_counter'] = control_drone(
            self.drone, drone_state, depth1, depth2, 
            self.cf_vel_forward, self.cf_vel_backward, 
            self.vel_side, self.rot_z_45, self.controller, 
            self.yaw_left, self.yaw_right, 
            self.MIN_THRESHOLD, self.MAX_THRESHOLD, 
            self.state_vars['CF_action_counter']
        )
        residuals = self.state_vars['depth_now'] - self.state_vars['depth_last']
        if depth1 < 0.48 and self.state_vars['CF_action_counter'] % 40 == 0 and not self.state_vars['outside']:
            self.state_vars['laser_value1'] = 0
        if depth2 < 0.48 and self.state_vars['CF_action_counter'] % 40 == 0 and not self.state_vars['outside']:
            self.state_vars['laser_value2'] = 0
        if residuals > 0.05 and not self.state_vars['last_trajectory']:
            self.state_vars['goal_counter'] = 100
            self.state_vars['CF_action_counter'] = 0
            self.state_vars['backward_action_counter'] = 0
            self.state_vars['direction_change_counter'] = 0
            self.transition_to(DroneState.FIND_GOAL)
            self.state_vars['finish_CF'] = True
        
        elif self.state_vars['CF_action_counter'] <= 0:
            self.transition_to(DroneState.BACKWARD)
            self.state_vars['finish_CF'] = True
            self.state_vars['depth_now'] = 1
        print(self.state_vars['depth_last'] )
        print(self.state_vars['depth_now'] )

    
    def state_backward(self, drone_state, depth1, depth2):
        """后退状态"""
        R_transpose, _ = process_quaternion(drone_state, self.rot_z_45)
        backward_world = transform_velocity(self.vel_backward, R_transpose)
        apply_control(self.drone, drone_state, self.controller, backward_world, "fly backward")
        
        self.state_vars['backward_action_counter'] -= 1
        if depth1 > 0.48 and depth2 > 0.48 and self.state_vars['backward_action_counter'] % 200 == 0:
            self.state_vars['laser_value1'] = -1
            self.state_vars['laser_value2'] = -1
            _, self.state_vars['before_yaw'] = process_quaternion(drone_state, self.rot_z_45)
            self.state_vars['target_yaw'] = self.state_vars['before_yaw'] + self.state_vars['random_yaw']
        
        if self.state_vars['backward_action_counter'] <= 0 and self.state_vars['direction_changes_completed'] < 3:
            self.transition_to(DroneState.CHANGE_DIRECTION)
        elif self.state_vars['backward_action_counter'] <= 0 and self.state_vars['direction_changes_completed'] >= 3 and self.state_vars['finish_CF']:
            self.handle_gpis_analysis()
            self.transition_to(DroneState.FOLLOW_TRAJECTORY)
    
    def state_change_direction(self, drone_state):
        """改变方向状态"""
        
        perform_attitude_control(
            self.drone, drone_state, self.controller, 
            self.state_vars['target_yaw'], "change orientation"
        )
        
        if torch.abs(normalize_angle(self.state_vars['current_yaw']) - normalize_angle(self.state_vars['target_yaw'])) < 1.5:
            self.state_vars['direction_changes_completed'] += 1
            self.transition_to(DroneState.IDLE)
    
    def handle_gpis_analysis(self):
        """处理GPIS分析和轨迹规划"""
        gpis = GPISModel(
            self.data_records['state_xs'], 
            self.data_records['state_ys'], 
            self.data_records['state_yaws'], 
            self.data_records['state_lasers1'],
            self.data_records['state_lasers2'], 
            self.data_records['laser_values1'], 
            self.data_records['laser_values2'], 
            curvature_threshold=-0.7
        )
        gpis.sample_data()
        gpis.train_model()
        gpis.predict()
        print(f"uncertainty percentage now is: {gpis.uncertainty_retained_percentage}%")
        if gpis.uncertainty_retained_percentage < 25 and self.state_vars['exit_point'] is not None:
            self.state_vars['next_point'] = self.state_vars['exit_point']
            print("finish exploration and found the exit")
        else:
            self.state_vars['next_point'] = gpis.find_max_uncertainty_point()
            gpis.significant_points = np.vstack([gpis.significant_points, self.state_vars['next_point']])
            print("exploration not yet complete")    
            gpis.plot_results(filename='gpis_results.png')
        # 计算目标朝向
        
        # 轨迹规划
        goal_yaw = gpis.compute_normal(
            self.state_vars['next_point'], gpis.weights, gpis.X_train, gpis.kernel
        )
        uncertainty_grid = np.resize(gpis.penalized_uncertainty_grid, (100, 100))
        value_grid = np.resize(gpis.Z, (100, 100)) 
        optimizer = TrajectoryOptimizer(
            uncertainty_grid, 
            value_grid,
            lambda_align=0.2, 
            lambda_smooth=0.8, 
            n_control=5, 
            steps=100
        )
        
        self.state_vars['trajectory'] = optimizer.optimize(
            np.array([self.state_vars['state_x'], self.state_vars['state_y']]), 
            self.state_vars['current_yaw']-0.7853981,
            self.state_vars['next_point'], 
            goal_yaw
        )
        optimizer.visualize_trajectory(
            self.state_vars['trajectory'], 
            np.array([self.state_vars['state_x'], self.state_vars['state_y']]), 
            self.state_vars['current_yaw']-0.7853981, 
            self.state_vars['next_point'], 
            goal_yaw
        )
        
        self.state_vars['finish_CF'] = False
        self.state_vars['traj_index'] = 0
    
    def state_follow_trajectory(self, drone_state, depth1, depth2):
        """跟随轨迹状态"""
        if self.state_vars['traj_index'] < len(self.state_vars['trajectory']) - 1:
            p1 = self.state_vars['trajectory'][self.state_vars['traj_index']]
            p2 = self.state_vars['trajectory'][self.state_vars['traj_index'] + 1]
            
            dir_vec = p2 - p1
            dir_vec /= np.linalg.norm(dir_vec) + 1e-6
            yaw = np.arctan2(dir_vec[1], dir_vec[0])
            
            ref_pos = torch.tensor([p1[0], p1[1], 1], dtype=torch.float32, device=self.sim.device)
            ref_yaw = torch.tensor([yaw], dtype=torch.float32, device=self.sim.device) + 0.7853981
            current_yaw = torch.tensor(self.state_vars['current_yaw'], dtype=torch.float32, device=self.sim.device)
            drone_pos = torch.tensor(
                [self.state_vars['state_x'], self.state_vars['state_y'], self.state_vars['state_z']], 
                dtype=torch.float32, device=self.sim.device
            )
            
            distance = torch.norm(drone_pos - ref_pos)
            yaw_diff = torch.arctan2(torch.sin(ref_yaw - current_yaw), torch.cos(ref_yaw - current_yaw))
            max_yaw_step = torch.tensor(np.pi / 6, device=self.sim.device)
            yaw_step = torch.clamp(yaw_diff, -max_yaw_step, max_yaw_step)
            intermediate_yaw = current_yaw + yaw_step
            
            action = self.controller(drone_state, target_pos=ref_pos, target_yaw=intermediate_yaw)
            self.drone.apply_action(action)
            
            if distance < 0.01 and torch.abs(yaw_diff) < np.deg2rad(2):
                self.state_vars['traj_index'] += 1
                print(f"Arrived at waypoint {self.state_vars['traj_index']}, moving to next.")
                # 如果是返回轨迹且到达终点
                if self.state_vars['back'] and self.state_vars['traj_index'] == len(self.state_vars['trajectory']) - 1:
                    self.state_vars['back'] = False
                    self.transition_to(DroneState.IDLE)
            
            if depth1 > 0.48 and depth2 > 0.48 and self.state_vars['Forward_counter'] % 200 == 0:
                if not self.state_vars.get('back'):
                    self.state_vars['laser_value1'] = -1
                    self.state_vars['laser_value2'] = -1
            
            self.state_vars['Forward_counter'] += 1
            
            if (self.MIN_THRESHOLD < depth1 < self.MAX_THRESHOLD or 
                self.MIN_THRESHOLD < depth2 < self.MAX_THRESHOLD):
                self.reset_to_cf_action()
        else:
            # 如果不是返回轨迹，则转回IDLE状态
            if not self.state_vars['back']:
                self.transition_to(DroneState.IDLE)
        
    def reset_to_cf_action(self):
        """重置到CF动作状态"""
        self.state_vars['CF_action_counter'] = 200
        self.state_vars['backward_action_counter'] = 150
        self.state_vars['direction_change_counter'] = 300
        self.state_vars['Forward_counter'] = 0
        self.state_vars['traj_index'] = 9999999
        self.transition_to(DroneState.CF_ACTION)

@hydra.main(version_base=None, config_path=".", config_name="demo")
def main(cfg):
    OmegaConf.resolve(cfg)
    simulation_app = init_simulation_app(cfg)
    print(OmegaConf.to_yaml(cfg))

    import omni_drones.utils.scene as scene_utils
    from omni.isaac.core import World
    from omni_drones.controllers import LeePositionController
    from omni_drones.robots.drone import MultirotorBase
    from omni.isaac.dynamic_control import _dynamic_control
    from pxr import Gf, Usd, UsdGeom, Sdf
    from omni.isaac.range_sensor import _range_sensor
    import carb

    carb.settings.get_settings().set("/app/show_developer_preference_section", True)

    sim = World(
        stage_units_in_meters=1.0,
        physics_dt=cfg.sim.dt,
        rendering_dt=cfg.sim.dt,
        sim_params=cfg.sim,
        backend="torch",
        device=cfg.sim.device,
    )

    scene_utils.design_scene()
    scene_utils.create_wall3()
    
    n = 1  # 设置无人机数量为1
    drone_cls = MultirotorBase.REGISTRY[cfg.drone_model]
    drone = drone_cls()

    translations = torch.zeros(n, 3, device=sim.device)
    translations[:, 1] = torch.arange(n)
    translations[:, 2] = 1
    orientations = torch.zeros(n, 4, device=sim.device)
    orientations[:, 0] = 0.9238795325  # w
    orientations[:, 3] = 0.3826834324  # z
    drone.spawn(translations=translations, orientations=orientations)

    lidarInterface = _range_sensor.acquire_lidar_sensor_interface()
    
    # 设置激光雷达传感器
    lidarPath1 = "/envs/env_0/Hummingbird_0/base_link/LidarSensor1"
    omni.kit.commands.execute(
        "RangeSensorCreateLidar",
        path=lidarPath1,
        parent="/World",
        min_range=0.3,
        max_range=0.5,
        draw_points=False,
        draw_lines=True,
        horizontal_fov=1,
        vertical_fov=1,
        horizontal_resolution=1,
        vertical_resolution=1,
        rotation_rate=0.0,
        high_lod=False,
        yaw_offset=-45.0,
        enable_semantics=False,
    )
    
    lidarPath2 = "/envs/env_0/Hummingbird_0/base_link/LidarSensor2"
    omni.kit.commands.execute(
        "RangeSensorCreateLidar",
        path=lidarPath2,
        parent="/World",
        min_range=0.3,
        max_range=0.5,
        draw_points=False,
        draw_lines=True,
        horizontal_fov=1,
        vertical_fov=1,
        horizontal_resolution=1,
        vertical_resolution=1,
        rotation_rate=0.0,
        high_lod=False,
        yaw_offset=-45.0,
        enable_semantics=False,
    )
    
    # 设置激光雷达位置和方向
    position1 = Gf.Vec3d(0.08, 0.08, 0)
    rotation1 = Gf.Vec3d(0, 0, 0)
    scale1 = Gf.Vec3d(1, 1, 1) 
    omni.kit.commands.execute("TransformMultiPrimsSRTCpp",
        count=1,
        paths=['/World'+lidarPath1],
        new_translations=[position1[0], position1[1], position1[2]],
        new_rotation_eulers=[rotation1[0], rotation1[1], rotation1[2]],
        new_rotation_orders=[1, 0, 2],
        new_scales=[scale1[0], scale1[1], scale1[2]],
        old_translations=[0.0, 0.0, 0.0],
        old_rotation_eulers=[0.0, -0.0, -0.0],
        old_rotation_orders=[1, 0, 2],
        old_scales=[1.0, 1.0, 1.0],
        time_code=0.0,
    )

    position2 = Gf.Vec3d(-0.08, -0.08, 0)
    rotation2 = Gf.Vec3d(0, 0, 0)
    scale2 = Gf.Vec3d(1, 1, 1) 
    omni.kit.commands.execute("TransformMultiPrimsSRTCpp",
        count=1,
        paths=['/World'+lidarPath2],
        new_translations=[position2[0], position2[1], position2[2]],
        new_rotation_eulers=[rotation2[0], rotation2[1], rotation2[2]],
        new_rotation_orders=[1, 0, 2],
        new_scales=[scale2[0], scale2[1], scale2[2]],
        old_translations=[0.0, 0.0, 0.0],
        old_rotation_eulers=[0.0, -0.0, -0.0],
        old_rotation_orders=[1, 0, 2],
        old_scales=[1.0, 1.0, 1.0],
        time_code=0.0,
    )
    
    sim.reset()
    drone.initialize()

    end_point = np.array([0, -3.5])  # 示例终点坐标

    # 创建状态机实例
    drone_fsm = DroneFSM(cfg, sim, drone, lidarInterface, lidarPath1, lidarPath2, end_point=end_point)

    # 主循环
    for i in tqdm(range(20000)):
        if sim.is_stopped():
            break
        if not sim.is_playing():
            sim.render()
            continue
        
        # 运行状态机
        drone_fsm.run_state()
        
        # 仿真步进
        sim.step(render=(i % 10 == 0))
        
        # 检查是否退出CF_ACTION
        if drone_fsm.current_state == DroneState.LAND:
            drone_fsm.save_data('T-90-ours_success.csv')
            print("find the goal and land, mission complete")
            break
    
    # 保存数据
    if drone_fsm.current_state != DroneState.EXIT:
        drone_fsm.save_data('T-90-ours_fail.csv')

    simulation_app.close()

if __name__ == "__main__":
    main()