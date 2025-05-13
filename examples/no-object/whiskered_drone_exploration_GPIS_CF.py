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

@hydra.main(version_base=None, config_path=".", config_name="demo")
def main(cfg):
    OmegaConf.resolve(cfg)
    simulation_app = init_simulation_app(cfg)
    print(OmegaConf.to_yaml(cfg))

    import omni_drones.utils.scene as scene_utils
    from omni.isaac.core import World
    from omni_drones.controllers import LeePositionController, AttitudeController
    from omni_drones.robots.drone import MultirotorBase
    from omni_drones.utils.torch import euler_to_quaternion
    from omni.isaac.dynamic_control import _dynamic_control
    from pxr import Gf, Usd, UsdGeom, Sdf
    import typing
    from omni.isaac.range_sensor import _range_sensor

    from omni_drones.utils.torch import quaternion_to_rotation_matrix
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
    MAX_THRESHOLD = 0.45
    MIN_THRESHOLD = 0.4

    drone_cls = MultirotorBase.REGISTRY[cfg.drone_model]
    drone = drone_cls()

    translations = torch.zeros(n, 3, device = sim.device)
    translations[:, 1] = torch.arange(n)
    translations[:, 2] = 1
    orientations = torch.zeros(n, 4, device = sim.device)
    orientations[:, 0] = 0.9238795325  # w
    orientations[:, 3] = 0.3826834324  # z
    drone.spawn(translations=translations,orientations=orientations)

    lidarInterface = _range_sensor.acquire_lidar_sensor_interface()
    
    lidarPath1 = "/envs/env_0/Hummingbird_0/base_link/LidarSensor1"
    omni.kit.commands.execute(
            "RangeSensorCreateLidar",
            path=lidarPath1,
            parent = "/World",
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
            parent = "/World",
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
    
    position1 = Gf.Vec3d(0.08,0.08,0) # HERE PUT THE WANTED VALUE
    rotation1 = Gf.Vec3d(0,0,0)
    scale1 = Gf.Vec3d(1,1,1) 
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

    position2 = Gf.Vec3d(-0.08,-0.08,0) # HERE PUT THE WANTED VALUE
    rotation2 = Gf.Vec3d(0,0,0)
    scale2 = Gf.Vec3d(1,1,1) 
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


    # 定义圆形轨迹

    init_pos = translations
    init_rot = orientations
    init_vels = torch.zeros(n, 6, device=sim.device)

    pos = torch.tensor([0.0, 0.0, 0.0], device=sim.device)
    vel_forward = torch.tensor([0.2, 0.0, 0.0], device=sim.device)
    vel_side = torch.tensor([0, -0.2, 0.0], device=sim.device)
    vel_backward = torch.tensor([-0.2, 0.0, 0.0], device=sim.device)
    yaw = torch.tensor([0.0], device=sim.device)
    yaw_right_rad = np.deg2rad(-25)
    yaw_left_rad = np.deg2rad(25)
    yaw_right = torch.tensor([yaw_right_rad], device=sim.device)
    yaw_left = torch.tensor([yaw_left_rad], device=sim.device)
    vel_backward = torch.tensor([-0.2, 0, 0], device=sim.device)
    cf_vel_backward = torch.tensor([-0.05, 0, 0], device=sim.device)
    cf_vel_forward = torch.tensor([0.05, 0, 0], device=sim.device)
    rot = 0
    depth_now = 0.5
    depth_last = 0.5

    random_direction = 0
    random_direction_rad = 0
    random_yaw = 0
    CF_action_counter = 0
    backward_action_counter = 0
    direction_change_counter = 0
    goal_counter = 0
    before_yaw = 0
    current_yaw = 0
    depth1_noisy = 0
    depth2_noisy = 0

    traj_index=9999999

    # Define the rotation quaternion for a -45 degree rotation around the Z-axis
    theta = -45.0  # degrees
    theta_rad = np.radians(theta)
    cos_half_theta = np.cos(theta_rad / 2)
    sin_half_theta = np.sin(theta_rad / 2)
    rot_z_45 = torch.tensor([cos_half_theta, 0.0, 0.0, sin_half_theta], device=sim.device)
    state_yaw = 0
    state_x = 0
    state_y = 0
    state_xs = []
    state_ys = []
    state_yaws = []
    state_lasers1 = []
    state_lasers2 = []
    laser_value1 = 0
    laser_value2 = 0
    laser_values1 = []
    laser_values2 = []
    direction_changes_completed = 0
    finish_CF = False
    Forward_counter = 0
    ROOM_X_MIN, ROOM_X_MAX = -2.8 * 1.33, 2.8 * 1.33
    ROOM_Y_MIN, ROOM_Y_MAX = -2.8 * 1.33, 2.8 * 1.33
    exit_point = None
    trajectory = [] 

    # 创建位置控制器
    controller = LeePositionController(g=9.81, uav_params=drone.params).to(sim.device)
    # Att_controller = AttitudeController(g=9.81, uav_params=drone.params).to(sim.device)

    def reset():
        drone._reset_idx(torch.tensor([0]))
        drone.set_world_poses(init_pos, init_rot)
        drone.set_velocities(init_vels)
        sim._physics_sim_view.flush()

    reset()
    drone_state = drone.get_state()[..., :13].squeeze(0)

    from tqdm import tqdm
    for i in tqdm(range(20000)):
        if sim.is_stopped():
            break
        if not sim.is_playing():
            sim.render()
            continue
        # 获取深度数据
        depth1 = lidarInterface.get_linear_depth_data("/World" + lidarPath1)
        depth2 = lidarInterface.get_linear_depth_data("/World" + lidarPath2)

        # 打印原始深度数据
        # print("Original depth1:", depth1)
        # print("Original depth2:", depth2)

        # 定义高斯噪声的均值和标准差
        mean = 0.0
        std_dev = 0.01

        # 为深度数据生成高斯噪声
        noise1 = np.random.normal(mean, std_dev, depth1.shape)
        noise2 = np.random.normal(mean, std_dev, depth2.shape)

        # 将噪声添加到深度数据中
        depth1_noisy = depth1 + noise1
        depth2_noisy = depth2 + noise2

        # 打印添加噪声后的深度数据
        # print("Noisy depth1:", depth1_noisy)
        # print("Noisy depth2:", depth2_noisy)
        #find the goal
        state_x = drone.get_state()[..., 0].item()
        state_y = drone.get_state()[..., 1].item()
        state_z = drone.get_state()[..., 2].item()
        if goal_counter > 0:
            R_transpose, _ = process_quaternion(drone_state, rot_z_45)
            goal_world = transform_velocity(vel_side, R_transpose)
            apply_control(drone, drone_state, controller, goal_world, "Find the goal")
            goal_counter -= 1
        elif CF_action_counter > 0:
            CF_action_counter = control_drone(drone, drone_state, depth1_noisy, depth2_noisy, cf_vel_forward, cf_vel_backward, 
                                                vel_side, rot_z_45, controller, yaw_left, yaw_right, 
                                                MIN_THRESHOLD, MAX_THRESHOLD, CF_action_counter)
            depth_last = depth_now
            depth_now = depth2_noisy
            residuals = depth_now - depth_last
            if residuals > 0.08:
                goal_counter = 100
                CF_action_counter = 0
                backward_action_counter = 0
                direction_change_counter = 0
            if depth1_noisy < 0.48 and CF_action_counter % 40 == 0:
                laser_value1 = 1
            if depth2_noisy < 0.48 and CF_action_counter % 40 == 0:
                laser_value2 = 1
        elif backward_action_counter > 0:
            R_transpose, _ = process_quaternion(drone_state, rot_z_45)
            backward_world = transform_velocity(vel_backward, R_transpose)
            apply_control(drone, drone_state, controller, backward_world, "fly backward")
            backward_action_counter -= 1
            if depth1_noisy > 0.48 and depth2_noisy > 0.48 and backward_action_counter % 200 == 0:
                laser_value1 = -1
                laser_value2 = -1
                depth1_noisy = depth1 + noise1
                depth2_noisy = depth2 + noise2
                _, before_yaw = process_quaternion(drone_state, rot_z_45)
            target_yaw = before_yaw + random_yaw
        elif direction_change_counter > 0:
            _, current_yaw = process_quaternion(drone_state, rot_z_45)
            perform_attitude_control(drone, drone_state, controller, target_yaw, "change orientation")
            direction_change_counter -= 1
            if torch.abs(normalize_angle(current_yaw) - normalize_angle(target_yaw)) < 1.5:
                direction_change_counter = 0
                direction_changes_completed += 1
            if direction_changes_completed >= 4 and finish_CF:
                gpis = GPISModel(state_xs, state_ys, state_yaws, state_lasers1,state_lasers2, laser_values1, laser_values2, curvature_threshold=-0.7)
                gpis.sample_data()
                gpis.train_model()
                gpis.predict()
                print(f"uncertainty percentage now is: {gpis.uncertainty_retained_percentage}%")
                if gpis.uncertainty_retained_percentage < 25 and exit_point is not None:
                    next_point = exit_point
                    target_yaw = torch.tensor([np.arctan2(next_point[1] - state_y, next_point[0] - state_x)], device=sim.device) + 0.7853981
                    direction_change_counter = 500
                    gpis.plot_results(filename='gpis_results.png')
                    finish_CF = False
                    print("finish exploration and found the exit")
                else: 
                    next_point = gpis.find_max_uncertainty_point()
                    target_yaw = torch.tensor([np.arctan2(next_point[1] - state_y, next_point[0] - state_x)], device=sim.device) + 0.7853981
                    direction_change_counter = 500
                    gpis.plot_results(filename='gpis_results.png')
                    finish_CF = False
                # Instantiate and plan using the new weighted RRT*
                goal_yaw = gpis.compute_normal(next_point,gpis.weights,gpis.X_train,gpis.kernel)
                uncertainty_grid=np.resize(gpis.penalized_uncertainty_grid, (100, 100))
                optimizer = TrajectoryOptimizer(uncertainty_grid, lambda_align=0.2, lambda_smooth=0.8, n_control=5, steps=100)
                trajectory = optimizer.optimize(np.array([state_x,state_y]), next_point, goal_yaw )
                optimizer.visualize_trajectory(trajectory, np.array([state_x,state_y]), current_yaw, next_point, goal_yaw )
                direction_change_counter = 0
                traj_index = 0
            # print(torch.rad2deg(current_yaw))
            # print(torch.rad2deg(target_yaw))
        elif traj_index < len(trajectory) - 1:
            _, current_yaw = process_quaternion(drone_state, rot_z_45)
            # 获取当前和下一个轨迹点
            p1 = trajectory[traj_index]  # 当前轨迹点
            p2 = trajectory[traj_index + 1]  # 下一个轨迹点
            
            
            # 计算相邻两个轨迹点之间的方向向量
            dir_vec = p2- p1 # 计算方向向量
            dir_vec /= np.linalg.norm(dir_vec) + 1e-6  # 单位化方向向量
            
            # 使用方向向量计算航向（yaw）
            yaw = np.arctan2(dir_vec[1], dir_vec[0])  # 使用 atan2 计算航向角度
            
            # 将目标位置和航向转换为 Tensor
            ref_pos = torch.tensor([p1[0], p1[1], 1], dtype=torch.float32, device=sim.device)  # 当前轨迹点的位置作为目标位置
            ref_yaw = torch.tensor([yaw], dtype=torch.float32, device=sim.device) + 0.7853981  # 计算得到的目标航向
            current_yaw = torch.tensor(current_yaw, dtype=torch.float32, device=sim.device)
            # 获取无人机当前位置
            drone_pos = torch.tensor([state_x, state_y, state_z], dtype=torch.float32, device=sim.device)  # 假设无人机状态的前三个是位置
            
            # 计算当前位置和目标位置之间的欧几里得距离
            distance = torch.norm(drone_pos - ref_pos)  # 计算欧几里得距离

            # 计算航向差异，考虑角度的周期性（[-pi, pi]）
            yaw_diff = torch.arctan2(torch.sin(ref_yaw - current_yaw), torch.cos(ref_yaw - current_yaw))
            max_yaw_step = torch.tensor(np.pi / 6, device=sim.device)  # 每步最多旋转30°
            yaw_step = torch.clamp(yaw_diff, -max_yaw_step, max_yaw_step)

            # 计算本步调整后的目标朝向（不是最终ref_yaw，而是朝它靠近一步）
            intermediate_yaw = current_yaw + yaw_step
            # 控制器计算出动作（命令）
            action = controller(drone_state, target_pos=ref_pos, target_yaw=intermediate_yaw)
            drone.apply_action(action)
            # print(f"ref pos: x={ref_pos[0]:.4f}, y={ref_pos[1]:.4f}, z={ref_pos[2]:.4f}")
            # print(f"ref yaw: {ref_yaw.item():.4f}")
            # print(f"current pos: x={drone_pos[0]:.4f}, y={drone_pos[1]:.4f}, z={drone_pos[2]:.4f}")
            # print("current yaw: %.4f" % current_yaw )
            # print("yaw diff: %.4f" % yaw_diff)
            # print("ditance: %.4f" % distance)
            # 如果距离小于阈值，切换到下一个轨迹点
            if distance < 0.01 and torch.abs(yaw_diff) < np.deg2rad(2):
                traj_index += 1  # 移动到下一个轨迹点
                print(f"Arrived at waypoint {traj_index}, moving to next.")
            if depth1_noisy > 0.48 and depth2_noisy > 0.48 and Forward_counter % 200 == 0:
                laser_value1 = -1
                laser_value2 = -1
            Forward_counter += 1
            if MAX_THRESHOLD > depth1_noisy > MIN_THRESHOLD or MAX_THRESHOLD > depth2_noisy > MIN_THRESHOLD:
                CF_action_counter = 200
                backward_action_counter = 150
                direction_change_counter = 300
                Forward_counter = 0
                finish_CF = True
                traj_index = 9999999
        else:
            if MAX_THRESHOLD > depth1_noisy > MIN_THRESHOLD and MAX_THRESHOLD > depth2_noisy > MIN_THRESHOLD:
                CF_action_counter = 200
                backward_action_counter = 150
                direction_change_counter = 300
                Forward_counter = 0
                finish_CF = True
                random_direction_rad = np.deg2rad(-90)
                random_yaw = torch.tensor([random_direction_rad], device=sim.device)
                # print("CF start")
            else:
                if depth1_noisy > 0.48 and depth2_noisy > 0.48 and Forward_counter % 200 == 0:
                    laser_value1 = -1
                    laser_value2 = -1
                Forward_counter += 1
                control_drone(drone, drone_state, depth1_noisy, depth2_noisy, vel_forward, 
                            vel_backward, vel_side, rot_z_45, controller, 
                            yaw_left, yaw_right,  MIN_THRESHOLD, MAX_THRESHOLD)
        sim.step(render=(i % 10 == 0))
        # lidarInterface.update() 
        drone_state = drone.get_state()[..., :13].squeeze(0)
        # print(drone_state)
        # print(direction_changes_completed)
        state_xs.append(state_x)
        state_ys.append(state_y) 
        _ , state_yaw = process_quaternion(drone_state, rot_z_45)
        state_yaws.append(state_yaw.item())
        state_lasers1.append(depth1_noisy.item()) 
        state_lasers2.append(depth2_noisy.item())
        laser_values1.append(laser_value1)
        laser_values2.append(laser_value2)
        laser_value1 = 0
        laser_value2 = 0 
    

            # **检查无人机是否飞出房间**
        if state_x < ROOM_X_MIN or state_x > ROOM_X_MAX or state_y < ROOM_Y_MIN or state_y > ROOM_Y_MAX:
            print(f"🚨 无人机超出房间范围 (x={state_x}, y={state_y})， 检查地图不确定性是否符合要求！")
            gpis = GPISModel(state_xs, state_ys, state_yaws, state_lasers1,state_lasers2, laser_values1, laser_values2, curvature_threshold=-0.7)
            gpis.sample_data()
            gpis.train_model()
            gpis.predict()
            print(f"uncertainty percentage now is: {gpis.uncertainty_retained_percentage}%")
            if gpis.uncertainty_retained_percentage < 25:
                # **每轮循环都保存 CSV**
                data = {
                    'state_xs': state_xs,
                    'state_ys': state_ys,
                    'state_yaws': state_yaws,
                    'state_lasers1': state_lasers1,
                    'state_lasers2': state_lasers2,
                    'laser_values1': laser_values1,
                    'laser_values2': laser_values2,
                }
                df = pd.DataFrame(data)
                df.to_csv('T-90-ours_success.csv', index=False)  # **实时保存**
                print("find the goal, mission complete")
                break  # 无人机飞出房间，结束任务
            else:
                direction_change_counter = 600
                random_direction_rad = np.deg2rad(-180)
                random_yaw = torch.tensor([random_direction_rad], device=sim.device)
                gpis.significant_points=np.vstack([gpis.significant_points, next_point])
                exit_point = next_point
                print("exploration not yet complete, back to the room")




    # **保存数据并确保仿真关闭**
    data = {
        'state_xs': state_xs,
        'state_ys': state_ys,
        'state_yaws': state_yaws,
        'state_lasers1': state_lasers1,
        'state_lasers2': state_lasers2,
        'laser_values1': laser_values1,
        'laser_values2': laser_values2,
    }
    df = pd.DataFrame(data)
    df.to_csv('T-90-ours_fail.csv', index=False)

    simulation_app.close()  # 仿真结束后关闭

if __name__ == "__main__":
    main()