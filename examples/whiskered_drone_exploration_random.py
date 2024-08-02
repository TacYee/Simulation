import torch

import hydra
from omegaconf import OmegaConf
from omni_drones import CONFIG_PATH, init_simulation_app
import omni
import numpy as np


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


    sim = World(
        stage_units_in_meters=1.0,
        physics_dt=cfg.sim.dt,
        rendering_dt=cfg.sim.dt,
        sim_params=cfg.sim,
        backend="torch",
        device=cfg.sim.device,
    )

    scene_utils.design_scene()
    scene_utils.create_wall()
    
    n = 1  # 设置无人机数量为1
    MAX_THRESHOLD = 0.42
    MIN_THRESHOLD = 0.37

    drone_cls = MultirotorBase.REGISTRY[cfg.drone_model]
    drone = drone_cls()

    translations = torch.zeros(n, 3, device = sim.device)
    translations[:, 1] = torch.arange(n)
    translations[:, 2] = 1
    orientations = torch.zeros(n, 4, device = sim.device)
    drone.spawn(translations=translations,orientations=orientations)

    lidarInterface = _range_sensor.acquire_lidar_sensor_interface()
    
    lidarPath1 = "/envs/env_0/Hummingbird_0/base_link/LidarSensor1"
    omni.kit.commands.execute(
            "RangeSensorCreateLidar",
            path=lidarPath1,
            parent = "/World",
            min_range=0.3,
            max_range=0.45,
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
            max_range=0.45,
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
    
    position1 = Gf.Vec3d(0.1,0.1,0) # HERE PUT THE WANTED VALUE
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

    position2 = Gf.Vec3d(-0.1,-0.1,0) # HERE PUT THE WANTED VALUE
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
    yaw_rate_right_rad = np.deg2rad(-15)
    yaw_rate_left_rad = np.deg2rad(15)
    yaw_rate_right = torch.tensor([yaw_rate_right_rad], device=sim.device)
    yaw_rate_left = torch.tensor([yaw_rate_left_rad], device=sim.device)
    vel_backward = torch.tensor([-0.2, 0, 0], device=sim.device)
    rot = 0

    random_direction = 0
    random_direction_rad = 0
    random_yaw = 0
    special_action_counter = 0
    direction_change_counter = 0
    

    # 创建位置控制器
    controller = LeePositionController(g=9.81, uav_params=drone.params).to(sim.device)
    Att_controller = AttitudeController(g=9.81, uav_params=drone.params).to(sim.device)

    def reset():
        drone._reset_idx(torch.tensor([0]))
        drone.set_world_poses(init_pos, init_rot)
        drone.set_velocities(init_vels)
        sim._physics_sim_view.flush()

    reset()
    drone_state = drone.get_state()[..., :13].squeeze(0)

    from tqdm import tqdm
    for i in tqdm(range(2000)):
        if sim.is_stopped():
            break
        if not sim.is_playing():
            sim.render()
            continue
        # 获取深度数据
        depth1 = lidarInterface.get_linear_depth_data("/World" + lidarPath1)
        depth2 = lidarInterface.get_linear_depth_data("/World" + lidarPath2)

        # 打印原始深度数据
        print("Original depth1:", depth1)
        print("Original depth2:", depth2)

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
        print("Noisy depth1:", depth1_noisy)
        print("Noisy depth2:", depth2_noisy)

        if special_action_counter>0:
            action1 = controller(drone_state, target_vel=vel_backward)
            drone.apply_action(action1)
            special_action_counter -= 1
        elif direction_change_counter > 0:
            direction_change_counter -= 1
            # 执行角度更改动作
            action2 = Att_controller(drone_state, target_yaw_rate=random_yaw, target_thrust=(drone.MASS_0 * 10.76))
            drone.apply_action(action2)

        else:


            if MAX_THRESHOLD > depth1_noisy  > MIN_THRESHOLD and MAX_THRESHOLD > depth2_noisy> MIN_THRESHOLD:
                special_action_counter = 100
                direction_change_counter = 50
                random_direction = np.random.uniform(-90, 0)
                random_direction_rad = np.deg2rad(random_direction)
                random_yaw = torch.tensor([random_direction_rad], device=sim.device)

            elif MAX_THRESHOLD > depth1_noisy > MIN_THRESHOLD and depth2_noisy < MIN_THRESHOLD:
                action = Att_controller(drone_state, target_yaw_rate=yaw_rate_right, target_thrust=(drone.MASS_0 * 10.76))
                drone.apply_action(action)
            elif MAX_THRESHOLD > depth1_noisy > MIN_THRESHOLD and depth2_noisy > MAX_THRESHOLD:
                action = Att_controller(drone_state, target_yaw_rate=yaw_rate_left, target_thrust=(drone.MASS_0 * 10.76))
                drone.apply_action(action)
            elif depth1_noisy < MIN_THRESHOLD and MAX_THRESHOLD > depth2_noisy > MIN_THRESHOLD:
                action = Att_controller(drone_state, target_yaw_rate=yaw_rate_left, target_thrust=(drone.MASS_0 * 10.76))
                drone.apply_action(action)
            elif depth1_noisy > MAX_THRESHOLD and MAX_THRESHOLD > depth2_noisy > MIN_THRESHOLD:
                action = Att_controller(drone_state, target_yaw_rate=yaw_rate_right,target_thrust=(drone.MASS_0 * 10.76))
                drone.apply_action(action)
            elif depth1_noisy > MAX_THRESHOLD and depth2_noisy > MAX_THRESHOLD :
                _, rot, _, _ = torch.split(drone_state, [3, 4, 3, 3], dim=-1)
                R = quaternion_to_rotation_matrix(rot)  # Convert quaternion to rotation matrix
                R_transpose = R.transpose(0, 1)
                forward_world = vel_forward @ R.transpose(-2, -1)
                print(forward_world)
                # 使用转换后的速度
                action = controller(drone_state, target_vel=forward_world)
                print(1)
                drone.apply_action(action)

            else :
                action = controller(drone_state, target_vel=vel_backward)
                drone.apply_action(action)

        sim.step(render=True)

        if i % 2000 == 0:
            reset()
        drone_state = drone.get_state()[..., :13].squeeze(0)
        print(drone_state)

    simulation_app.close()

if __name__ == "__main__":
    main()
