import numpy as np
from scipy.spatial import KDTree

class Node:
    def __init__(self, position, yaw):
        self.position = np.array(position)
        self.yaw = yaw
        self.parent = None
        self.cost = 0.0
        self.path_length = 0.0
        self.info_gain = 0.0

class RRTStarPlanner:
    def __init__(self, start, goal, start_yaw, goal_yaw, uncertainty_grid,
                 alpha=1.0, beta=3.0, step_size=0.5, max_yaw_change=np.pi/6,
                 goal_tolerance=0.5, yaw_tolerance=np.pi/12,
                 search_radius=1.5, max_iter=1000,
                 x_range=(-4, 4), y_range=(-4, 4)):
        
        self.start = Node(start, start_yaw)
        self.goal = Node(goal, goal_yaw)
        self.uncertainty_grid = np.clip(uncertainty_grid, 0, None)
        self.grid_size = uncertainty_grid.shape[0]
        self.x_range = x_range
        self.y_range = y_range
        
        # 算法参数
        self.alpha = alpha  # 路径长度权重
        self.beta = beta    # 信息增益权重
        self.step_size = step_size
        self.max_yaw_change = max_yaw_change  # 最大单步yaw变化
        self.goal_tolerance = goal_tolerance
        self.yaw_tolerance = yaw_tolerance
        self.search_radius = search_radius
        self.max_iter = max_iter
        
        self.nodes = [self.start]
        
    def sample(self, goal_bias=0.1):
        # 目标偏向采样
        if np.random.rand() < goal_bias:
            return self.goal.position.copy(), self.goal.yaw
        
        # 在不确定性高的区域增加采样概率
        if np.random.rand() < 0.3:  # 30%概率在高不确定性区域采样
            max_uncertainty = np.max(self.uncertainty_grid)
            if max_uncertainty > 0:
                prob = self.uncertainty_grid / np.sum(self.uncertainty_grid)
                idx = np.random.choice(self.grid_size*self.grid_size, p=prob.flatten())
                gy, gx = np.unravel_index(idx, self.uncertainty_grid.shape)
                x = self.x_range[0] + (gx / (self.grid_size-1)) * (self.x_range[1]-self.x_range[0])
                y = self.y_range[0] + (gy / (self.grid_size-1)) * (self.y_range[1]-self.y_range[0])
                return np.array([x, y]), np.random.uniform(-np.pi, np.pi)
        
        # 常规随机采样
        x = np.random.uniform(*self.x_range)
        y = np.random.uniform(*self.y_range)
        yaw = np.random.uniform(-np.pi, np.pi)
        return np.array([x, y]), yaw
    
    def nearest(self, point):
        positions = [node.position for node in self.nodes]
        tree = KDTree(positions)
        _, idx = tree.query(point)
        return self.nodes[idx]
    
    def steer(self, from_node, to_pos, to_yaw):
        direction = to_pos - from_node.position
        distance = np.linalg.norm(direction)
        if distance < 1e-6:
            return None
            
        # 计算期望的yaw角
        desired_yaw = np.arctan2(direction[1], direction[0])
        
        # 限制yaw变化幅度
        yaw_diff = desired_yaw - from_node.yaw
        yaw_diff = np.arctan2(np.sin(yaw_diff), np.cos(yaw_diff))  # 归一化到[-pi, pi]
        
        # 限制单步最大yaw变化
        yaw_step = np.clip(yaw_diff, -self.max_yaw_change, self.max_yaw_change)
        new_yaw = from_node.yaw + yaw_step
        
        # 计算新位置
        step = min(self.step_size, distance)
        new_pos = from_node.position + step * np.array([np.cos(new_yaw), np.sin(new_yaw)])
        
        # 创建新节点
        new_node = Node(new_pos, new_yaw)
        new_node.parent = from_node
        
        # 计算路径长度和信息增益
        segment_length = np.linalg.norm(new_pos - from_node.position)
        new_node.path_length = from_node.path_length + segment_length
        new_node.info_gain = from_node.info_gain + self.calculate_info_gain(from_node.position, new_pos)
        
        # 综合成本计算
        new_node.cost = self.alpha * new_node.path_length - self.beta * new_node.info_gain
        
        return new_node
    
    def calculate_info_gain(self, p1, p2):
        # 沿路径线段采样计算信息增益
        num_samples = max(2, int(np.linalg.norm(p2 - p1) / 0.1))
        line = np.linspace(p1, p2, num_samples)
        total_info = 0.0
        
        for pt in line:
            gx, gy = self.position_to_grid_index(pt)
            total_info += self.uncertainty_grid[gy, gx]
            
        return total_info / num_samples
    
    def position_to_grid_index(self, position):
        x = np.clip(position[0], *self.x_range)
        y = np.clip(position[1], *self.y_range)
        gx = int((x - self.x_range[0]) / (self.x_range[1] - self.x_range[0]) * (self.grid_size - 1))
        gy = int((y - self.y_range[0]) / (self.y_range[1] - self.y_range[0]) * (self.grid_size - 1))
        return np.clip(gx, 0, self.grid_size-1), np.clip(gy, 0, self.grid_size-1)
    
    def near(self, new_node):
        positions = [node.position for node in self.nodes]
        tree = KDTree(positions)
        indices = tree.query_ball_point(new_node.position, self.search_radius)
        return [self.nodes[i] for i in indices]
    
    def rewire(self, new_node, neighbors):
        for neighbor in neighbors:
            # 计算通过新节点到达邻居的成本
            segment_length = np.linalg.norm(neighbor.position - new_node.position)
            if segment_length < 1e-6:
                continue
                
            new_path_length = new_node.path_length + segment_length
            new_info_gain = new_node.info_gain + self.calculate_info_gain(new_node.position, neighbor.position)
            new_cost = self.alpha * new_path_length - self.beta * new_info_gain
            
            # 如果新路径更好，则重连
            if new_cost < neighbor.cost:
                # 检查yaw变化是否平滑
                yaw_diff = np.arctan2(np.sin(new_node.yaw - neighbor.yaw), 
                                     np.cos(new_node.yaw - neighbor.yaw))
                if abs(yaw_diff) < self.max_yaw_change * 1.5:  # 允许稍大的yaw变化
                    neighbor.parent = new_node
                    neighbor.path_length = new_path_length
                    neighbor.info_gain = new_info_gain
                    neighbor.cost = new_cost
    
    def plan(self):
        best_goal_node = None
        best_cost = float('inf')
        
        for iteration in range(self.max_iter):
            # 定期检查是否应该提前终止
            if iteration % 100 == 0 and best_goal_node is not None:
                print(f"Iteration {iteration}, best cost so far: {best_cost:.2f}")
                # 如果已经找到不错的结果，可以提前终止
                if iteration > self.max_iter // 2 and best_cost < 0:  # 根据你的成本函数调整
                    print("Early termination with good enough path")
                    return self.extract_path(best_goal_node)
            
            # 采样新点
            rand_pos, rand_yaw = self.sample(goal_bias=0.2)
            nearest_node = self.nearest(rand_pos)
            new_node = self.steer(nearest_node, rand_pos, rand_yaw)
            
            if new_node is None or not self.in_bounds(new_node.position):
                continue
                
            # 限制节点数量以避免内存爆炸
            if len(self.nodes) > 5000:  # 根据你的内存调整
                self.nodes = sorted(self.nodes, key=lambda n: n.cost)[:4000]  # 保留成本最低的4000个节点
                print("Pruned nodes to avoid memory overflow")
            
            # 寻找附近节点并重连
            neighbors = self.near(new_node)
            self.nodes.append(new_node)
            
            # 优化重连操作，限制邻居数量
            if len(neighbors) > 20:  # 如果邻居太多，只考虑最近的20个
                neighbors = sorted(neighbors, key=lambda n: np.linalg.norm(n.position - new_node.position))[:20]
            
            self.rewire(new_node, neighbors)
            
            # 检查是否到达目标并记录最佳候选
            if self.reached_goal(new_node):
                if new_node.cost < best_cost:
                    best_goal_node = new_node
                    best_cost = new_node.cost
                    print(f"New best goal node found at iteration {iteration}, cost: {best_cost:.2f}")
        
        # 最终检查是否有接近目标的节点
        if best_goal_node is not None:
            print(f"Reached max iterations, returning best path (cost: {best_cost:.2f})")
            return self.extract_path(best_goal_node)
        
        # 如果没有精确到达目标的节点，寻找接近的节点
        print("No exact goal node found, looking for close candidates...")
        close_nodes = []
        for node in self.nodes:
            pos_distance = np.linalg.norm(node.position - self.goal.position)
            yaw_diff = np.abs(np.arctan2(np.sin(node.yaw - self.goal.yaw), 
                            np.cos(node.yaw - self.goal.yaw)))
            if pos_distance < self.goal_tolerance * 2:  # 放宽位置条件
                close_nodes.append((node, pos_distance, yaw_diff))
        
        if close_nodes:
            # 按综合条件排序：先位置距离，再yaw差异，最后成本
            close_nodes.sort(key=lambda x: (x[1], x[2], x[0].cost))
            best_close_node = close_nodes[0][0]
            print(f"Found close node at distance {close_nodes[0][1]:.2f}, yaw diff {np.degrees(close_nodes[0][2]):.1f}°")
            return self.extract_path(best_close_node)
        
        print("Failed to find a path to the goal")
        return None
    
    def in_bounds(self, position):
        return (self.x_range[0] <= position[0] <= self.x_range[1] and 
                self.y_range[0] <= position[1] <= self.y_range[1])
    
    def reached_goal(self, node):
        pos_distance = np.linalg.norm(node.position - self.goal.position)
        yaw_diff = np.abs(np.arctan2(np.sin(node.yaw - self.goal.yaw), 
                                   np.cos(node.yaw - self.goal.yaw)))
        return pos_distance <= self.goal_tolerance and yaw_diff <= self.yaw_tolerance
    
    def extract_path(self, goal_node):
        path = []
        current_node = goal_node
        visited_nodes = set()  # 用于检测循环
        
        max_steps = len(self.nodes) * 2  # 安全限制
        step = 0
        
        while current_node and step < max_steps:
            if id(current_node) in visited_nodes:
                print("Warning: Cycle detected in path!")
                break
                
            visited_nodes.add(id(current_node))
            path.append({
                'position': current_node.position.copy(),
                'yaw': current_node.yaw,
                'info_gain': current_node.info_gain,
                'path_length': current_node.path_length
            })
            current_node = current_node.parent
            step += 1
        
        if step >= max_steps:
            print("Warning: Path extraction reached maximum steps")
        
        return path[::-1]  # 反转路径
    
import numpy as np
import casadi as cs
from scipy.interpolate import RegularGridInterpolator

class InputProcessor:
    def __init__(self, uncertainty_map):
        """
        :param uncertainty_map: 3D numpy数组 (x,y,z)->uncertainty
        """
        self.uncertainty_interp = RegularGridInterpolator(
            points=[np.arange(d) for d in uncertainty_map.shape],
            values=uncertainty_map,
            method='linear'
        )
    
    def get_uncertainty_at(self, pos):
        """安全获取位置处的不确定性值（自动处理越界点）"""
        # 检查坐标是否越界
        if (pos[0] < -5 or pos[0] > 5 or 
            pos[1] < -5 or pos[1] > 5 or 
            pos[2] < 0 or pos[2] > 10):
            return 0.0  # 越界点返回默认值（或抛出警告）
        return self.uncertainty_interp(pos)
    
    def compute_gradient(self, pos, delta=0.1):
        """数值计算不确定性梯度"""
        dx = np.array([delta, 0, 0])
        dy = np.array([0, delta, 0])
        dz = np.array([0, 0, delta])
        grad = np.array([
            self.get_uncertainty_at(pos + dx) - self.get_uncertainty_at(pos - dx),
            self.get_uncertainty_at(pos + dy) - self.get_uncertainty_at(pos - dy),
            self.get_uncertainty_at(pos + dz) - self.get_uncertainty_at(pos - dz)
        ]) / (2 * delta)
        return grad
    
class TrajectoryOptimizer:
    def __init__(self):
        self.opti = cs.Opti()
        self.N = 7  # 多项式次数
        self.T = self.opti.variable()  # 轨迹时间
        
    def setup_problem(self, start_pos, start_yaw, target_pos, target_normal):
        # ===== 优化变量 =====
        # 4个维度(x,y,z,yaw)的(N+1)次多项式系数
        poly_coeffs = self.opti.variable((self.N+1)*4)  
        
        # ===== 参数输入 =====
        start_vel = self.opti.parameter(3)  # 初始速度（可测或置零）
        uncertainty_weights = self.opti.parameter(self.N+1)  # 路径点不确定性权重
        
        # ===== 轨迹定义 =====
        def get_trajectory(t, dim):
            """获取t时刻某维度的轨迹值 (dim: 0=x,1=y,2=z,3=yaw)"""
            coeffs = poly_coeffs[dim*(self.N+1): (dim+1)*(self.N+1)]
            return cs.polyval(cs.MX(coeffs), t/self.T)
        
        # ===== 硬约束 =====
        # 1. 初始状态约束
        self.opti.subject_to(get_trajectory(0, 0) == start_pos[0])  # x
        self.opti.subject_to(cs.polyder(get_trajectory(cs.MX.sym('t'), 0))(0) == start_vel[0])
        # ... (y,z,yaw同理)
        
        # 2. 终端约束
        self.opti.subject_to(get_trajectory(1, 0) == target_pos[0])  # 位置到达
        # 速度方向与法向量对齐
        end_vel = cs.vertcat(
            cs.polyder(get_trajectory(cs.MX.sym('t'), 0))(1),
            cs.polyder(get_trajectory(cs.MX.sym('t'), 1))(1),
            cs.polyder(get_trajectory(cs.MX.sym('t'), 2))(1)
        )
        self.opti.subject_to(end_vel == -0.3 * target_normal)  # 系数控制接近速度
        
        # ===== 目标函数 =====
        # 1. 平滑性 (Minimum Snap)
        snap_cost = 0
        for dim in range(3):  # x,y,z
            snap = cs.polyder(get_trajectory(cs.MX.sym('t'), dim), 4)
            snap_cost += cs.integrator('cost', 'cvodes', {'x': cs.MX.sym('x'), 'ode': snap**2}, 
                                     {'t0':0, 'tf':1}).res['xf']
        
        # 2. 信息增益 (沿路径积分不确定性)
        info_cost = 0
        for i in range(self.N+1):
            t = i/self.N
            pos = cs.vertcat(get_trajectory(t,0), get_trajectory(t,1), get_trajectory(t,2))
            info_cost += uncertainty_weights[i] * cs.norm_2(pos)**2  # 权重来自预处理
        
        # 3. 时间惩罚 (避免过慢)
        time_cost = self.T
        
        # 总目标函数
        total_cost = 0.4*snap_cost + 0.5*info_cost + 0.1*time_cost
        self.opti.minimize(total_cost)
        
        # 返回需要外部设置的参数句柄
        return {
            'coeffs': poly_coeffs,
            'start_vel': start_vel,
            'uncertainty_weights': uncertainty_weights
        }

    def solve(self, params, max_time=0.5):
        """求解优化问题"""
        self.opti.solver('ipopt', {
            'ipopt.max_iter': 1000,
            'ipopt.max_cpu_time': max_time,
            'ipopt.print_level': 0
        })
        try:
            sol = self.opti.solve()
            return {
                'coeffs': sol.value(self.opti.variables()),
                'T': sol.value(self.T),
                'success': True
            }
        except:
            return {'success': False}
        
class TactileExplorationPlanner:
    def __init__(self, uncertainty_map):
        self.input_processor = InputProcessor(uncertainty_map)
        self.optimizer = TrajectoryOptimizer()
        self.params = None
        
    def plan(self, start_pos, start_yaw, target_pos, target_normal):
        # 1. 预处理不确定性权重
        path_guess = self._generate_initial_path(start_pos, target_pos)
        uncertainty_weights = np.array([self.input_processor.get_uncertainty_at(p) for p in path_guess])
        
        # 2. 设置优化问题
        if self.params is None:  # 首次初始化
            self.params = self.optimizer.setup_problem(start_pos, start_yaw, target_pos, target_normal)
        
        # 3. 设置参数值
        self.optimizer.opti.set_value(self.params['start_vel'], np.zeros(3))  # 假设零初始速度
        self.optimizer.opti.set_value(self.params['uncertainty_weights'], uncertainty_weights)
        
        # 4. 求解
        result = self.optimizer.solve(self.params)
        if not result['success']:
            return self._fallback_plan(start_pos, target_pos)
            
        # 5. 返回轨迹参数
        return {
            'coeffs': result['coeffs'].reshape(4, -1),  # 4×(N+1)矩阵
            'duration': result['T']
        }
    
    def _generate_initial_path(self, start, end, num=10):
        """生成直线初始路径，并限制在地图范围内"""
        path = np.linspace(start, end, num)
        # 限制坐标在网格范围内
        path[:, 0] = np.clip(path[:, 0], -5, 5)  # x ∈ [-5,5]
        path[:, 1] = np.clip(path[:, 1], -5, 5)  # y ∈ [-5,5]
        path[:, 2] = np.clip(path[:, 2], 0, 10)  # z ∈ [0,10]
        return path
    
    def _fallback_plan(self, start, end):
        """应急策略：直线轨迹+减速"""
        coeffs = np.zeros((4, self.optimizer.N+1))
        coeffs[0,0] = start[0]  # x: c0
        coeffs[0,1] = end[0] - start[0]  # x: c1*t
        # ... (y,z同理)
        return {'coeffs': coeffs, 'duration': 5.0}  # 保守低速
    
# def execute_plan(drone, plan):
#     """将多项式轨迹转换为无人机控制指令"""
#     from scipy.interpolate import CubicHermiteSpline
    
#     # 1. 采样轨迹
#     t_samples = np.linspace(0, plan['duration'], 100)
#     pos = np.array([def execute_plan(drone, plan):
#     """将多项式轨迹转换为无人机控制指令"""
#     from scipy.interpolate import CubicHermiteSpline
    
#     # 1. 采样轨迹
#     t_samples = np.linspace(0, plan['duration'], 100)
#     pos = np.array([
#         np.polyval(plan['coeffs'][i][::-1], t/plan['duration'])  # 注意系数顺序
#         for t in t_samples for i in range(3)
#     ]).T
    
#     # 2. 生成速度指令
#     spline = CubicHermiteSpline(t_samples, pos, 
#                                drone.estimate_velocity(pos))
    
#     # 3. 发送指令
#     for t in np.arange(0, plan['duration'], 0.1):
#         drone.send_velocity_command(spline(t))
        
#         if drone.detect_contact():
#             update_uncertainty_map(drone.get_contact_position())
#             return 'CONTACT_DETECTED'
    
#     return 'SUCCESS'
#         np.polyval(plan['coeffs'][i][::-1], t/plan['duration'])  # 注意系数顺序
#         for t in t_samples for i in range(3)
#     ]).T
    
#     # 2. 生成速度指令
#     spline = CubicHermiteSpline(t_samples, pos, 
#                                drone.estimate_velocity(pos))
    
#     # 3. 发送指令
#     for t in np.arange(0, plan['duration'], 0.1):
#         drone.send_velocity_command(spline(t))
        
#         if drone.detect_contact():
#             update_uncertainty_map(drone.get_contact_position())
#             return 'CONTACT_DETECTED'
    
#     return 'SUCCESS'