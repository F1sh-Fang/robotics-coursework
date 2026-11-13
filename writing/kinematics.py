"""6 自由度球形腕工业机器人运动学建模与解析求解模块。

单位规范：
  - 空间坐标与连杆长度：毫米 (mm)
  - 内部关节角与角速度：弧度 (rad) / 弧度每秒 (rad/s)
  - 配置与报告角度：度 (deg)

坐标系与工具说明：
  - 机器人的手腕中心点为握笔夹持点 TCP (Grip Point)。
  - 毛笔笔尖沿工具坐标系 +Z 轴延伸 tip_offset 距离。
  - R_HOLD 姿态将工具 +Z 轴映射到基座世界坐标系的 -Z 轴（笔尖严格垂直纸面向下）。
"""
from dataclasses import dataclass
import numpy as np

# 机器人连杆尺寸参数（单位：mm）
# l1: 基座到肩关节高度 (491 mm)
# l2: 大臂长度 (450 mm)
# l3: 小臂长度 (450 mm)
# l4: 腕部手腕中心到末端法兰距离 (84 mm)
LENGTHS = np.array([491., 450., 450., 84.])

# 握笔姿态旋转矩阵（固定垂直向下姿态）：
# 工具坐标系 Z 轴与世界坐标系 -Z 轴对齐，X 轴沿世界 -X，Y 轴沿世界 +Y
R_HOLD = np.diag([-1., 1., -1.])

# 各关节软限位范围（单位：度）
# 分别对应 J1 至 J6 的 [最小值, 最大值]
LIMITS_DEG = np.array([[-170, 170], [-120, 120], [-140, 140],
                       [-170, 170], [-120, 120], [-360, 360]], dtype=float)


def tip_to_grip(tip, tip_offset):
    """根据笔尖空间位置和笔长偏移量，计算手腕中心 (握笔 TCP) 的空间三维位置。

    Args:
        tip: 笔尖空间坐标 (N, 3) 或 (3,)，单位 mm
        tip_offset: 沿工具 +Z 轴的笔尖偏移量，单位 mm

    Returns:
        np.ndarray: 手腕中心握笔 TCP 坐标，单位 mm
    """
    return np.asarray(tip, dtype=float) - R_HOLD @ np.array([0., 0., tip_offset])


def forward_kinematics(q):
    """基于指数积公式 (POE, Product of Exponentials) 计算机器人的正运动学。

    机器人结构为 6 自由度垂直多关节型 (Z-Y-Y-Z-Y-Z 轴线分布)，最后三轴构成正交球形腕。

    Args:
        q: 关节角度数组，形状为 (N, 6)，单位 rad

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - position: 手腕中心/法兰在世界坐标系下的三维坐标 (N, 3)，单位 mm
            - rotation: 末端执行器在世界坐标系下的 3x3 旋转矩阵 (N, 3, 3)
    """
    q = np.atleast_2d(np.asarray(q, dtype=float))
    if q.shape[1] != 6 or not np.isfinite(q).all():
        raise ValueError("Expected finite N x 6 joint angles in radians")
    n = len(q)
    rotation = np.broadcast_to(np.eye(3), (n, 3, 3)).copy()
    position = np.zeros((n, 3))
    l1, l2, l3, l4 = LENGTHS
    # 关节轴心在基坐标系下的高度分布
    heights = [0, l1, l1+l2, l1+l2+l3, l1+l2+l3, l1+l2+l3]

    # 依次累乘 6 个关节轴的旋转指数变换
    for j, axis in enumerate(('z', 'y', 'y', 'z', 'y', 'z')):
        w = np.array([0., 0., 1.]) if axis == 'z' else np.array([0., 1., 0.])
        # 旋转向量对应的反对称矩阵 (hat operator)
        k = np.array([[0., -w[2], w[1]], [w[2], 0., -w[0]], [-w[1], w[0], 0.]])
        angle = q[:, j, None, None]
        # 罗德里格斯旋转公式 (Rodrigues' rotation formula)
        r = np.eye(3) + np.sin(angle)*k + (1-np.cos(angle))*(k@k)
        # 旋转中心不在原点时的位移偏置更新
        p = np.einsum('nij,j->ni', np.eye(3)-r, [0., 0., heights[j]])
        position += np.einsum('nij,nj->ni', rotation, p)
        rotation = rotation @ r

    # 累加腕部到末端连杆的固定轴向偏移
    position += np.einsum('nij,j->ni', rotation, [0., 0., l1+l2+l3+l4])
    # 变换到标准工具基准姿态
    return position, rotation @ np.diag([-1., -1., 1.])


def forward_tip(q, tip_offset):
    """计算含笔尖偏移量的完整正运动学。

    Args:
        q: 关节角度数组 (N, 6)，单位 rad
        tip_offset: 笔长偏移量，单位 mm

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]:
            - tip: 笔尖空间坐标 (N, 3)，单位 mm
            - grip: 手腕中心握笔坐标 (N, 3)，单位 mm
            - rotations: 末端姿态旋转矩阵 (N, 3, 3)
    """
    grip, rotations = forward_kinematics(q)
    tip = grip + rotations[:, :, 2]*tip_offset
    return tip, grip, rotations


def branch_angles(tip, tip_offset, shoulder, elbow, wrist=1):
    """针对固定书写姿态 R_HOLD 的球形腕机械臂解析逆运动学闭合解。

    通过几何解耦法求解：
      1. 笔尖位置 tip 转换为腕部中心位置 grip；
      2. 求解前三关节 (θ1, θ2, θ3) 确定腕部中心坐标；
      3. 根据末端固定垂直姿态 R_HOLD 解算后三关节 (θ4, θ5, θ6)。

    Args:
        tip: 笔尖空间坐标数组 (N, 3)，单位 mm
        tip_offset: 笔长偏移量，单位 mm
        shoulder: 肩部构型标志 (+1: 正向 / -1: 翻转反向)
        elbow: 肘部构型标志 (+1: 肘上 Elbow Up / -1: 肘下 Elbow Down)
        wrist: 腕部构型标志 (+1: 腕正 / -1: 腕翻转)

    Returns:
        np.ndarray: 连续无跳跃的关节角矩阵 (N, 6)，单位 rad
    """
    grip = np.atleast_2d(tip_to_grip(tip, tip_offset))
    l1, l2, l3, l4 = LENGTHS
    # 水平投影半径 r 与竖直相对高度 h
    r = np.hypot(grip[:, 0], grip[:, 1])
    h = grip[:, 2] + l4 - l1

    # 奇异性检测：不能穿过基座轴线原点
    if np.any(r < 1e-6):
        raise ValueError('Path crosses the base-axis singularity')

    # 余弦定理求解肘关节角 θ3
    c3 = (r*r+h*h-l2*l2-l3*l3)/(2*l2*l3)
    if np.any(np.abs(c3) > 1+1e-10):
        raise ValueError('Pen path is outside the arm reach')

    # 关节 1: 基座旋转角
    q1 = np.arctan2(grip[:, 1], grip[:, 0]) + (np.pi if shoulder < 0 else 0)
    # 关节 3: 肘部弯曲角
    q3 = elbow*np.arccos(np.clip(c3, -1, 1))
    # 关节 2: 肩部俯仰角
    q2 = np.arctan2(shoulder*r, h)-np.arctan2(l3*np.sin(q3), l2+l3*np.cos(q3))

    # 在 R_HOLD 保持垂直姿态下的腕部三轴解析配对：
    # 默认构型下，前三轴与后三轴保持对称补偿
    q4 = np.zeros(len(r))
    q5 = np.pi-q2-q3
    q6 = q1-np.pi
    if wrist < 0:
        q4 += np.pi
        q5 *= -1
        q6 += np.pi
    q = np.column_stack([q1, q2, q3, q4, q5, q6])

    # 在固定构型分支内对角度序列执行相位连续展开 (unwrap)，消除 2π 缠绕跳变
    return np.unwrap(np.arctan2(np.sin(q), np.cos(q)), axis=0)


@dataclass
class IKBranch:
    """锁定的单一无奇异逆运动学分支求解器。

    Attributes:
        shoulder: 肩部构型标志 (±1)
        elbow: 肘部构型标志 (±1)
        wrist: 腕部构型标志 (±1)
        q_start: 轨迹起点处的基准关节角 (6,)，单位 rad
        limits: 关节角限制矩阵 (6, 2)，单位 rad
        tip_offset: 笔长偏移量，单位 mm
    """
    shoulder: int
    elbow: int
    wrist: int
    q_start: np.ndarray
    limits: np.ndarray
    tip_offset: float

    def solve(self, tip, previous=None):
        """对给定的一组三维点计算逆解，并保持与上一时刻姿态的连续无跳跃过渡。

        Args:
            tip: 笔尖空间坐标序列 (N, 3)，单位 mm
            previous: 参考上一时刻关节角 (6,)，缺省时使用 q_start

        Returns:
            np.ndarray: 连续、严格满足限位的关节角矩阵 (N, 6)，单位 rad
        """
        q = branch_angles(tip, self.tip_offset, self.shoulder, self.elbow, self.wrist)
        reference = self.q_start if previous is None else previous
        # 整体平移 2kπ，使得当前段起点最贴近 reference，消除周期性多解跳跃
        q += 2*np.pi*np.round((reference-q[0])/(2*np.pi))

        # 限位检查
        if np.any(q < self.limits[:, 0]-1e-10) or np.any(q > self.limits[:, 1]+1e-10):
            raise ValueError('Selected IK branch exceeds joint limits')
        # 奇异点邻域检测（避免 sin(q3)=0 肘部伸直/折叠，以及 sin(q5)=0 腕部奇异）
        if np.min(np.abs(np.sin(q[:, 2]))) < 1e-3 or np.min(np.abs(np.sin(q[:, 4]))) < 1e-3:
            raise ValueError('Path approaches an elbow/wrist singularity; change placement')
        # 相邻采样点连续性突变检查
        if len(q) > 1 and np.max(np.abs(np.diff(q, axis=0))) > np.pi/2:
            raise ValueError('Discontinuous IK branch; refusing joint interpolation')
        return q


def select_branch(tip, tip_offset, limits_deg=LIMITS_DEG, seed_deg=None):
    """预先扫描整条轨迹的所有点，在全部 8 种逆解构型中选择全局最优且合法的单一分支。

    评选标准：
      1. 该分支必须在整条轨迹上全程连续、无任何构型跳变；
      2. 全程远离奇异点且严格在限位内；
      3. 综合考量与初始参考姿态 seed 的接近程度，以及距离限位边界的裕度（安全边界惩罚项）。

    Args:
        tip: 稠密采样后的整条书写笔尖轨迹 (N, 3)，单位 mm
        tip_offset: 笔长偏移量，单位 mm
        limits_deg: 六关节限位 (6, 2)，单位 deg
        seed_deg: 起点首选角度参考值 (6,)，单位 deg

    Returns:
        IKBranch: 锁定的全局最优逆解分支实例
    """
    limits = np.deg2rad(np.asarray(limits_deg, dtype=float))
    seed = np.zeros(6) if seed_deg is None else np.deg2rad(seed_deg)
    if seed.shape != (6,) or not np.isfinite(seed).all():
        raise ValueError('IK seed must contain six finite joint angles')
    candidates = []

    # 遍历 shoulder (±1), elbow (±1), wrist (±1) 共 8 种几何构型
    for shoulder in (1, -1):
        for elbow in (1, -1):
            for wrist in (1, -1):
                try:
                    q = branch_angles(tip, tip_offset, shoulder, elbow, wrist)
                    # 检查各轴能否在 [-limit, +limit] 范围内找到合法 2kπ 整数偏置
                    for j in range(6):
                        low = int(np.ceil((limits[j, 0]-q[:, j].min())/(2*np.pi)-1e-12))
                        high = int(np.floor((limits[j, 1]-q[:, j].max())/(2*np.pi)+1e-12))
                        if low > high:
                            raise ValueError('No legal full-path angle representation')
                        turn = int(np.clip(np.round((seed[j]-q[0, j])/(2*np.pi)), low, high))
                        q[:, j] += 2*np.pi*turn
                    branch = IKBranch(shoulder, elbow, wrist, q[0], limits, tip_offset)
                    q = branch.solve(tip)
                    # 计算各关节距离限位边界的最小间隙裕度
                    margin = min(np.min(q-limits[:, 0]), np.min(limits[:, 1]-q))
                    # 目标函数：起点偏离代价 + 边界惩罚项
                    cost = np.sum((q[0]-seed)**2) + 0.05/(margin+0.01)
                    candidates.append((cost, branch))
                except ValueError:
                    continue
    if not candidates:
        raise ValueError('No single nonsingular IK branch covers the whole pen path within limits')
    # 返回代价值最小的构型分支
    return min(candidates, key=lambda item: item[0])[1]

