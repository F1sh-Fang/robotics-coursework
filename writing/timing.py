"""第二/三层：末端动力学时间规划、局部自适应时间律与锁定分支逆解 (Layers 2 & 3)。

核心算法与流程：
  1. 七次多项式时间缩放 (7th-order polynomial): 起终点速度、加速度、加加速度均为零 ($C^3$ 连续)；
  2. 局部自适应时间密度重分配 (LocalTimeLaw): 针对长笔画及复杂曲率段，自适应动态分配时间，
     使平缓段提速、高曲率段慢行，兼顾书写效率与平稳性，彻底解决长笔画与起步停滞问题；
  3. 综合动力学时间拉伸 (Time Stretch): 严格限制末端 Cartesian 空间及 6 个关节轴的速度、加速度、加加速度；
  4. 1 ms 严格离散采样与整段平滑拼接；
  5. 全闭环正运动学 (FK) 与动力学校验报告生成。
"""
from dataclasses import dataclass, field
import math
import numpy as np
from scipy.interpolate import CubicSpline, PchipInterpolator
from scipy.ndimage import gaussian_filter1d, maximum_filter1d

from .geometry import ArcCurve, PHASES, validate_geometry
from .kinematics import LIMITS_DEG, R_HOLD, forward_tip, select_branch

# 系统固定控制周期与时间参数（单位：s）
DT = 0.001                 # 控制周期 1 ms (1000 Hz)
MIN_SEGMENT_DURATION = 0.5  # 单段最小持续时长 0.5 s (避免极短笔画造成加速度超调)
TIMING_RESERVE = 1.05       # 5% 的保守安全动力学时间裕度


@dataclass
class MotionLimits:
    """机械臂运动学与动力学约束参数容器。

    Attributes:
        write_speed: 纸面书写阶段 (DRAW) 最大线速度上限 (mm/s)
        travel_speed: 空中平移阶段 (TRANSFER) 最大线速度上限 (mm/s)
        vertical_speed: 垂直升降阶段 (DESCEND/LIFT) 最大线速度上限 (mm/s)
        cart_accel: 笛卡尔末端最大加速度上限 (mm/s²)
        cart_jerk: 笛卡尔末端最大加加速度上限 (mm/s³)
        joint_speed_deg: 各关节最大角速度限制 (deg/s)
        joint_accel_deg: 各关节最大角加速度限制 (deg/s²)
        joint_jerk_deg: 各关节最大角加加速度限制 (deg/s³)
        joint_limits_deg: 各关节软限位范围 (6, 2) (deg)
        max_duration: 整条轨迹允许的最大超时时间 (s)
    """
    write_speed: float = 20.
    travel_speed: float = 40.
    vertical_speed: float = 15.
    cart_accel: float = 100.
    cart_jerk: float = 500.
    joint_speed_deg: np.ndarray = field(default_factory=lambda: np.full(6, 45.))
    joint_accel_deg: np.ndarray = field(default_factory=lambda: np.full(6, 180.))
    joint_jerk_deg: np.ndarray = field(default_factory=lambda: np.full(6, 1000.))
    joint_limits_deg: np.ndarray = field(default_factory=lambda: LIMITS_DEG.copy())
    max_duration: float = 1200.

    def validate(self):
        """严格校验各项运动限值是否为有限正数以及数组维度是否合法。"""
        values = [self.write_speed, self.travel_speed, self.vertical_speed,
                  self.cart_accel, self.cart_jerk, self.max_duration]
        for name in ('joint_speed_deg', 'joint_accel_deg', 'joint_jerk_deg'):
            a = np.asarray(getattr(self, name), dtype=float)
            if a.shape != (6,) or not np.isfinite(a).all() or np.any(a <= 0):
                raise ValueError(f'{name} must contain six finite positive limits')
            setattr(self, name, a)
        if not np.isfinite(values).all() or min(values) <= 0:
            raise ValueError('Motion limits must be finite and positive')
        a = np.asarray(self.joint_limits_deg, dtype=float)
        if a.shape != (6, 2) or not np.isfinite(a).all() or np.any(a[:, 0] >= a[:, 1]):
            raise ValueError('Joint position limits must be a finite 6 x 2 matrix')

    def phase_speed(self, phase):
        """获取指定动作阶段所允许的最大线速度上限。"""
        return self.write_speed if phase == 'DRAW' else self.travel_speed if phase == 'TRANSFER' else self.vertical_speed


def seventh_order(tau):
    """归一化七次单调多项式时间缩放曲线及其 1~3 阶导数 (C^3 连续)。

    多项式形式：
      s(τ) = 35 τ⁴ - 84 τ⁵ + 70 τ⁶ - 20 τ⁷,  τ ∈ [0, 1]
    导数性质：
      一阶导 (速度因子):      s'(τ) = 140 τ³ (1-τ)³
      二阶导 (加速度因子):    s''(τ) = 420 τ² (1-τ)² (1 - 2τ)
      三阶导 (加加速度因子):  s'''(τ) = 840 τ (1-τ) (1 - 5τ + 5τ²)

    边界条件：
      在 τ=0 和 τ=1 处，s(0)=0, s(1)=1，且 s', s'', s''' 均严格恒等于 0，
      保证机器人从静止平滑启动并平稳停止，无加速度冲击与冲击振动。

    Args:
        tau: 归一化时间标量或数组，取值范围 [0, 1]

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            归一化位移 s, 速度因子 v, 加速度因子 a, 加加速度因子 j
    """
    t = np.clip(np.asarray(tau, dtype=float), 0., 1.)
    # 利用关于 τ=0.5 的对称性减少浮点截断误差
    u = np.minimum(t, 1-t)
    small = u**4*(35+u*(-84+u*(70-20*u)))
    s = np.where(t <= .5, small, 1-small)
    v = 140*t**3*(1-t)**3
    a = 420*t**2*(1-t)**2*(1-2*t)
    j = 840*t*(1-t)*(1-5*t+5*t*t)
    return s, v, a, j


def evaluate_curve(arc, tau, duration, time_law=None):
    """结合时间律与空间弧长几何导数，计算笛卡尔三维空间的位置、速度、加速度和加加速度。

    微分几何链式法则公式：
      p(t) = C(s(t))
      v(t) = dp/dt = p'(s) · ṡ
      a(t) = d²p/dt² = p'(s) · s̈ + p''(s) · ṡ²
      j(t) = d³p/dt³ = p'(s) · s⃛ + 3 p''(s) · (ṡ · s̈) + p'''(s) · ṡ³
      其中 p', p'', p''' 为空间关于弧长 s 的几何导数。

    Args:
        arc: ArcCurve 弧长参数化几何对象
        tau: 归一化时间序列 τ ∈ [0, 1]
        duration: 本段实际物理耗时 T (s)
        time_law: 可选的高级局部自适应时间律对象 (LocalTimeLaw)，缺省则采用经典七次多项式

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            空间位置 p, 速度 velocity, 加速度 acceleration, 加加速度 jerk
    """
    s, sd, sdd, sddd = seventh_order(tau) if time_law is None else time_law.at(tau)
    # 按实际段长 L 与实际时长 T 缩放物理时间导数
    s, sd, sdd, sddd = (s*arc.length, sd*arc.length/duration,
                         sdd*arc.length/duration**2, sddd*arc.length/duration**3)
    p, ps, pss, psss = arc.at(s, derivatives=True)
    velocity = ps*sd[:, None]
    acceleration = ps*sdd[:, None]+pss*sd[:, None]**2
    jerk = ps*sddd[:, None]+3*pss*(sd*sdd)[:, None]+psss*sd[:, None]**3
    return p, velocity, acceleration, jerk


class LocalTimeLaw:
    """局部自适应时间密度重分配时间律 (Local Time Law)。

    设计动机：
      在标准七次多项式下，整段笔画采用统一的时间缩放因子。当笔画较长或局部曲率较大时，
      为了满足最严苛点的动力学约束，整条曲线都被迫大幅拉伸变慢，导致平直段和起笔段速度极低。
      本类通过分析沿曲线的动力学负担分布，自适应分配时间：
        - 难走的高曲率段/大加加速度段：分配较多时间（慢行）；
        - 平缓顺直段：分配较少时间（提速行笔）；
        - 同时严格保持在起点与终点处速度、加速度、加加速度为 0。

    数学实现：
      1. 根据速度、加速度、加加速度计算综合负荷权重 weight(u)；
      2. 经极大值滤波与高斯平滑构造连续时间密度函数 w(u)；
      3. 计算时间积分与反函数映射 u(τ)；
      4. 严格利用复合函数三阶链式法则计算 s(τ) 及其 1~3 阶导数。
    """
    def __init__(self, arc, limits):
        grid = np.linspace(0., 1., 2001)
        _, velocity, acceleration, jerk = evaluate_curve(arc, grid, 1.)
        # 计算整条曲线各点在各项动力学限值下的相对负担
        weight = np.maximum.reduce([
            np.linalg.norm(velocity, axis=1) / limits.write_speed,
            np.sqrt(np.linalg.norm(acceleration, axis=1) / limits.cart_accel),
            np.cbrt(np.linalg.norm(jerk, axis=1) / limits.cart_jerk),
        ])
        # 截断下界并进行高斯滤波平滑，避免时间密度突变引入额外的 jerk 冲击
        weight = np.maximum(weight, weight.max() * 0.1)
        weight = gaussian_filter1d(maximum_filter1d(weight, size=81), sigma=40.)
        density = CubicSpline(grid, weight)
        integral = density.antiderivative()
        elapsed = integral(grid) - integral(0.)
        # 归一化时间密度与反函数插值映射
        self.density = CubicSpline(grid, weight / elapsed[-1])
        self.integral = self.density.antiderivative()
        self.inverse = PchipInterpolator(elapsed / elapsed[-1], grid)

    def at(self, tau):
        """计算给定归一化时间 τ 处的位移及 1~3 阶时间导数。

        Args:
            tau: 归一化时间标量或数组 τ ∈ [0, 1]

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
                局部重分配后的 s(τ), s'(τ), s''(τ), s'''(τ)
        """
        tau = np.clip(np.asarray(tau, dtype=float), 0., 1.)
        u = self.inverse(tau)
        # 牛顿迭代精修反函数精度
        for _ in range(3):
            residual = self.integral(u) - self.integral(0.) - tau
            u = np.clip(u - residual / self.density(u), 0., 1.)
        s, v, a, j = seventh_order(u)
        w, dw, ddw = self.density(u), self.density(u, 1), self.density(u, 2)
        # 复合函数链式法则 (u 对 τ 的 1~3 阶导数变换):
        # du/dτ = 1/w
        # d²u/dτ² = -w' / w³
        # d³u/dτ³ = (3 w'² - w w'') / w⁵
        return (s, v / w, a / w**2 - v * dw / w**3,
                j / w**3 - 3 * a * dw / w**4 - v * ddw / w**4
                + 3 * v * dw**2 / w**5)


def numerical_derivatives(q, step):
    """采用二阶精度中心差分法计算关节角度的时间导数（角速度、角加速度、角加加速度）。

    仅用于动力学校验与诊断，绝不用于插值或修改原始几何。

    Args:
        q: 关节角度序列 (N, 6)，单位 rad
        step: 采样时间步长 (s)

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]:
            - v: 关节角速度 (N, 6)，单位 rad/s
            - a: 关节角加速度 (N, 6)，单位 rad/s²
            - j: 关节角加加速度 (N, 6)，单位 rad/s³
    """
    v = np.gradient(q, step, axis=0, edge_order=2)
    a = np.gradient(v, step, axis=0, edge_order=2)
    j = np.gradient(a, step, axis=0, edge_order=2)
    return v, a, j


def peak_norm(a):
    """计算向量序列沿时间轴的最大欧氏模长。"""
    return float(np.linalg.norm(a, axis=1).max())


def required_stretch(cart_derivatives, joint_derivatives, limits, phase):
    """根据当前段笛卡尔与关节空间的各阶导数，计算满足全部物理限值所需的时间放大倍数。

    对于 n 阶导数（速度 n=1，加速度 n=2，加加速度 n=3），时间拉伸因子比例为：
      scale = (peak_value / limit_value) ** (1 / n)
    总拉伸因子取所有指标中的最大值。

    Args:
        cart_derivatives: 笛卡尔末端导数三元组 (v, a, j)
        joint_derivatives: 六关节导数三元组 (qd_v, qd_a, qd_j)
        limits: MotionLimits 动力学限值对象
        phase: 动作阶段 ('DRAW' / 'TRANSFER' / 'DESCEND' / 'LIFT')

    Returns:
        float: 综合最小无超限所需的时间放大系数
    """
    requirements = [0.]
    # 笛卡尔末端线速度、加速度、加加速度检验
    for order, data, limit in zip((1, 2, 3), cart_derivatives,
                                  (limits.phase_speed(phase), limits.cart_accel, limits.cart_jerk)):
        requirements.append((peak_norm(data)/limit)**(1/order))
    # 6 个关节轴的角速度、角加速度、角加加速度检验
    for order, data, limit in zip((1, 2, 3), joint_derivatives,
                                  (limits.joint_speed_deg, limits.joint_accel_deg, limits.joint_jerk_deg)):
        requirements.append(float(np.max(np.abs(data)/np.deg2rad(limit)))**(1/order))
    return max(requirements)


def plan_trajectory(geometry, limits=None, seed_deg=None, progress=None, *, adaptive_drawing=True):
    """执行分层动力学时间规划与逆解全流程。

    规划流程：
      1. 提取所有段的几何样条，并在整条路径上执行预扫描，锁定唯一无奇异、无跳跃的逆解构型分支；
      2. 对每一段执行预试探（Dense tau check），根据动力学限值计算安全时长；
      3. 若为书写段 (DRAW) 且启用了自适应局部时间律，尝试采用 LocalTimeLaw 优化时长；
      4. 将时长精确取整至 1 ms (DT) 整数倍，迭代检验实际离散序列直至所有动力学指标严格不超限；
      5. 拼接各段轨迹并执行严格的全闭环验证。

    Args:
        geometry: 第一层构建的三维几何数据
        limits: 动力学限制参数 (缺省则采用默认 MotionLimits)
        seed_deg: 初始姿态首选逆解种子角度 (deg)
        progress: 进度回调函数 fn(current_index, total_count, phase, duration)
        adaptive_drawing: 是否在书写段启用局部自适应时间优化 (默认 True)

    Returns:
        tuple[dict, dict]:
            - trajectory: 包含完整 1 ms 离散点数组的字典 (time, q, tip, grip, ...)
            - report: 包含逆解构型、峰值指标和验证数据的详细报告字典
    """
    validate_geometry(geometry)
    limits = limits or MotionLimits()
    limits.validate()
    arcs = [ArcCurve(s['curve']) for s in geometry['segments']]
    offset = geometry['tip_offset_tool_z_mm']
    preflight = np.vstack([a.at(np.linspace(0, a.length, max(257, len(a.spline.c)*16))) for a in arcs])
    # 锁定整条轨迹唯一可行的逆解构型分支
    branch = select_branch(preflight, offset, limits.joint_limits_deg, seed_deg)
    arrays = {key: [] for key in ('tip', 'q', 'velocity', 'acceleration', 'jerk', 'segment_id', 'stroke_id', 'phase_id')}
    records = []
    ticks = 0
    previous = branch.q_start
    # 稠密归一化时间采样网格，捕捉笔画急弯处的导数峰值
    tau_check = np.linspace(0., 1., 10001)

    for index, (segment, arc) in enumerate(zip(geometry['segments'], arcs)):
        phase = segment['phase']
        p, v, a, j = evaluate_curve(arc, tau_check, 1.)
        try:
            q = branch.solve(p, previous)
        except ValueError as exc:
            raise ValueError(f'Segment {index}, stroke {segment["stroke_id"]}, {phase}: {exc}') from exc
        qd = numerical_derivatives(q, 1/(len(tau_check)-1))
        # 初步评估七次多项式所需的基准时长
        duration = max(MIN_SEGMENT_DURATION,
                       required_stretch((v, a, j), qd, limits, phase)*TIMING_RESERVE)
        time_law = None

        # 若为书写段且允许自适应优化，对比 LocalTimeLaw 与标准七次多项式
        if adaptive_drawing and phase == 'DRAW':
            candidate = LocalTimeLaw(arc, limits)
            cp, cv, ca, cj = evaluate_curve(arc, tau_check, 1., candidate)
            cq = branch.solve(cp, previous)
            cqd = numerical_derivatives(cq, 1/(len(tau_check)-1))
            candidate_duration = max(MIN_SEGMENT_DURATION,
                                    required_stretch((cv, ca, cj), cqd, limits, phase)*TIMING_RESERVE)
            # 仅当自适应时间律确实能缩短耗时时采纳
            if candidate_duration < duration:
                time_law, duration = candidate, candidate_duration

        # 实际 1 ms 离散网格迭代收敛循环（确保圆整到 1 ms 网格后依然绝不超限）
        for attempt in range(6):
            count = max(20, int(math.ceil(duration/DT)))
            duration = count*DT
            if (ticks+count)*DT > limits.max_duration:
                raise ValueError(f'Trajectory exceeds max-duration {limits.max_duration:g} s; inspect curvature/limits')
            tau = np.arange(count+1, dtype=float)/count
            p, v, a, j = evaluate_curve(arc, tau, duration, time_law)
            q = branch.solve(p, previous)
            qd = numerical_derivatives(q, DT)
            factor = required_stretch((v, a, j), qd, limits, phase)
            if factor <= 1.00001:
                break
            duration *= factor*TIMING_RESERVE
        else:
            raise ValueError(f'Timing did not converge at segment {index}')

        # 首尾段平滑拼接：除第一段外，跳过首点避免时间戳重复
        start = 0 if index == 0 else 1
        for key, data in (('tip', p), ('q', q), ('velocity', v), ('acceleration', a), ('jerk', j)):
            arrays[key].append(data[start:])
        for key, value in (('segment_id', index), ('stroke_id', segment['stroke_id']), ('phase_id', PHASES[phase])):
            arrays[key].append(np.full(count+1-start, value, dtype=np.int32))
        records.append({'segment_id': index, 'stroke_id': segment['stroke_id'], 'phase': phase,
                        'start_index': ticks, 'end_index': ticks+count,
                        'start_time_s': ticks*DT, 'end_time_s': (ticks+count)*DT,
                        'duration_s': duration, 'length_mm': arc.length,
                        'peak_tip_speed_mm_s': peak_norm(v), 'peak_tip_accel_mm_s2': peak_norm(a),
                        'peak_tip_jerk_mm_s3': peak_norm(j), 'retiming_iterations': attempt,
                        'timing_profile': 'local' if time_law is not None else 'seventh_order'})
        ticks += count
        previous = q[-1]
        if progress:
            progress(index+1, len(arcs), phase, duration)

    trajectory = {key: np.concatenate(value) for key, value in arrays.items()}
    trajectory['time'] = np.arange(ticks+1)*DT
    trajectory['records'] = records
    trajectory['branch'] = {'shoulder': branch.shoulder, 'elbow': branch.elbow, 'wrist': branch.wrist}
    report = validate_trajectory(geometry, trajectory, limits)
    return trajectory, report


def validate_trajectory(geometry, trajectory, limits):
    """对生成好的离散轨迹执行全面的独立闭环验证。

    验证项目：
      1. 时间轴严格 1 ms 单调递增；
      2. 独立正运动学验证：FK 计算出的笔尖位置与规划目标位置最大误差 < 1e-5 mm，旋转矩阵误差 < 1e-8；
      3. 关节软限位检验；
      4. 关节角速度、角加速度、角加加速度全轨迹不超限检验（包含段与段之间的拼接边界）；
      5. 纸面书写高度、落笔深度、安全抬笔高度与垂直升降几何检验；
      6. 各动作段交界处速度、加速度、加加速度严格归零停稳（顿笔）检验。
    """
    t, q, tip = trajectory['time'], trajectory['q'], trajectory['tip']
    if len(t) < 2 or len(t) != len(q) or not np.isfinite(q).all() or not np.isfinite(tip).all():
        raise ValueError('Invalid trajectory arrays')
    if not np.allclose(np.diff(t), DT, rtol=0, atol=2e-12) or t[0] != 0:
        raise ValueError('Time grid is not strictly 1 ms starting at zero')

    # 独立正运动学闭环校验
    tip_fk, grip_fk, rotations = forward_tip(q, geometry['tip_offset_tool_z_mm'])
    position_error = np.linalg.norm(tip_fk-tip, axis=1)
    rotation_error = np.max(np.abs(rotations-R_HOLD))
    if position_error.max() > 1e-5 or rotation_error > 1e-8:
        raise ValueError('Forward-kinematics validation failed')
    trajectory['grip'] = grip_fk

    # 关节限位校验
    qdeg = np.rad2deg(q)
    joint_limits = np.asarray(limits.joint_limits_deg)
    if np.any(qdeg < joint_limits[:, 0]-1e-7) or np.any(qdeg > joint_limits[:, 1]+1e-7):
        raise ValueError('Joint position limit exceeded')

    # 关节空间动力学（速度/加速度/加加速度）校验
    qd = numerical_derivatives(q, DT)
    for deriv, bound, name in zip(qd, (limits.joint_speed_deg, limits.joint_accel_deg, limits.joint_jerk_deg),
                                 ('joint velocity', 'joint acceleration', 'joint jerk')):
        if np.any(np.max(np.abs(np.rad2deg(deriv)), axis=0) > bound*1.001):
            raise ValueError(f'{name} exceeds configured limit after segment concatenation')
    for name, data in zip(('q_velocity', 'q_acceleration', 'q_jerk'), qd):
        trajectory[name] = data

    z, lift = geometry['tip_plane_z_mm'], geometry['tip_lift_mm']
    brush_depth = geometry.get('brush_depth_mm', 0.0)
    for record in trajectory['records']:
        a, b = record['start_index'], record['end_index']
        segment_tip = tip[a:b+1]
        phase = record['phase']
        if phase == 'DRAW' and (segment_tip[:, 2].min() < z - brush_depth - 1e-6 or segment_tip[:, 2].max() > z + 1e-6):
            raise ValueError('Drawing leaves paper plane/press zone')
        if phase == 'TRANSFER' and np.max(np.abs(segment_tip[:, 2]-z-lift)) > 1e-8:
            raise ValueError('Transfer leaves clearance plane')
        if phase in ('DESCEND', 'LIFT') and np.max(np.abs(segment_tip[:, :2]-segment_tip[0, :2])) > 1e-8:
            raise ValueError('Nonvertical pen lift/descent')
        if phase == 'DESCEND':
            if abs(segment_tip[0, 2] - (z+lift)) > 1e-7 or segment_tip[-1, 2] < z - brush_depth - 1e-6 or segment_tip[-1, 2] > z + 1e-6:
                raise ValueError('Incorrect descent end height')
        if phase == 'LIFT':
            if abs(segment_tip[-1, 2] - (z+lift)) > 1e-7 or segment_tip[0, 2] < z - brush_depth - 1e-6 or segment_tip[0, 2] > z + 1e-6:
                raise ValueError('Incorrect lift start height')
        # 边界停稳检验：起止点导数必须严格为 0
        for field_name in ('velocity', 'acceleration', 'jerk'):
            if np.max(np.abs(trajectory[field_name][[a, b]])) > 1e-7:
                raise ValueError('Nonzero derivative at a stopped segment boundary')

    # 工作空间包络范围检验
    for axis, key in ((0, 'x_range_mm'), (1, 'y_range_mm')):
        lo, hi = geometry[key]
        if tip[:, axis].min() < lo-1e-8 or tip[:, axis].max() > hi+1e-8:
            raise ValueError('Pen path exceeds supplied XY range')
    if tip[:, 2].min() < z - brush_depth - 1e-6 or tip[:, 2].max() > z+lift+1e-8:
        raise ValueError('Pen path exceeds supplied height range')

    # 生成可导出的验证报告
    report = {
        'status': 'validated', 'text': geometry['text'], 'dt_s': DT, 'samples': len(t),
        'duration_s': float(t[-1]), 'stroke_count': len(geometry['strokes']),
        'segment_count': len(geometry['segments']), 'ik_branch': trajectory['branch'],
        'tip_offset_mm': geometry['tip_offset_tool_z_mm'],
        'brush_depth_mm': brush_depth,
        'tip_plane_z_mm': z, 'grip_writing_z_mm': z+geometry['tip_offset_tool_z_mm'],
        'tip_clearance_z_mm': z+lift, 'grip_clearance_z_mm': z+lift+geometry['tip_offset_tool_z_mm'],
        'tip_min_xyz_mm': tip.min(axis=0).tolist(), 'tip_max_xyz_mm': tip.max(axis=0).tolist(),
        'max_fk_tip_error_mm': float(position_error.max()), 'max_rotation_matrix_error': float(rotation_error),
        'q_start_deg': qdeg[0].tolist(), 'q_end_deg': qdeg[-1].tolist(),
        'joint_min_deg': qdeg.min(axis=0).tolist(), 'joint_max_deg': qdeg.max(axis=0).tolist(),
        'max_joint_step_deg': np.max(np.abs(np.diff(qdeg, axis=0)), axis=0).tolist(),
        'max_joint_speed_deg_s': np.max(np.abs(np.rad2deg(qd[0])), axis=0).tolist(),
        'max_joint_accel_deg_s2': np.max(np.abs(np.rad2deg(qd[1])), axis=0).tolist(),
        'max_joint_jerk_deg_s3': np.max(np.abs(np.rad2deg(qd[2])), axis=0).tolist(),
        'max_tip_speed_mm_s': peak_norm(trajectory['velocity']),
        'max_tip_accel_mm_s2': peak_norm(trajectory['acceleration']),
        'max_tip_jerk_mm_s3': peak_norm(trajectory['jerk']),
        'limits': {'write_speed_mm_s': limits.write_speed, 'travel_speed_mm_s': limits.travel_speed,
                   'vertical_speed_mm_s': limits.vertical_speed, 'cart_accel_mm_s2': limits.cart_accel,
                   'cart_jerk_mm_s3': limits.cart_jerk, 'joint_speed_deg_s': limits.joint_speed_deg.tolist(),
                   'joint_accel_deg_s2': limits.joint_accel_deg.tolist(), 'joint_jerk_deg_s3': limits.joint_jerk_deg.tolist(),
                   'joint_position_deg': joint_limits.tolist()},
        'model': 'Existing structured spherical wrist; L=[491,450,450,84] mm; fixed R=diag(-1,1,-1)',
        'start_condition': 'At rest above the first stroke. No move from an external robot pose is included.',
        'drawing_timing': ('Local time redistribution with original stopped corners preserved.'
                           if any(r.get('timing_profile') == 'local' for r in trajectory['records'])
                           else 'Seventh-order rest-to-rest timing.'),
        'derivative_check': 'Analytic Cartesian derivatives; dense numerical joint derivatives; 5% timing reserve; minimum segment duration 0.5 s.',
        'segments': trajectory['records'],
    }
    return report

