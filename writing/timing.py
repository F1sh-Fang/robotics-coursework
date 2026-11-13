"""Layers 2/3: fixed-path timing, locked-branch IK, and dynamics validation."""
from dataclasses import dataclass, field
import math
import numpy as np
from scipy.interpolate import CubicSpline, PchipInterpolator
from scipy.ndimage import gaussian_filter1d, maximum_filter1d

from .geometry import ArcCurve, PHASES, validate_geometry
from .kinematics import LIMITS_DEG, R_HOLD, forward_tip, select_branch

DT = 0.001
MIN_SEGMENT_DURATION = 0.5
TIMING_RESERVE = 1.05


@dataclass
class MotionLimits:
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
        return self.write_speed if phase == 'DRAW' else self.travel_speed if phase == 'TRANSFER' else self.vertical_speed


def seventh_order(tau):
    """Normalized rest-to-rest profile, with velocity/acceleration/jerk zero at ends."""
    t = np.clip(np.asarray(tau, dtype=float), 0., 1.)
    u = np.minimum(t, 1-t)
    small = u**4*(35+u*(-84+u*(70-20*u)))
    s = np.where(t <= .5, small, 1-small)
    v = 140*t**3*(1-t)**3
    a = 420*t**2*(1-t)**2*(1-2*t)
    j = 840*t*(1-t)*(1-5*t+5*t*t)
    return s, v, a, j


def evaluate_curve(arc, tau, duration, time_law=None):
    s, sd, sdd, sddd = seventh_order(tau) if time_law is None else time_law.at(tau)
    s, sd, sdd, sddd = (s*arc.length, sd*arc.length/duration,
                         sdd*arc.length/duration**2, sddd*arc.length/duration**3)
    p, ps, pss, psss = arc.at(s, derivatives=True)
    velocity = ps*sd[:, None]
    acceleration = ps*sdd[:, None]+pss*sd[:, None]**2
    jerk = ps*sddd[:, None]+3*pss*(sd*sdd)[:, None]+psss*sd[:, None]**3
    return p, velocity, acceleration, jerk


class LocalTimeLaw:
    """Redistribute time within a stroke, retaining its path and stopped ends.

    A smooth, positive time density reserves more time near difficult curvature
    and less on easy portions. Reparameterizing the seventh-order profile keeps
    velocity, acceleration and jerk zero at the original stroke/corner stops.
    The final duration is still checked against every Cartesian and joint limit.
    """
    def __init__(self, arc, limits):
        grid = np.linspace(0., 1., 2001)
        _, velocity, acceleration, jerk = evaluate_curve(arc, grid, 1.)
        weight = np.maximum.reduce([
            np.linalg.norm(velocity, axis=1) / limits.write_speed,
            np.sqrt(np.linalg.norm(acceleration, axis=1) / limits.cart_accel),
            np.cbrt(np.linalg.norm(jerk, axis=1) / limits.cart_jerk),
        ])
        # Smooth the time density itself: abrupt speed changes would add jerk.
        weight = np.maximum(weight, weight.max() * 0.1)
        weight = gaussian_filter1d(maximum_filter1d(weight, size=81), sigma=40.)
        density = CubicSpline(grid, weight)
        integral = density.antiderivative()
        elapsed = integral(grid) - integral(0.)
        self.density = CubicSpline(grid, weight / elapsed[-1])
        self.integral = self.density.antiderivative()
        self.inverse = PchipInterpolator(elapsed / elapsed[-1], grid)

    def at(self, tau):
        tau = np.clip(np.asarray(tau, dtype=float), 0., 1.)
        u = self.inverse(tau)
        for _ in range(3):
            residual = self.integral(u) - self.integral(0.) - tau
            u = np.clip(u - residual / self.density(u), 0., 1.)
        s, v, a, j = seventh_order(u)
        w, dw, ddw = self.density(u), self.density(u, 1), self.density(u, 2)
        # Chain rule through the inverse cumulative time map, through order 3.
        return (s, v / w, a / w**2 - v * dw / w**3,
                j / w**3 - 3 * a * dw / w**4 - v * ddw / w**4
                + 3 * v * dw**2 / w**5)


def numerical_derivatives(q, step):
    """Diagnostics on the unrounded joint series, never used to alter geometry."""
    v = np.gradient(q, step, axis=0, edge_order=2)
    a = np.gradient(v, step, axis=0, edge_order=2)
    j = np.gradient(a, step, axis=0, edge_order=2)
    return v, a, j


def peak_norm(a):
    return float(np.linalg.norm(a, axis=1).max())


def required_stretch(cart_derivatives, joint_derivatives, limits, phase):
    requirements = [0.]
    for order, data, limit in zip((1, 2, 3), cart_derivatives,
                                  (limits.phase_speed(phase), limits.cart_accel, limits.cart_jerk)):
        requirements.append((peak_norm(data)/limit)**(1/order))
    for order, data, limit in zip((1, 2, 3), joint_derivatives,
                                  (limits.joint_speed_deg, limits.joint_accel_deg, limits.joint_jerk_deg)):
        requirements.append(float(np.max(np.abs(data)/np.deg2rad(limit)))**(1/order))
    return max(requirements)


def plan_trajectory(geometry, limits=None, seed_deg=None, progress=None, *, adaptive_drawing=True):
    validate_geometry(geometry)
    limits = limits or MotionLimits()
    limits.validate()
    arcs = [ArcCurve(s['curve']) for s in geometry['segments']]
    offset = geometry['tip_offset_tool_z_mm']
    preflight = np.vstack([a.at(np.linspace(0, a.length, max(257, len(a.spline.c)*16))) for a in arcs])
    branch = select_branch(preflight, offset, limits.joint_limits_deg, seed_deg)
    arrays = {key: [] for key in ('tip', 'q', 'velocity', 'acceleration', 'jerk', 'segment_id', 'stroke_id', 'phase_id')}
    records = []
    ticks = 0
    previous = branch.q_start
    # Dense normalized-time check captures curved-stroke derivative maxima.
    tau_check = np.linspace(0., 1., 10001)
    for index, (segment, arc) in enumerate(zip(geometry['segments'], arcs)):
        phase = segment['phase']
        p, v, a, j = evaluate_curve(arc, tau_check, 1.)
        try:
            q = branch.solve(p, previous)
        except ValueError as exc:
            raise ValueError(f'Segment {index}, stroke {segment["stroke_id"]}, {phase}: {exc}') from exc
        qd = numerical_derivatives(q, 1/(len(tau_check)-1))
        duration = max(MIN_SEGMENT_DURATION,
                       required_stretch((v, a, j), qd, limits, phase)*TIMING_RESERVE)
        time_law = None
        if adaptive_drawing and phase == 'DRAW':
            candidate = LocalTimeLaw(arc, limits)
            cp, cv, ca, cj = evaluate_curve(arc, tau_check, 1., candidate)
            cq = branch.solve(cp, previous)
            cqd = numerical_derivatives(cq, 1/(len(tau_check)-1))
            candidate_duration = max(MIN_SEGMENT_DURATION,
                                    required_stretch((cv, ca, cj), cqd, limits, phase)*TIMING_RESERVE)
            # Keep the original profile when local timing offers no improvement.
            if candidate_duration < duration:
                time_law, duration = candidate, candidate_duration
        # Re-evaluate actual 1-ms sampling after timing, including all joint constraints.
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
        start = 0 if index == 0 else 1  # one shared endpoint, no duplicated timestamp
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
    t, q, tip = trajectory['time'], trajectory['q'], trajectory['tip']
    if len(t) < 2 or len(t) != len(q) or not np.isfinite(q).all() or not np.isfinite(tip).all():
        raise ValueError('Invalid trajectory arrays')
    if not np.allclose(np.diff(t), DT, rtol=0, atol=2e-12) or t[0] != 0:
        raise ValueError('Time grid is not strictly 1 ms starting at zero')
    tip_fk, grip_fk, rotations = forward_tip(q, geometry['tip_offset_tool_z_mm'])
    position_error = np.linalg.norm(tip_fk-tip, axis=1)
    rotation_error = np.max(np.abs(rotations-R_HOLD))
    if position_error.max() > 1e-5 or rotation_error > 1e-8:
        raise ValueError('Forward-kinematics validation failed')
    trajectory['grip'] = grip_fk
    qdeg = np.rad2deg(q)
    joint_limits = np.asarray(limits.joint_limits_deg)
    if np.any(qdeg < joint_limits[:, 0]-1e-7) or np.any(qdeg > joint_limits[:, 1]+1e-7):
        raise ValueError('Joint position limit exceeded')
    qd = numerical_derivatives(q, DT)
    # Also checks concatenation boundaries; slowing segments cannot fix a position jump.
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
        for field_name in ('velocity', 'acceleration', 'jerk'):
            if np.max(np.abs(trajectory[field_name][[a, b]])) > 1e-7:
                raise ValueError('Nonzero derivative at a stopped segment boundary')
    for axis, key in ((0, 'x_range_mm'), (1, 'y_range_mm')):
        lo, hi = geometry[key]
        if tip[:, axis].min() < lo-1e-8 or tip[:, axis].max() > hi+1e-8:
            raise ValueError('Pen path exceeds supplied XY range')
    if tip[:, 2].min() < z - brush_depth - 1e-6 or tip[:, 2].max() > z+lift+1e-8:
        raise ValueError('Pen path exceeds supplied height range')
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
