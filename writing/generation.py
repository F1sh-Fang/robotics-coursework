#!/usr/bin/env python3
"""
生成“哈工大”校徽毛笔书法轨迹、逐点 1 ms 关节角度和末端数据文件。
"""
import argparse
import json
from pathlib import Path
import tempfile

import numpy as np

from .geometry import HERE, build_geometry
from .kinematics import forward_tip
from .timing import DT, MotionLimits, numerical_derivatives, plan_trajectory


def json_write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def export_trajectory(directory, geometry, trajectory, report):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    json_write(directory / 'writing_geometry.json', geometry)
    t, q = trajectory['time'], trajectory['q']
    deg = np.rad2deg(q)
    deg[np.abs(deg) < .5e-9] = 0.
    np.savetxt(directory / 'writing_joint_deg.txt', deg, delimiter=' ', fmt='%.9f')
    header = 'time_s,' + ','.join(f'theta{i}_rad' for i in range(1, 7))
    np.savetxt(directory / 'writing_joint_timed_rad.txt', np.column_stack([t, q]), delimiter=',',
               header=header, comments='', fmt=['%.3f'] + ['%.12f'] * 6)
    tcp_header = 'time_s,tip_x_mm,tip_y_mm,tip_z_mm,grip_x_mm,grip_y_mm,grip_z_mm,phase_id,stroke_id,segment_id'
    tcp = np.column_stack([t, trajectory['tip'], trajectory['grip'], trajectory['phase_id'],
                           trajectory['stroke_id'], trajectory['segment_id']])
    np.savetxt(directory / 'writing_tcp_trajectory.csv', tcp, delimiter=',', header=tcp_header,
               comments='', fmt=['%.3f'] + ['%.9f'] * 6 + ['%d'] * 3)

    # Verify exported text files
    degree_file = np.atleast_2d(np.loadtxt(directory / 'writing_joint_deg.txt'))
    timed_file = np.atleast_2d(np.loadtxt(directory / 'writing_joint_timed_rad.txt', delimiter=',', skiprows=1))
    if degree_file.shape != q.shape or timed_file.shape != (len(t), 7):
        raise ValueError('Export row/column counts disagree')
    if not np.allclose(np.diff(timed_file[:, 0]), DT, rtol=0, atol=2e-12):
        raise ValueError('Exported timeline is not 1 ms')
    conversion_error = float(np.max(np.abs(np.deg2rad(degree_file) - timed_file[:, 1:])))
    if conversion_error > 5.2e-10:
        raise ValueError('The two joint files do not contain the same trajectory')
    recovered, _, _ = forward_tip(timed_file[:, 1:], geometry['tip_offset_tool_z_mm'])
    recovered_deg, _, _ = forward_tip(np.deg2rad(degree_file), geometry['tip_offset_tool_z_mm'])
    error = float(np.linalg.norm(recovered - trajectory['tip'], axis=1).max())
    error_deg = float(np.linalg.norm(recovered_deg - trajectory['tip'], axis=1).max())
    if max(error, error_deg) > 1e-4:
        raise ValueError('TXT precision causes excessive FK error')
    for filename, angles in [('radian', timed_file[:, 1:]), ('degree', np.deg2rad(degree_file))]:
        d = numerical_derivatives(angles, DT)
        for name, value, limit_key in zip(('velocity', 'acceleration', 'jerk'), d,
                                          ('joint_speed_deg_s', 'joint_accel_deg_s2', 'joint_jerk_deg_s3')):
            peak = np.max(np.abs(np.rad2deg(value)), axis=0)
            if np.any(peak > np.asarray(report['limits'][limit_key]) * 1.001):
                raise ValueError(f'Rounded {filename} TXT exceeds {name} limit')
    report['export_check'] = {
        'row_count': len(t), 'column_counts': [6, 7],
        'max_deg_rad_difference_rad': conversion_error,
        'max_fk_error_from_rad_txt_mm': error,
        'max_fk_error_from_deg_txt_mm': error_deg,
        'rounded_txt_dynamics_passed': True
    }
    json_write(directory / 'writing_report.json', report)


def create_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--brush-depth', type=float, default=5.0,
                        help='毛笔下压最大进给量，mm；根据笔画粗细控制Z进给，0为平面书写')
    parser.add_argument('--z', type=float, default=595., help='笔尖接触写字平面高度，mm')
    parser.add_argument('--x', type=float, nargs=2, default=[480., 620.], metavar=('MIN', 'MAX'), help='工作空间 X 范围')
    parser.add_argument('--y', type=float, nargs=2, default=[-130., 130.], metavar=('MIN', 'MAX'), help='工作空间 Y 范围')
    parser.add_argument('--lift', type=float, default=10., help='笔尖竖直提起高度，mm')
    parser.add_argument('--tip-offset', type=float, default=0., help='握笔TCP沿工具+Z到笔尖的距离，mm')
    parser.add_argument('--margin', type=float, default=2., help='字框边距，mm')
    parser.add_argument('--gap', type=float, default=4., help='字间距，mm')
    parser.add_argument('--curve-tolerance', type=float, default=.35, help='平滑曲线拟合容差，mm')
    parser.add_argument('--corner-angle', type=float, default=75., help='折角顿笔停稳角度阈值，deg')
    parser.add_argument('--write-speed', type=float, default=50., help='书写速度上限，mm/s')
    parser.add_argument('--travel-speed', type=float, default=80., help='空移速度上限，mm/s')
    parser.add_argument('--vertical-speed', type=float, default=40., help='提落笔速度上限，mm/s')
    parser.add_argument('--cart-accel', type=float, default=100., help='末端加速度上限，mm/s^2')
    parser.add_argument('--cart-jerk', type=float, default=1000., help='末端加加速度上限，mm/s^3')
    parser.add_argument('--joint-vmax-deg', type=float, nargs=6, default=[50.] * 6, help='六关节速度上限，deg/s')
    parser.add_argument('--joint-amax-deg', type=float, nargs=6, default=[180.] * 6, help='六关节加速度上限，deg/s^2')
    parser.add_argument('--joint-jmax-deg', type=float, nargs=6, default=[1000.] * 6, help='六关节加加速度上限，deg/s^3')
    parser.add_argument('--joint-limits', type=Path, help='6x2角度限位JSON，deg；默认采用项目配置')
    parser.add_argument('--seed-deg', type=float, nargs=6, help='起点选解参考角度，deg')
    parser.add_argument('--max-duration', type=float, default=1200., help='总时长上限，s')
    parser.add_argument('--output-dir', type=Path, default=HERE / 'output', help='输出目录')
    parser.add_argument('--no-plots', action='store_true', help='跳过轨迹图生成')
    return parser


def main():
    parser = create_parser()
    args = parser.parse_args()

    geometry = build_geometry(
        text='哈工大',
        z=args.z,
        x_range=args.x,
        y_range=args.y,
        lift=args.lift,
        tip_offset=args.tip_offset,
        brush_depth=args.brush_depth,
        margin=args.margin,
        gap=args.gap,
        tolerance=args.curve_tolerance,
        corner_angle=args.corner_angle,
    )
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    limits = MotionLimits(
        args.write_speed, args.travel_speed, args.vertical_speed,
        args.cart_accel, args.cart_jerk,
        np.array(args.joint_vmax_deg),
        np.array(args.joint_amax_deg),
        np.array(args.joint_jmax_deg),
        max_duration=args.max_duration
    )
    if args.joint_limits:
        limits.joint_limits_deg = np.asarray(json.loads(args.joint_limits.read_text(encoding='utf-8')), dtype=float)

    def progress(index, count, phase, duration):
        print(f'[{index:02d}/{count:02d}] {phase:8s} {duration:8.3f} s', flush=True)

    trajectory, report = plan_trajectory(geometry, limits, args.seed_deg, progress)

    with tempfile.TemporaryDirectory(prefix='.writing-', dir=output) as staging:
        stage = Path(staging)
        export_trajectory(stage, geometry, trajectory, report)
        if not args.no_plots:
            from .plotting import plot_outputs
            plot_outputs(stage)
        for path in stage.iterdir():
            path.replace(output / path.name)

    print(f'Validated: {report["samples"]:,} points, dt=0.001 s, T={report["duration_s"]:.3f} s')
    print(f'Character source: HIT Logo (哈工大), Brush depth: {args.brush_depth:g} mm')
    print(f'Writing tip plane Z={report["tip_plane_z_mm"]:g}, max press Z={report["tip_plane_z_mm"] - args.brush_depth:g} mm')
    print(f'Air tip clearance Z={report["tip_clearance_z_mm"]:g} mm')
    print('Start joints (deg):', ' '.join(f'{v:.6f}' for v in report['q_start_deg']))
    print(f'Exported radian TXT FK error: {report["export_check"]["max_fk_error_from_rad_txt_mm"]:.3g} mm')
    print('Output directory:', output)


if __name__ == '__main__':
    main()
