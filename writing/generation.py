#!/usr/bin/env python3
"""生成“哈工大”校徽毛笔书法轨迹的主控流程与数据导出模块。

本模块实现：
  1. 命令行参数的解析与配置（工作空间范围、书写高度、毛笔下压深度、动力学限值等）；
  2. 驱动 Layer 1 (几何排版)、Layer 2 (时间律规划)、Layer 3 (运动学逆解) 协同执行；
  3. 导出标准控制器直读文本与科研分析数据 (TXT, CSV, JSON)；
  4. 对导出的量化文本文件实施严密的多重闭环数值验证（FK 误差与动力学边界检验）；
  5. 采用临时暂存目录 (Staging Directory) 原子化更新输出结果，防止写入中断导致数据损坏。
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
    """将 Python 字典以标准 UTF-8 格式缩进写入 JSON 文件。"""
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def export_trajectory(directory, geometry, trajectory, report):
    """将规划好的轨迹导出为多种工业控制器与仿真软件所要求的标准格式文件，并进行严格自检。

    输出文件清单：
      1. writing_geometry.json: 第一层无时间几何路径与分段信息；
      2. writing_joint_deg.txt: 6 列关节角度（单位度，空格分隔，保留 9 位小数，机器人控制器直读）；
      3. writing_joint_timed_rad.txt: 1 列时间 + 6 列关节角度（单位弧度，逗号分隔，含表头，供 MATLAB 读取）；
      4. writing_tcp_trajectory.csv: 笔尖空间坐标、握笔 TCP 坐标及动作阶段标识；
      5. writing_report.json: 全套校验报告与指标统计。

    严格自检逻辑：
      - 检查行数与列数是否完全吻合；
      - 检查时间网格是否严格 1 ms；
      - 检查度数与弧度文本文件之间的数值一致性（转换误差 < 5.2e-10 rad）；
      - 重新加载导出的文本文件计算正运动学 FK，确保由于小数位截断引入的末端位置误差 < 1e-4 mm；
      - 对导出的四舍五入后角度文本重新做中心数值差分，验证截断后各阶动力学指标依然完全满足限值。

    Args:
        directory: 目标输出目录 Path
        geometry: 几何路径字典
        trajectory: 离散轨迹字典
        report: 检验报告字典
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    json_write(directory / 'writing_geometry.json', geometry)
    t, q = trajectory['time'], trajectory['q']

    # 1. 导出度数格式文本 (writing_joint_deg.txt)
    deg = np.rad2deg(q)
    deg[np.abs(deg) < .5e-9] = 0.
    np.savetxt(directory / 'writing_joint_deg.txt', deg, delimiter=' ', fmt='%.9f')

    # 2. 导出带时间戳弧度格式文本 (writing_joint_timed_rad.txt)
    header = 'time_s,' + ','.join(f'theta{i}_rad' for i in range(1, 7))
    np.savetxt(directory / 'writing_joint_timed_rad.txt', np.column_stack([t, q]), delimiter=',',
               header=header, comments='', fmt=['%.3f'] + ['%.12f'] * 6)

    # 3. 导出笛卡尔空间 TCP 与笔尖轨迹 (writing_tcp_trajectory.csv)
    tcp_header = 'time_s,tip_x_mm,tip_y_mm,tip_z_mm,grip_x_mm,grip_y_mm,grip_z_mm,phase_id,stroke_id,segment_id'
    tcp = np.column_stack([t, trajectory['tip'], trajectory['grip'], trajectory['phase_id'],
                           trajectory['stroke_id'], trajectory['segment_id']])
    np.savetxt(directory / 'writing_tcp_trajectory.csv', tcp, delimiter=',', header=tcp_header,
               comments='', fmt=['%.3f'] + ['%.9f'] * 6 + ['%d'] * 3)

    # 4. 重新读取磁盘文本，实施严密的闭环精度与动力学校验
    degree_file = np.atleast_2d(np.loadtxt(directory / 'writing_joint_deg.txt'))
    timed_file = np.atleast_2d(np.loadtxt(directory / 'writing_joint_timed_rad.txt', delimiter=',', skiprows=1))
    if degree_file.shape != q.shape or timed_file.shape != (len(t), 7):
        raise ValueError('Export row/column counts disagree')
    if not np.allclose(np.diff(timed_file[:, 0]), DT, rtol=0, atol=2e-12):
        raise ValueError('Exported timeline is not 1 ms')
    conversion_error = float(np.max(np.abs(np.deg2rad(degree_file) - timed_file[:, 1:])))
    if conversion_error > 5.2e-10:
        raise ValueError('The two joint files do not contain the same trajectory')

    # 验证截断误差对正运动学精度的影响
    recovered, _, _ = forward_tip(timed_file[:, 1:], geometry['tip_offset_tool_z_mm'])
    recovered_deg, _, _ = forward_tip(np.deg2rad(degree_file), geometry['tip_offset_tool_z_mm'])
    error = float(np.linalg.norm(recovered - trajectory['tip'], axis=1).max())
    error_deg = float(np.linalg.norm(recovered_deg - trajectory['tip'], axis=1).max())
    if max(error, error_deg) > 1e-4:
        raise ValueError('TXT precision causes excessive FK error')

    # 验证导出的四舍五入角度数值进行差分后是否仍然满足速度、加速度、加加速度限制
    for filename, angles in [('radian', timed_file[:, 1:]), ('degree', np.deg2rad(degree_file))]:
        d = numerical_derivatives(angles, DT)
        for name, value, limit_key in zip(('velocity', 'acceleration', 'jerk'), d,
                                          ('joint_speed_deg_s', 'joint_accel_deg_s2', 'joint_jerk_deg_s3')):
            peak = np.max(np.abs(np.rad2deg(value)), axis=0)
            if np.any(peak > np.asarray(report['limits'][limit_key]) * 1.001):
                raise ValueError(f'Rounded {filename} TXT exceeds {name} limit')

    # 将检验数据记入报告并写入 writing_report.json
    report['export_check'] = {
        'row_count': len(t), 'column_counts': [6, 7],
        'max_deg_rad_difference_rad': conversion_error,
        'max_fk_error_from_rad_txt_mm': error,
        'max_fk_error_from_deg_txt_mm': error_deg,
        'rounded_txt_dynamics_passed': True
    }
    json_write(directory / 'writing_report.json', report)


def create_parser():
    """配置命令行参数解析器，包含几何参数、工作空间、毛笔物理下压模型和动力学限制。"""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--brush-depth', type=float, default=5.0,
                        help='毛笔下压最大进给量，mm；根据笔画粗细控制Z进给，0为平面书写')
    parser.add_argument('--z', type=float, default=593., help='笔尖接触写字平面高度，mm')
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
    parser.add_argument('--cart-accel', type=float, default=150., help='末端加速度上限，mm/s^2')
    parser.add_argument('--cart-jerk', type=float, default=3000., help='末端加加速度上限，mm/s^3')
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
    """主程序入口：解析参数、生成轨迹、验证并原子导出文件与图表。"""
    parser = create_parser()
    args = parser.parse_args()

    # 1. 第一层：构建 3D 几何路径
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

    # 2. 第二/三层：配置动力学限值并执行时间规划与连续逆解
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

    # 3. 采用临时目录暂存并原子化替换，保证生成安全
    with tempfile.TemporaryDirectory(prefix='.writing-', dir=output) as staging:
        stage = Path(staging)
        export_trajectory(stage, geometry, trajectory, report)
        if not args.no_plots:
            from .plotting import plot_outputs
            plot_outputs(stage)
        for path in stage.iterdir():
            path.replace(output / path.name)

    # 4. 打印生成汇总摘要
    print(f'Validated: {report["samples"]:,} points, dt=0.001 s, T={report["duration_s"]:.3f} s')
    print(f'Character source: HIT Logo (哈工大), Brush depth: {args.brush_depth:g} mm')
    print(f'Writing tip plane Z={report["tip_plane_z_mm"]:g}, max press Z={report["tip_plane_z_mm"] - args.brush_depth:g} mm')
    print(f'Air tip clearance Z={report["tip_clearance_z_mm"]:g} mm')
    print('Start joints (deg):', ' '.join(f'{v:.6f}' for v in report['q_start_deg']))
    print(f'Exported radian TXT FK error: {report["export_check"]["max_fk_error_from_rad_txt_mm"]:.3g} mm')
    print('Output directory:', output)


if __name__ == '__main__':
    main()

