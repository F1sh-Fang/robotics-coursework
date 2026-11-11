#!/usr/bin/env python3
"""Plot the exported joint trajectory via independent FK, including pen offset.

python PlotWritingTcpTrajectory.py
The default is noninteractive PNG output; add --show to open plot windows.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.collections import PatchCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Rectangle
import numpy as np

from .geometry import decode_spline
from .kinematics import forward_tip
from .timing import DT

HERE = Path(__file__).resolve().parent.parent


def configure_fonts():
    font_candidates = [
        'C:/Windows/Fonts/msyh.ttc',
        'C:/Windows/Fonts/simhei.ttf',
        '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
        '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc',
        '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc',
    ]
    for filename in font_candidates:
        if Path(filename).exists():
            prop = font_manager.FontProperties(fname=filename)
            plt.rcParams['font.family'] = prop.get_name()
            break
    plt.rcParams['axes.unicode_minus'] = False


def plot_layout(geometry, filename):
    configure_fonts()
    fig, ax = plt.subplots(figsize=(10, 5.5), facecolor='white')
    for cell in geometry['cells']:
        xmin, xmax = cell['x_mm']
        ymin, ymax = cell['y_mm']
        ax.add_patch(Rectangle((ymin, -xmax), ymax - ymin, xmax - xmin,
                               edgecolor='#a8b5c2', facecolor='none', linestyle='--', linewidth=0.9))

    z_plane = float(geometry['tip_plane_z_mm'])
    brush_depth = max(1e-3, float(geometry['brush_depth_mm']))
    cell_side = float(geometry.get('cell_side_mm', 28.0))

    all_circles = []
    for stroke in geometry['strokes']:
        for sid in stroke['draw_segment_ids']:
            seg = geometry['segments'][sid]
            spline = decode_spline(seg['curve'])
            s_grid = np.linspace(0, 1, 150)
            p = spline(s_grid)
            dz = np.maximum(0.0, z_plane - p[:, 2])
            tau = np.clip(dz / brush_depth, 0.15, 1.0)
            # Physical brush contact circle radius in mm
            radius_mm = cell_side * (0.015 + 0.050 * tau)

            for yi, xi, ri in zip(p[:, 1], -p[:, 0], radius_mm):
                all_circles.append(Circle((yi, xi), ri))

            # Centerline of stroke trajectory
            ax.plot(p[:, 1], -p[:, 0], color='#c0392b', linewidth=0.9, alpha=0.85, zorder=4)

        # Mark stroke start point with stroke index
        first = geometry['segments'][stroke['draw_segment_ids'][0]]
        p0 = np.asarray(first['start_tip_mm'])
        ax.scatter(p0[1], -p0[0], s=18, c='#e67e22', edgecolors='#9d471b', linewidth=0.8, zorder=5)
        ax.annotate(str(stroke['stroke_id'] + 1), (p0[1], -p0[0]), xytext=(4, 4),
                    textcoords='offset points', color='#9d471b', fontsize=8.5, fontweight='bold', zorder=6)

    # Add brush circles collection (translucent ink with visible circle edges)
    if all_circles:
        collection = PatchCollection(all_circles, facecolor='#1a334d', edgecolor='#0e1e2f',
                                     linewidth=0.35, alpha=0.25, zorder=3)
        ax.add_collection(collection)

    xr, yr = geometry['x_range_mm'], geometry['y_range_mm']
    ax.set_xlim(yr[0] - 2, yr[1] + 2)
    ax.set_ylim(-xr[1] - 2, -xr[0] + 2)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('Y (mm) — 书写前进方向 (+Y)')
    ax.set_ylabel('−X (mm) — 字体正上方 (−X)')
    ax.set_title(f'HIT 校徽毛体书法 “{geometry["text"]}” 笔画布局与落笔粗细圆盘模拟\n'
                 f'Z进给深度控制粗细 (Depth={brush_depth:.1f} mm) | 落笔基准平面 Z={z_plane:.1f} mm')
    ax.grid(alpha=0.18)

    # Custom legend
    handles = [
        Line2D([0], [0], color='#1a334d', lw=8, alpha=0.6, label='毛笔落笔粗细轮廓 (截面圆盘叠加)'),
        Line2D([0], [0], color='#c0392b', lw=1.2, label='末端轨迹中心线 (Trajectory Centerline)'),
        Line2D([0], [0], color='#e67e22', marker='o', ls='', markersize=5, label='落笔起点与序号 (Stroke Start)'),
        Line2D([0], [0], color='#a8b5c2', ls='--', lw=1.0, label='字框约束 (Character Cell)'),
    ]
    ax.legend(handles=handles, loc='upper left', fontsize=8.5, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(filename, dpi=200)
    plt.close(fig)


def plot_outputs(directory=HERE/'output', show=False):
    directory = Path(directory)
    configure_fonts()
    geometry = json.loads((directory/'writing_geometry.json').read_text(encoding='utf-8'))
    report = json.loads((directory/'writing_report.json').read_text(encoding='utf-8'))
    data = np.atleast_2d(np.loadtxt(directory/'writing_joint_timed_rad.txt', delimiter=',', skiprows=1))
    t, q = data[:, 0], data[:, 1:]
    if data.shape[1] != 7 or len(data) != report['samples'] or not np.allclose(np.diff(t), DT, rtol=0, atol=2e-12):
        raise ValueError('Exported trajectory does not match report / 1-ms timeline')
    tip, grip, _ = forward_tip(q, geometry['tip_offset_tool_z_mm'])
    colors = plt.get_cmap('tab10')
    fig = plt.figure(figsize=(11, 8), facecolor='white')
    ax = fig.add_subplot(111, projection='3d')
    for record in report['segments']:
        start, end = record['start_index'], record['end_index']
        stride = max(1, (end-start)//400)
        indices = np.unique(np.r_[np.arange(start, end+1, stride), end])
        p = tip[indices]
        drawing = record['phase'] == 'DRAW'
        color = colors(record['stroke_id'] % 10) if drawing else '#929aa5'
        ax.plot(p[:, 0], p[:, 1], p[:, 2], color=color, linewidth=2.6 if drawing else .9,
                linestyle='-' if drawing else '--', alpha=1 if drawing else .7)
        if drawing:
            ax.text(*p[len(p)//2], f' {record["stroke_id"]+1}', fontsize=9)
    sample = np.unique(np.r_[np.arange(0, len(t), max(1, len(t)//2000)), len(t)-1])
    ax.plot(grip[sample, 0], grip[sample, 1], grip[sample, 2], color='#70a3c4',
            linewidth=.6, alpha=.55, label='Grip TCP (20 mm offset)' if geometry['tip_offset_tool_z_mm'] == 20 else 'Grip TCP')
    ax.scatter(*tip[0], color='#277a43', s=40, marker='o')
    ax.scatter(*tip[-1], color='#b23d3d', s=55, marker='x')
    xr, yr, z = geometry['x_range_mm'], geometry['y_range_mm'], geometry['tip_plane_z_mm']
    px, py = np.meshgrid(xr, yr)
    ax.plot_surface(px, py, np.full_like(px, z, dtype=float), color='#e7edf3', alpha=.22, shade=False)
    bounds = np.vstack([tip, grip])
    low, high = bounds.min(axis=0), bounds.max(axis=0)
    low[:2] = [xr[0], yr[0]]
    high[:2] = [xr[1], yr[1]]
    padding = np.maximum((high-low)*.06, 2.)
    ax.set_xlim(low[0]-padding[0], high[0]+padding[0])
    ax.set_ylim(low[1]-padding[1], high[1]+padding[1])
    ax.set_zlim(low[2]-padding[2], high[2]+padding[2])
    ax.set_box_aspect(np.maximum(high-low+2*padding, 1.))
    ax.view_init(elev=28, azim=42)
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    ax.set_title(f'笔尖与握笔 TCP 三维轨迹：{geometry["text"].replace(chr(10), " / ")}\n'
                 f'FK from exported joints | {len(t):,} samples | dt=0.001 s | T={t[-1]:.3f} s')
    handles = [Line2D([0], [0], color='#1d4f79', lw=2.5, label='Pen-tip writing'),
               Line2D([0], [0], color='#929aa5', ls='--', label='Vertical lift/descent & air transfer'),
               Line2D([0], [0], color='#70a3c4', lw=1, label='Grip TCP'),
               Line2D([0], [0], color='#277a43', marker='o', ls='', label='Start'),
               Line2D([0], [0], color='#b23d3d', marker='x', ls='', label='End')]
    ax.legend(handles=handles, loc='upper left', fontsize=8)
    fig.tight_layout()
    fig.savefig(directory/'writing_tcp_trajectory_3d.png', dpi=180)
    plt.close(fig)
    plot_layout(geometry, directory/'writing_layout_xy.png')

    if show:
        # Keep generation headless. Open saved images only on explicit request.
        import webbrowser
        for name in ('writing_tcp_trajectory_3d.png', 'writing_layout_xy.png'):
            webbrowser.open((directory/name).resolve().as_uri())
    return {'samples': len(t), 'fk_tip_min_mm': tip.min(axis=0).tolist(), 'fk_tip_max_mm': tip.max(axis=0).tolist()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=HERE/'output')
    parser.add_argument('--show', action='store_true')
    args = parser.parse_args()
    print(json.dumps(plot_outputs(args.directory, args.show), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
