"""自动化测试套件：校徽字符几何、毛笔下压进给、局部时间律、运动学与动力学校验。"""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from writing.generation import export_trajectory
from writing.geometry import ArcCurve, HERE, build_geometry, make_layout
from writing.kinematics import R_HOLD, forward_tip
from writing.timing import DT, LocalTimeLaw, MotionLimits, numerical_derivatives, plan_trajectory, required_stretch


class LogoGeometryTests(unittest.TestCase):
    """测试第一层无时间几何路径、字框排版与笔画下压深度特性。"""

    def test_cli_help(self):
        """测试 CLI 命令行帮助信息是否正常输出，且包含关键参数。"""
        result = subprocess.run([sys.executable, '-B', str(HERE / 'GenerateWritingTrajectory.py'), '--help'],
                                capture_output=True)
        self.assertEqual(result.returncode, 0)
        self.assertIn(b'--brush-depth', result.stdout)

    def test_ha_gong_da_strokes_and_layout(self):
        """测试哈工大三字 15 笔画分解、动作阶段生成与三维工作空间边界约束。"""
        g = build_geometry()
        # 哈工大书法规范共 15 笔：哈 (9) + 工 (3) + 大 (3)
        self.assertEqual(len(g['strokes']), 15)
        self.assertEqual(len(g['cells']), 3)
        self.assertAlmostEqual(g['brush_depth_mm'], 1.0)

        phases = [s['phase'] for s in g['segments']]
        self.assertEqual(phases.count('DESCEND'), 13)
        self.assertEqual(phases.count('LIFT'), 13)
        self.assertLessEqual(phases.count('TRANSFER'), 12)
        self.assertGreaterEqual(phases.count('TRANSFER'), 10)

        for segment in g['segments']:
            curve = ArcCurve(segment['curve'])
            points = curve.at(np.linspace(0., curve.length, 101))
            self.assertTrue(np.all(points[:, 0] >= 450 - 1e-8))
            self.assertTrue(np.all(points[:, 0] <= 500 + 1e-8))
            self.assertTrue(np.all(points[:, 1] >= 50 - 1e-8))
            self.assertTrue(np.all(points[:, 1] <= 150 + 1e-8))
            if segment['phase'] in ('LIFT', 'DESCEND'):
                np.testing.assert_allclose(points[:, :2], np.tile(points[0, :2], (len(points), 1)), atol=1e-10)
            if segment['phase'] == 'DRAW':
                self.assertTrue(np.all(points[:, 2] >= 419.0 - 1e-6))
                self.assertTrue(np.all(points[:, 2] <= 420.0 + 1e-6))

    def test_gong_stays_on_paper_between_strokes(self):
        """测试“工”字三笔连写特性：三笔之间笔尖始终保持在纸面接触，不抬空。"""
        for text in ('工', '哈工大', '工工'):
            with self.subTest(text=text):
                g = build_geometry(text=text)
                for cell in g['cells']:
                    if cell['character'] != '工':
                        continue
                    segments = [s for s in g['segments']
                                if s['character_index'] == cell['character_index']]
                    if segments[0]['phase'] == 'TRANSFER':
                        segments = segments[1:]
                    self.assertEqual(segments[0]['phase'], 'DESCEND')
                    self.assertEqual(segments[-1]['phase'], 'LIFT')
                    self.assertTrue(all(s['phase'] == 'DRAW' for s in segments[1:-1]))
                    self.assertEqual(len({s['stroke_id'] for s in segments[1:-1]}), 3)
                    for before, after in zip(segments, segments[1:]):
                        np.testing.assert_allclose(before['end_tip_mm'], after['start_tip_mm'], atol=1e-9)

    def test_brush_depth_variations(self):
        """测试毛笔下压进给参数 brush_depth 的物理下压效果。"""
        # brush_depth=0 时退化为完全平面书写 (Z=420.0)
        g_flat = build_geometry(brush_depth=0.0)
        for s in g_flat['segments']:
            if s['phase'] == 'DRAW':
                curve = ArcCurve(s['curve'])
                pts = curve.at(np.linspace(0, curve.length, 51))
                np.testing.assert_allclose(pts[:, 2], 420.0, atol=1e-9)

        # brush_depth=1.5 时，书写段 Z 在 [418.5, 420.0] 范围内随笔宽连续变化
        g_deep = build_geometry(brush_depth=1.5)
        draw_z = []
        for s in g_deep['segments']:
            if s['phase'] == 'DRAW':
                curve = ArcCurve(s['curve'])
                pts = curve.at(np.linspace(0, curve.length, 51))
                draw_z.extend(pts[:, 2])
        self.assertLess(min(draw_z), 419.0)
        self.assertGreater(max(draw_z), 419.5)


class LogoKinematicsAndTimingTests(unittest.TestCase):
    """测试运动学逆解锁定、时间网格、限位约束与导出文件验证。"""

    @classmethod
    def setUpClass(cls):
        cls.geometry = build_geometry(brush_depth=1.0)
        cls.limits = MotionLimits(write_speed=40., travel_speed=60., vertical_speed=20.)
        cls.trajectory, cls.report = plan_trajectory(cls.geometry, cls.limits)

    def test_timing_minimum_and_subsecond_segments(self):
        """测试单段最小持续时长保护逻辑。"""
        durations = [record['duration_s'] for record in self.report['segments']]
        self.assertGreaterEqual(min(durations), 0.5)
        self.assertLess(min(durations), 1.0)
        zeros = np.zeros((3, 3))
        joint_zeros = np.zeros((3, 6))
        self.assertEqual(required_stretch((zeros, zeros, zeros),
                                         (joint_zeros, joint_zeros, joint_zeros),
                                         self.limits, 'DRAW'), 0.)

    def test_forward_kinematics_and_timeline(self):
        """测试 1 ms 时间网格以及 POE 正运动学闭环精度 (误差 < 1e-5 mm)。"""
        t = self.trajectory['time']
        q = self.trajectory['q']
        tip = self.trajectory['tip']
        self.assertAlmostEqual(np.diff(t)[0], DT)
        recovered, _, rotations = forward_tip(q, self.geometry['tip_offset_tool_z_mm'])
        max_error = np.linalg.norm(recovered - tip, axis=1).max()
        self.assertLess(max_error, 1e-5)
        self.assertLess(np.max(np.abs(rotations - R_HOLD)), 1e-8)

    def test_joint_limits_and_dynamics(self):
        """测试关节位置限位及速度、加速度、加加速度动力学上限。"""
        q = self.trajectory['q']
        qd = numerical_derivatives(q, DT)
        qdeg = np.rad2deg(q)
        for i in range(6):
            self.assertTrue(np.all(qdeg[:, i] >= self.limits.joint_limits_deg[i, 0]))
            self.assertTrue(np.all(qdeg[:, i] <= self.limits.joint_limits_deg[i, 1]))
        for deriv, bound in zip(qd, (self.limits.joint_speed_deg, self.limits.joint_accel_deg, self.limits.joint_jerk_deg)):
            peak = np.max(np.abs(np.rad2deg(deriv)), axis=0)
            self.assertTrue(np.all(peak <= bound * 1.001))

    def test_export_file_verification(self):
        """测试轨迹导出文件的格式与完整性。"""
        with tempfile.TemporaryDirectory() as temp_dir:
            out = Path(temp_dir)
            export_trajectory(out, self.geometry, self.trajectory, self.report)
            deg_file = out / 'writing_joint_deg.txt'
            rad_file = out / 'writing_joint_timed_rad.txt'
            self.assertTrue(deg_file.exists())
            self.assertTrue(rad_file.exists())
            self.assertEqual(len(deg_file.read_text().strip().splitlines()), len(self.trajectory['time']))


class LocalDrawingTimingTests(unittest.TestCase):
    """测试第四次迭代引入的局部自适应时间律 (LocalTimeLaw)。"""

    @classmethod
    def setUpClass(cls):
        cls.geometry = build_geometry(z=593., x_range=(480., 620.),
                                      y_range=(-130., 130.), lift=10.,
                                      tip_offset=0., brush_depth=5.)
        cls.limits = MotionLimits(write_speed=50., travel_speed=80., vertical_speed=40.,
                                  cart_accel=150., cart_jerk=3000.,
                                  joint_speed_deg=np.full(6, 50.))
        cls.old, cls.old_report = plan_trajectory(cls.geometry, cls.limits, adaptive_drawing=False)
        cls.new, cls.new_report = plan_trajectory(cls.geometry, cls.limits)

    def test_long_stroke_and_startup_improve(self):
        """测试局部自适应时间律缩短总时长，并显著加速长笔画与起步阶段。"""
        # 笔画已全部光滑，统一时间律下也不慢，总时长只要求缩短；
        # 3 = “哈”字撇，长且光滑、主要受速度限制，局部时间律收益最明显。
        self.assertLess(self.new_report['duration_s'], self.old_report['duration_s'])
        records = [next(r for r in report['segments'] if r['stroke_id'] == 3 and r['phase'] == 'DRAW')
                   for report in (self.old_report, self.new_report)]
        self.assertLess(records[1]['duration_s'], records[0]['duration_s'] * .7)
        startup = []
        for trajectory, record in zip((self.old, self.new), records):
            a, b = record['start_index'], record['end_index']
            distance = np.linalg.norm(trajectory['tip'][a:b+1] - trajectory['tip'][a], axis=1)
            startup.append(np.flatnonzero(distance >= 1.)[0] * DT)
        self.assertLess(startup[1], startup[0] * .5)

    def test_original_stops_and_air_motion_are_preserved(self):
        """测试局部时间律保持了原有的顿笔停稳零冲击性质，且空中平移段耗时保持不变。"""
        for old, new in zip(self.old_report['segments'], self.new_report['segments']):
            self.assertEqual((old['phase'], old['stroke_id']), (new['phase'], new['stroke_id']))
            a, b = new['start_index'], new['end_index']
            for field in ('velocity', 'acceleration', 'jerk'):
                np.testing.assert_allclose(self.new[field][[a, b]], 0., atol=1e-7)
            np.testing.assert_allclose(self.old['tip'][[old['start_index'], old['end_index']]],
                                       self.new['tip'][[a, b]], atol=1e-8)
            if new['phase'] != 'DRAW':
                self.assertEqual(old['duration_s'], new['duration_s'])

    def test_local_law_is_monotone_and_derivatives_agree(self):
        """测试 LocalTimeLaw 的严格单调性以及各阶链式求导公式与有限差分一致性。"""
        stroke = self.geometry['strokes'][12]
        arc = ArcCurve(self.geometry['segments'][stroke['draw_segment_ids'][0]]['curve'])
        law = LocalTimeLaw(arc, self.limits)
        tau = np.linspace(0., 1., 10001)
        fields = law.at(tau)
        self.assertEqual(fields[0][0], 0.)
        self.assertAlmostEqual(fields[0][-1], 1.)
        self.assertTrue(np.all(np.diff(fields[0]) >= 0.))
        for field in fields[1:]:
            np.testing.assert_allclose(field[[0, -1]], 0., atol=1e-8)
        step = 1e-6
        minus, plus = law.at(tau[3:-3] - step), law.at(tau[3:-3] + step)
        for order in range(3):
            numerical = (plus[order] - minus[order]) / (2 * step)
            np.testing.assert_allclose(numerical, fields[order+1][3:-3], rtol=.003, atol=.01)

    def test_local_timing_respects_cartesian_and_joint_limits(self):
        """测试局部时间重分配下，所有末端与关节动力学指标严格满足限制。"""
        for key, limit in (('velocity', self.limits.write_speed),

                           ('acceleration', self.limits.cart_accel),
                           ('jerk', self.limits.cart_jerk)):
            for record in self.new_report['segments']:
                a, b = record['start_index'], record['end_index']
                bound = self.limits.phase_speed(record['phase']) if key == 'velocity' else limit
                self.assertLessEqual(np.linalg.norm(self.new[key][a:b+1], axis=1).max(), bound * 1.001)
        for derivative, bound in zip(numerical_derivatives(self.new['q'], DT),
                                     (self.limits.joint_speed_deg, self.limits.joint_accel_deg,
                                      self.limits.joint_jerk_deg)):
            self.assertTrue(np.all(np.max(np.abs(np.rad2deg(derivative)), axis=0) <= bound * 1.001))
        # Include TXT rounding and independent FK in the production-size case.
        with tempfile.TemporaryDirectory() as directory:
            export_trajectory(Path(directory), self.geometry, self.new, self.new_report)

if __name__ == '__main__':
    unittest.main()
