"""Unit tests for HIT logo character writing, brush press depth, and dynamics."""
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
from writing.timing import DT, MotionLimits, numerical_derivatives, plan_trajectory, required_stretch


class LogoGeometryTests(unittest.TestCase):
    def test_cli_help(self):
        result = subprocess.run([sys.executable, '-B', str(HERE / 'GenerateWritingTrajectory.py'), '--help'],
                                capture_output=True)
        self.assertEqual(result.returncode, 0)
        self.assertIn(b'--brush-depth', result.stdout)

    def test_ha_gong_da_strokes_and_layout(self):
        g = build_geometry()
        # 15 strokes total for authentic calligraphy: 哈 (9) + 工 (3) + 大 (3)
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
        # When brush_depth=0, all draw points must strictly equal z
        g_flat = build_geometry(brush_depth=0.0)
        for s in g_flat['segments']:
            if s['phase'] == 'DRAW':
                curve = ArcCurve(s['curve'])
                pts = curve.at(np.linspace(0, curve.length, 51))
                np.testing.assert_allclose(pts[:, 2], 420.0, atol=1e-9)

        # When brush_depth=1.5, draw points must vary between 418.5 and 420.0
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
    @classmethod
    def setUpClass(cls):
        cls.geometry = build_geometry(brush_depth=1.0)
        cls.limits = MotionLimits(write_speed=40., travel_speed=60., vertical_speed=20.)
        cls.trajectory, cls.report = plan_trajectory(cls.geometry, cls.limits)

    def test_timing_minimum_and_subsecond_segments(self):
        durations = [record['duration_s'] for record in self.report['segments']]
        self.assertGreaterEqual(min(durations), 0.5)
        self.assertLess(min(durations), 1.0)
        zeros = np.zeros((3, 3))
        joint_zeros = np.zeros((3, 6))
        self.assertEqual(required_stretch((zeros, zeros, zeros),
                                         (joint_zeros, joint_zeros, joint_zeros),
                                         self.limits, 'DRAW'), 0.)

    def test_forward_kinematics_and_timeline(self):
        t = self.trajectory['time']
        q = self.trajectory['q']
        tip = self.trajectory['tip']
        self.assertAlmostEqual(np.diff(t)[0], DT)
        recovered, _, rotations = forward_tip(q, self.geometry['tip_offset_tool_z_mm'])
        max_error = np.linalg.norm(recovered - tip, axis=1).max()
        self.assertLess(max_error, 1e-5)
        self.assertLess(np.max(np.abs(rotations - R_HOLD)), 1e-8)

    def test_joint_limits_and_dynamics(self):
        q = self.trajectory['q']
        qd = numerical_derivatives(q, DT)
        qdeg = np.rad2deg(q)
        # Check position limits
        for i in range(6):
            self.assertTrue(np.all(qdeg[:, i] >= self.limits.joint_limits_deg[i, 0]))
            self.assertTrue(np.all(qdeg[:, i] <= self.limits.joint_limits_deg[i, 1]))
        # Check velocity, acceleration, jerk bounds
        for deriv, bound in zip(qd, (self.limits.joint_speed_deg, self.limits.joint_accel_deg, self.limits.joint_jerk_deg)):
            peak = np.max(np.abs(np.rad2deg(deriv)), axis=0)
            self.assertTrue(np.all(peak <= bound * 1.001))

    def test_export_file_verification(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            out = Path(temp_dir)
            export_trajectory(out, self.geometry, self.trajectory, self.report)
            deg_file = out / 'writing_joint_deg.txt'
            rad_file = out / 'writing_joint_timed_rad.txt'
            self.assertTrue(deg_file.exists())
            self.assertTrue(rad_file.exists())
            self.assertEqual(len(deg_file.read_text().strip().splitlines()), len(self.trajectory['time']))


if __name__ == '__main__':
    unittest.main()
