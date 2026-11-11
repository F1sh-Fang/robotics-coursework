"""Layer 1: ordered, time-free pen-tip geometry and layout using HIT logo calligraphy."""
import json
import math
from pathlib import Path

import numpy as np
try:
    from scipy.integrate import cumulative_simpson
except ImportError:
    from scipy.integrate import cumtrapz as cumulative_simpson
from scipy.interpolate import BSpline, CubicSpline, PchipInterpolator

from .logo_extractor import get_logo_character_strokes

HERE = Path(__file__).resolve().parent.parent
PHASES = {'DESCEND': 0, 'DRAW': 1, 'LIFT': 2, 'TRANSFER': 3}


def point_polyline_distances(points, polyline):
    """Compute minimum Euclidean distance from each point to the 3D/2D polyline."""
    a, b = polyline[:-1], polyline[1:]
    d = b - a
    denom = np.maximum(np.sum(d * d, axis=1), 1e-24)
    w = points[:, None, :] - a
    fraction = np.clip(np.sum(w * d, axis=2) / denom, 0, 1)
    return np.sqrt(np.min(np.sum((w - fraction[:, :, None] * d) ** 2, axis=2), axis=1))


def simplify(points, tolerance):
    """Ramer-Douglas-Peucker simplification for 3D trajectory."""
    if len(points) <= 2:
        return points
    distances = point_polyline_distances(points[1:-1], points[[0, -1]])
    i = int(np.argmax(distances)) + 1
    if distances[i - 1] <= tolerance:
        return points[[0, -1]]
    return np.vstack([simplify(points[:i+1], tolerance)[:-1], simplify(points[i:], tolerance)])


def control_spline(points):
    """Clamped B-spline approximation; convex hull prevents boundary overshoot."""
    n = len(points)
    degree = min(5, n - 1)
    interior = np.linspace(0., 1., n - degree + 1)[1:-1]
    knots = np.r_[np.zeros(degree + 1), interior, np.ones(degree + 1)]
    return BSpline(knots, points, degree, extrapolate=False)


def fit_curve(points, tolerance=0.35):
    """Fit a clamped B-spline within tolerance to 3D stroke path."""
    points = points[np.r_[True, np.linalg.norm(np.diff(points, axis=0), axis=1) > 1e-9]]
    if len(points) < 2:
        raise ValueError('Zero-length stroke in character data')
    reference = points.copy()
    controls = points.copy()
    for _ in range(7):
        spline = control_spline(controls)
        grid = np.unique(np.r_[np.linspace(0, 1, max(501, len(controls) * 12)), spline.t])
        curve = spline(grid)
        error = max(float(point_polyline_distances(curve, reference).max()),
                    float(point_polyline_distances(reference, curve).max()))
        if error <= tolerance:
            return spline, error
        refined = np.empty((len(controls) * 2 - 1, 3))
        refined[::2] = controls
        refined[1::2] = (controls[:-1] + controls[1:]) / 2
        controls = refined
    raise ValueError('Could not fit stroke within geometric tolerance')


def encode_spline(spline):
    return {'degree': int(spline.k), 'knots': spline.t.tolist(), 'controls_mm': spline.c.tolist()}


def decode_spline(obj):
    return BSpline(np.asarray(obj['knots']), np.asarray(obj['controls_mm']), obj['degree'], extrapolate=False)


def split_corners(points, angle_deg=75):
    """Split at sharp corners (pauses/顿笔 in calligraphy) into separate continuous pieces."""
    edges = np.diff(points, axis=0)
    norm = np.linalg.norm(edges, axis=1)
    cosine = np.sum(edges[:-1] * edges[1:], axis=1) / (norm[:-1] * norm[1:])
    corners = np.flatnonzero(np.degrees(np.arccos(np.clip(cosine, -1, 1))) >= angle_deg) + 1
    ids = np.r_[0, corners, len(points) - 1]
    return [points[a:b+1] for a, b in zip(ids[:-1], ids[1:])]


def make_layout(text, x_range, y_range, margin, gap):
    """Layout characters along +Y direction; upper side of characters points to -X."""
    count = len(text)
    if count == 0:
        raise ValueError('Nonempty character string required')
    height = np.diff(x_range)[0] - 2 * margin
    width = np.diff(y_range)[0] - 2 * margin
    if min(height, width) <= 0:
        raise ValueError('Margins consume the whole writing area')

    # Single-line layout along +Y
    side = min(height, (width - (count - 1) * gap) / count)
    if side <= 0:
        raise ValueError('Area is too small for character spacing')

    total_w = count * side + (count - 1) * gap
    y0 = np.mean(y_range) - total_w / 2
    xmin = np.mean(x_range) - side / 2

    cells = []
    for c, char in enumerate(text):
        ymin = y0 + c * (side + gap)
        cells.append({
            'character': char,
            'character_index': c,
            'x_mm': [xmin, xmin + side],
            'y_mm': [ymin, ymin + side],
        })
    return side, cells


def build_geometry(text='哈工大', z=420., x_range=(450., 500.), y_range=(50., 150.),
                   lift=30., tip_offset=20., brush_depth=1.0, margin=2., gap=4.,
                   tolerance=0.35, corner_angle=75.):
    """Construct 3D writing geometry for HIT logo characters with brush press depth."""
    values = [z, *x_range, *y_range, lift, tip_offset, brush_depth, margin, gap, tolerance, corner_angle]
    if not np.isfinite(values).all() or x_range[0] >= x_range[1] or y_range[0] >= y_range[1]:
        raise ValueError('Finite coordinates and increasing XY ranges are required')
    if lift <= 0 or tip_offset < 0 or brush_depth < 0 or min(margin, gap) < 0 or tolerance <= 0 or not 0 < corner_angle < 180:
        raise ValueError('Invalid lift, pen offset, brush depth, margin, gap or curve settings')

    side, cells = make_layout(text, x_range, y_range, margin, gap)
    geometry = {
        'schema': 'writing-geometry-v2',
        'text': text,
        'units': 'mm',
        'tip_plane_z_mm': z,
        'tip_lift_mm': lift,
        'tip_offset_tool_z_mm': tip_offset,
        'brush_depth_mm': brush_depth,
        'x_range_mm': list(x_range),
        'y_range_mm': list(y_range),
        'reading_right': '+Y',
        'character_up': '-X',
        'cell_side_mm': side,
        'margin_mm': margin,
        'gap_mm': gap,
        'curve_tolerance_mm': tolerance,
        'cells': cells,
        'strokes': [],
        'segments': [],
    }
    segments = geometry['segments']

    def add_segment(phase, spline, stroke_id, character_index, fit_error=0.):
        segments.append({
            'segment_id': len(segments),
            'phase': phase,
            'stroke_id': stroke_id,
            'character_index': character_index,
            'curve': encode_spline(spline),
            'start_tip_mm': spline(0.).tolist(),
            'end_tip_mm': spline(1.).tolist(),
            'fit_error_mm': fit_error,
            'stop_at_start': True,
            'stop_at_end': True,
        })

    previous_high = None
    for cell in cells:
        raw_medians, raw_thickness, raw_metas = get_logo_character_strokes(cell['character'])
        cloud = np.vstack(raw_medians)
        lo, hi = cloud.min(axis=0), cloud.max(axis=0)
        extent = float(np.max(hi - lo))
        if extent <= 0:
            raise ValueError('Degenerate character bounding box')
        center = (lo + hi) / 2
        scale = side / extent

        for local_stroke, (points, thick) in enumerate(zip(raw_medians, raw_thickness)):
            stroke_meta = raw_metas[local_stroke] if local_stroke < len(raw_metas) else {}
            stroke_name = stroke_meta.get('name', f'Stroke-{local_stroke+1}')
            stroke_type = stroke_meta.get('type', 'stroke')

            # Font Y points upward; map font-right to +Y and font-up to -X.
            xy = (points - center) * scale
            # Brush depth feed: tip moves deeper (smaller Z) at thicker strokes
            z_stroke = z - brush_depth * np.asarray(thick, dtype=float)
            xyz = np.column_stack([
                np.mean(cell['x_mm']) - xy[:, 1],
                np.mean(cell['y_mm']) + xy[:, 0],
                z_stroke,
            ])
            xyz = xyz[np.r_[True, np.linalg.norm(np.diff(xyz, axis=0), axis=1) > 1e-9]]
            if len(xyz) < 2:
                raise ValueError('Zero-length character stroke')
            xyz = simplify(xyz, min(0.10, tolerance / 3))

            sid = len(geometry['strokes'])
            start_high = np.array([xyz[0, 0], xyz[0, 1], z + lift])
            end_high = np.array([xyz[-1, 0], xyz[-1, 1], z + lift])

            # Air transfer between strokes
            if previous_high is not None and np.linalg.norm(start_high - previous_high) > 1e-9:
                add_segment('TRANSFER', control_spline(np.array([previous_high, start_high])), sid, cell['character_index'])

            # Vertical descent to initial stroke contact
            add_segment('DESCEND', control_spline(np.array([start_high, xyz[0]])), sid, cell['character_index'])

            # 3D drawing strokes (corner-split for pauses/顿笔 without lifting off paper)
            draw_ids = []
            for piece in split_corners(xyz, corner_angle):
                curve, error = fit_curve(piece, tolerance)
                draw_ids.append(len(segments))
                add_segment('DRAW', curve, sid, cell['character_index'], error)

            # Vertical lift from final stroke release
            add_segment('LIFT', control_spline(np.array([xyz[-1], end_high])), sid, cell['character_index'])
            previous_high = end_high

            geometry['strokes'].append({
                'stroke_id': sid,
                'character_index': cell['character_index'],
                'character': cell['character'],
                'stroke_in_character': local_stroke + 1,
                'stroke_name': stroke_name,
                'stroke_type': stroke_type,
                'draw_segment_ids': draw_ids,
                'source_polyline_mm': xyz.tolist(),
            })

    validate_geometry(geometry)
    return geometry


def validate_geometry(geometry):
    """Verify continuity, limits, and vertical lift/descent semantics."""
    if geometry.get('schema') not in ('writing-geometry-v1', 'writing-geometry-v2'):
        raise ValueError('Unsupported geometry schema')
    z, lift = geometry['tip_plane_z_mm'], geometry['tip_lift_mm']
    brush_depth = geometry.get('brush_depth_mm', 0.0)
    if not np.isfinite([z, lift, geometry['tip_offset_tool_z_mm'], brush_depth]).all() or lift <= 0 or geometry['tip_offset_tool_z_mm'] < 0 or brush_depth < 0:
        raise ValueError('Invalid geometry plane/lift/tool offset/brush depth')

    previous = None
    segments = geometry['segments']
    if not segments or segments[0]['phase'] != 'DESCEND' or segments[-1]['phase'] != 'LIFT':
        raise ValueError('Geometry must begin with DESCEND and end with LIFT')

    for index, segment in enumerate(segments):
        if segment['segment_id'] != index or segment['phase'] not in PHASES:
            raise ValueError('Invalid segment identity or phase')
        spline = decode_spline(segment['curve'])
        points = spline(np.linspace(0, 1, 257))
        if not np.isfinite(points).all():
            raise ValueError('Non-finite curve')
        if previous is not None and np.linalg.norm(points[0] - previous) > 1e-7:
            raise ValueError(f'Geometric discontinuity at segment {index}')
        previous = points[-1]

        for axis, key in ((0, 'x_range_mm'), (1, 'y_range_mm')):
            lower, upper = geometry[key]
            if np.any(spline.c[:, axis] < lower - 1e-8) or np.any(spline.c[:, axis] > upper + 1e-8):
                raise ValueError('Curve control polygon exceeds XY bounds')

        phase = segment['phase']
        if phase == 'DRAW':
            if np.any(spline.c[:, 2] < z - brush_depth - 1e-6) or np.any(spline.c[:, 2] > z + 1e-6):
                raise ValueError('Writing curve leaves the tip paper/press zone')
        if phase == 'TRANSFER' and not np.allclose(spline.c[:, 2], z + lift, atol=1e-9, rtol=0):
            raise ValueError('Transfer curve leaves the clearance plane')
        if phase in ('DESCEND', 'LIFT'):
            if np.max(np.abs(spline.c[:, :2] - spline.c[0, :2])) > 1e-9:
                raise ValueError('Lift/descent is not vertical')
            if phase == 'DESCEND':
                if abs(points[0, 2] - (z + lift)) > 1e-7 or points[-1, 2] < z - brush_depth - 1e-6 or points[-1, 2] > z + 1e-6:
                    raise ValueError('Incorrect descent height')
            else:
                if abs(points[-1, 2] - (z + lift)) > 1e-7 or points[0, 2] < z - brush_depth - 1e-6 or points[0, 2] > z + 1e-6:
                    raise ValueError('Incorrect lift height')

    for stroke in geometry['strokes']:
        phases = [s['phase'] for s in segments if s['stroke_id'] == stroke['stroke_id']]
        if phases[0] == 'TRANSFER':
            phases = phases[1:]
        if len(phases) < 3 or phases[0] != 'DESCEND' or phases[-1] != 'LIFT' or set(phases[1:-1]) != {'DRAW'}:
            raise ValueError('Each stroke needs descent, drawing, and a full lift')


class ArcCurve:
    """Arc-length map with analytic geometric derivatives through order three."""
    def __init__(self, spec):
        self.spline = decode_spline(spec)
        self.derivatives = [self.spline.derivative(n) if n <= self.spline.k else None for n in (1, 2, 3)]
        u = np.unique(np.r_[np.linspace(0., 1., max(4097, len(self.spline.c) * 64 + 1)), self.spline.t])
        u = u[np.r_[True, np.diff(u) > 1e-12]]
        speed = np.linalg.norm(self.derivatives[0](u), axis=1)
        if speed.min() < 1e-7:
            raise ValueError('Curve has a cusp/zero tangent; split it into stopped pieces')
        s = cumulative_simpson(speed, x=u, initial=0.)
        self.length = float(s[-1])
        if self.length < 1e-8 or np.any(np.diff(s) <= 0):
            raise ValueError('Invalid arc-length map')
        self.s_of_u = CubicSpline(u, s, bc_type=((1, speed[0]), (1, speed[-1])))
        self.u_of_s = PchipInterpolator(s, u)

    def at(self, distance, derivatives=False):
        distance = np.clip(np.atleast_1d(distance), 0., self.length)
        u = np.clip(self.u_of_s(distance), 0., 1.)
        for _ in range(2):
            u = np.clip(u - (self.s_of_u(u) - distance) / self.s_of_u(u, 1), 0., 1.)
        p = self.spline(u)
        if not derivatives:
            return p
        a, b, c = [d(u) if d is not None else np.zeros_like(p) for d in self.derivatives]
        v = np.linalg.norm(a, axis=1)[:, None]
        h = np.sum(a * b, axis=1)[:, None]
        hp = np.sum(b * b + a * c, axis=1)[:, None]
        ps = a / v
        pss = b / v ** 2 - a * h / v ** 4
        psss = c / v ** 3 - 3 * b * h / v ** 5 - a * hp / v ** 5 + 4 * a * h * h / v ** 7
        return p, ps, pss, psss
