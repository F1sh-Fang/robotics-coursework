"""第一层：无时间 3D 几何路径生成与校徽书法排版模块 (Layer 1 Geometry)。

主要功能：
  1. 读取哈工大校徽提取的书法字形中心线与笔宽数据；
  2. 沿阅读方向 (+Y) 和字形上方 (-X) 进行字框约束排版；
  3. 将笔画粗细映射为毛笔笔尖的 Z 向下压进给（越粗越下压，软毫受压铺开）；
  4. 对尖锐折角进行“纸面顿笔停稳”分段打断，保留书法气势；
  5. 拟合端点夹持的高阶 B 样条曲线，并建立高精度弧长参数化映射与 1~3 阶几何导数解析求导。
"""
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

# 动作阶段枚举定义：
# 0: DESCEND (下落落笔) - 垂直下降接触纸面
# 1: DRAW (纸面书写) - 接触纸面并根据粗细下压运笔
# 2: LIFT (提笔收笔) - 笔画完成垂直提升至安全高度
# 3: TRANSFER (空中平移) - 安全平面内平移至下一笔起点上方
PHASES = {'DESCEND': 0, 'DRAW': 1, 'LIFT': 2, 'TRANSFER': 3}


def point_polyline_distances(points, polyline):
    """计算三维或二维空间中一组点到给定折线段的最小欧氏距离。

    采用向量投影法：将测试点投影到折线每条线段上，限制投影比例在 [0, 1] 区间内，
    求点到线段的最短距离，并取所有线段距离的最小值。

    Args:
        points: 测试点集，形状为 (N, D)
        polyline: 构成折线的顶点序列，形状为 (M, D)

    Returns:
        np.ndarray: 每个测试点到折线的最小距离，形状为 (N,)
    """
    a, b = polyline[:-1], polyline[1:]
    d = b - a
    denom = np.maximum(np.sum(d * d, axis=1), 1e-24)
    w = points[:, None, :] - a
    # 计算点在每条线段方向上的正交投影标量比例，截断在 [0, 1]
    fraction = np.clip(np.sum(w * d, axis=2) / denom, 0, 1)
    # 计算垂直投影误差向量的欧氏模长
    return np.sqrt(np.min(np.sum((w - fraction[:, :, None] * d) ** 2, axis=2), axis=1))


def simplify(points, tolerance):
    """采用 Ramer-Douglas-Peucker (RDP) 算法对空间折线进行特征保留抽稀。

    递归寻找偏离首尾基线最远的点，若最大偏离距离大于 tolerance 则在该点递归切分，
    否则直接用首尾两点代替整段折线。

    Args:
        points: 折线点集 (N, 3)
        tolerance: 几何最大允许偏离容差，单位 mm

    Returns:
        np.ndarray: 抽稀后的精简折线点集 (K, 3)
    """
    if len(points) <= 2:
        return points
    distances = point_polyline_distances(points[1:-1], points[[0, -1]])
    i = int(np.argmax(distances)) + 1
    if distances[i - 1] <= tolerance:
        return points[[0, -1]]
    return np.vstack([simplify(points[:i+1], tolerance)[:-1], simplify(points[i:], tolerance)])


def control_spline(points):
    """构造以输入点集为控制点的端点夹持型 (Clamped) B 样条曲线。

    端点重复度等于 degree+1，保证曲线严格穿过起点和终点，且位于控制多边形的凸包内部，
    杜绝样条曲线在两端发生越界超调 (Overshoot)。

    Args:
        points: 控制点坐标矩阵 (N, 3)

    Returns:
        scipy.interpolate.BSpline: 夹持型 B 样条对象
    """
    n = len(points)
    degree = min(5, n - 1)
    interior = np.linspace(0., 1., n - degree + 1)[1:-1]
    # 节点向量两端填充 degree+1 个 0 和 1
    knots = np.r_[np.zeros(degree + 1), interior, np.ones(degree + 1)]
    return BSpline(knots, points, degree, extrapolate=False)


def fit_curve(points, tolerance=0.35):
    """自适应细化并拟合端点夹持 B 样条，确保与原始笔画路径的双向最大误差小于容差。

    若当前控制多边形生成的样条曲线最大偏离超出 tolerance，则在相邻控制点间插值细分控制网格，
    最多迭代 7 次，直至收敛。

    Args:
        points: 原始笔画采样点序列 (N, 3)，单位 mm
        tolerance: 最大几何拟合允许误差，默认 0.35 mm

    Returns:
        tuple[BSpline, float]:
            - spline: 拟合得到的 B 样条曲线
            - error: 最终实际的双向最大几何误差，单位 mm
    """
    # 过滤重合或极近的点
    points = points[np.r_[True, np.linalg.norm(np.diff(points, axis=0), axis=1) > 1e-9]]
    if len(points) < 2:
        raise ValueError('Zero-length stroke in character data')
    reference = points.copy()
    controls = points.copy()
    for _ in range(7):
        spline = control_spline(controls)
        grid = np.unique(np.r_[np.linspace(0, 1, max(501, len(controls) * 12)), spline.t])
        curve = spline(grid)
        # 双向 Hausdorff 距离近似：曲线到参考折线 & 参考折线到曲线
        error = max(float(point_polyline_distances(curve, reference).max()),
                    float(point_polyline_distances(reference, curve).max()))
        if error <= tolerance:
            return spline, error
        # 中点插入细化控制多边形
        refined = np.empty((len(controls) * 2 - 1, 3))
        refined[::2] = controls
        refined[1::2] = (controls[:-1] + controls[1:]) / 2
        controls = refined
    raise ValueError('Could not fit stroke within geometric tolerance')


def encode_spline(spline):
    """将 SciPy B 样条对象序列化为纯字典格式（便于导出 JSON）。"""
    return {'degree': int(spline.k), 'knots': spline.t.tolist(), 'controls_mm': spline.c.tolist()}


def decode_spline(obj):
    """从字典反序列化重构 SciPy B 样条对象。"""
    return BSpline(np.asarray(obj['knots']), np.asarray(obj['controls_mm']), obj['degree'], extrapolate=False)


def split_corners(points, angle_deg=75):
    """在笔画拐角折向过大处切分为独立段（对应书法中的“驻笔/顿笔变向”）。

    书法中“横折”、“竖折”等复合笔画内部不抬笔，但在折角处速度必须归零停稳（顿笔），
    因此在几何层将折角拆分为首尾相接的独立连续段。

    Args:
        points: 笔画三维坐标序列 (N, 3)
        angle_deg: 判定为折角的夹角阈值（单位度，默认 75°）

    Returns:
        list[np.ndarray]: 拆分后的平滑连续线段列表，每段首尾点严格贴合
    """
    edges = np.diff(points, axis=0)
    norm = np.linalg.norm(edges, axis=1)
    cosine = np.sum(edges[:-1] * edges[1:], axis=1) / (norm[:-1] * norm[1:])
    # 计算相邻线段折角，找出夹角超出阈值的拐点
    corners = np.flatnonzero(np.degrees(np.arccos(np.clip(cosine, -1, 1))) >= angle_deg) + 1
    ids = np.r_[0, corners, len(points) - 1]
    return [points[a:b+1] for a, b in zip(ids[:-1], ids[1:])]


def make_layout(text, x_range, y_range, margin, gap):
    """沿 +Y 轴（阅读前进方向）对文字进行等比例缩放与居中排版。

    坐标系映射：
      - 文字行进排版方向：+Y 轴
      - 文字正上方方向：-X 轴
      - 纸面法线向上：+Z 轴

    Args:
        text: 待书写文本字符串（如 '哈工大'）
        x_range: 工作空间 X 范围 [xmin, xmax]，对应字框高度约束
        y_range: 工作空间 Y 范围 [ymin, ymax]，对应排版宽度约束
        margin: 工作区外边距，单位 mm
        gap: 相邻两字之间的间隙，单位 mm

    Returns:
        tuple[float, list[dict]]:
            - side: 每个文字正方形字框的边长，单位 mm
            - cells: 包含每个文字在基坐标系下边界范围的字典列表
    """
    count = len(text)
    if count == 0:
        raise ValueError('Nonempty character string required')
    height = np.diff(x_range)[0] - 2 * margin
    width = np.diff(y_range)[0] - 2 * margin
    if min(height, width) <= 0:
        raise ValueError('Margins consume the whole writing area')

    # 计算能够容纳全部字符与间距的最大正方形字框边长
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
    """构建哈工大校徽书法的 3D 几何路径，融合毛笔下压进给模型与动作阶段序列。

    动作阶段生成规则：
      1. 第一笔开始前，笔尖从空中下降 DESCEND 至接触面；
      2. 笔画之间：若为同一笔画（如“工”字三笔连写），笔尖始终在纸面连续 DRAW；
         若为不同笔画，执行 LIFT -> TRANSFER -> DESCEND 循环；
      3. 笔画粗细控制：Z = z - brush_depth * thickness，粗笔画下压更深，铺毫更饱满。

    Args:
        text: 待书写文字
        z: 笔尖刚接触纸面的基准高度 (mm)
        x_range: 工作空间 X 范围 (mm)
        y_range: 工作空间 Y 范围 (mm)
        lift: 抬笔安全高度 (mm)
        tip_offset: 工具末端沿 +Z 轴到笔尖的长度偏移 (mm)
        brush_depth: 毛笔最大下压深度进给量 (mm)，0 表示平面书写
        margin: 排版边界裕量 (mm)
        gap: 字间距 (mm)
        tolerance: B 样条拟合几何容差 (mm)
        corner_angle: 顿笔折角打断阈值 (deg)

    Returns:
        dict: 完整的几何描述字典 (schema: writing-geometry-v2)
    """
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
        """辅助闭包：向几何描述中添加一段连续的曲线段"""
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
        # 从校徽提取模块获取真实字形骨架中心线与笔宽分布
        raw_medians, raw_thickness, raw_metas = get_logo_character_strokes(cell['character'])
        cloud = np.vstack(raw_medians)
        lo, hi = cloud.min(axis=0), cloud.max(axis=0)
        extent = float(np.max(hi - lo))
        if extent <= 0:
            raise ValueError('Degenerate character bounding box')
        center = (lo + hi) / 2
        scale = side / extent

        # “工”字采用经典连笔，三笔一气呵成不抬笔
        continuous = cell['character'] == '工'
        previous_contact = None
        for local_stroke, (points, thick) in enumerate(zip(raw_medians, raw_thickness)):
            stroke_meta = raw_metas[local_stroke] if local_stroke < len(raw_metas) else {}
            stroke_name = stroke_meta.get('name', f'Stroke-{local_stroke+1}')
            stroke_type = stroke_meta.get('type', 'stroke')

            # 字体坐标转换：字体向右对齐工作坐标系 +Y，字体向上对齐工作坐标系 -X
            xy = (points - center) * scale
            # 毛笔深度进给控制：笔画越粗，Z 向下压越深 (Z 值越小)
            z_stroke = z - brush_depth * np.asarray(thick, dtype=float)
            xyz = np.column_stack([
                np.mean(cell['x_mm']) - xy[:, 1],
                np.mean(cell['y_mm']) + xy[:, 0],
                z_stroke,
            ])
            xyz = xyz[np.r_[True, np.linalg.norm(np.diff(xyz, axis=0), axis=1) > 1e-9]]
            if len(xyz) < 2:
                raise ValueError('Zero-length character stroke')
            # 抽稀降低离散台阶噪声
            xyz = simplify(xyz, min(0.10, tolerance / 3))

            sid = len(geometry['strokes'])
            start_high = np.array([xyz[0, 0], xyz[0, 1], z + lift])
            end_high = np.array([xyz[-1, 0], xyz[-1, 1], z + lift])

            draw_ids = []
            if continuous and local_stroke > 0:
                # “工”字连笔：笔尖保持在纸面接触过渡，不抬入空中
                if np.linalg.norm(xyz[0] - previous_contact) > 1e-9:
                    draw_ids.append(len(segments))
                    add_segment('DRAW', control_spline(np.array([previous_contact, xyz[0]])),
                                sid, cell['character_index'])
            else:
                # 正常笔画：若存在上一抬笔点，先在空中平移 TRANSFER，再垂直下降 DESCEND
                if previous_high is not None and np.linalg.norm(start_high - previous_high) > 1e-9:
                    add_segment('TRANSFER', control_spline(np.array([previous_high, start_high])), sid, cell['character_index'])
                add_segment('DESCEND', control_spline(np.array([start_high, xyz[0]])), sid, cell['character_index'])

            # 纸面书写段：在急转折角处切分，实现顿笔停稳蓄势，转折不离纸
            for piece in split_corners(xyz, corner_angle):
                curve, error = fit_curve(piece, tolerance)
                draw_ids.append(len(segments))
                add_segment('DRAW', curve, sid, cell['character_index'], error)

            previous_contact = xyz[-1]
            # “工”字仅在最后一笔结束时才抬笔；其余字每笔结束均垂直提笔 LIFT
            if not continuous or local_stroke == len(raw_medians) - 1:
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
    """严格检验几何路径的连续性、边界范围、升降垂直度与落笔深度约束。

    检验项包含：
      1. 第一段必须为 DESCEND，最后一段必须为 LIFT；
      2. 相邻段端点欧氏距离必须小于 1e-7 mm（严格 $C^0$ 空间几何连续）；
      3. 所有曲线控制多边形均不得超出设定的 XY 矩形工作区域；
      4. 书写段 DRAW 的 Z 坐标必须落在 $[Z_{\\text{plane}} - \\Delta z_{\\max}, Z_{\\text{plane}}]$ 内；
      5. 空移段 TRANSFER 的 Z 坐标必须严格保持在 $Z_{\\text{plane}} + \\text{lift}$；
      6. 升降段 DESCEND 和 LIFT 的水平位移偏差必须小于 1e-9 mm（严格竖直升降）。
    """
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

        # 检查 XY 工作空间边界
        for axis, key in ((0, 'x_range_mm'), (1, 'y_range_mm')):
            lower, upper = geometry[key]
            if np.any(spline.c[:, axis] < lower - 1e-8) or np.any(spline.c[:, axis] > upper + 1e-8):
                raise ValueError('Curve control polygon exceeds XY bounds')

        # 检查 Z 方向运动学约束
        phase = segment['phase']
        if phase == 'DRAW':
            if np.any(spline.c[:, 2] < z - brush_depth - 1e-6) or np.any(spline.c[:, 2] > z + 1e-6):
                raise ValueError('Writing curve leaves the tip paper/press zone')
        if phase == 'TRANSFER' and not np.allclose(spline.c[:, 2], z + lift, atol=1e-9, rtol=0):
            raise ValueError('Transfer curve leaves the clearance plane')
        if phase in ('DESCEND', 'LIFT'):
            # 严格竖直升降检验
            if np.max(np.abs(spline.c[:, :2] - spline.c[0, :2])) > 1e-9:
                raise ValueError('Lift/descent is not vertical')
            if phase == 'DESCEND':
                if abs(points[0, 2] - (z + lift)) > 1e-7 or points[-1, 2] < z - brush_depth - 1e-6 or points[-1, 2] > z + 1e-6:
                    raise ValueError('Incorrect descent height')
            else:
                if abs(points[-1, 2] - (z + lift)) > 1e-7 or points[0, 2] < z - brush_depth - 1e-6 or points[0, 2] > z + 1e-6:
                    raise ValueError('Incorrect lift height')

    # 笔画状态完整性检验
    for stroke in geometry['strokes']:
        phases = [s['phase'] for s in segments if s['stroke_id'] == stroke['stroke_id']]
        if phases[0] == 'TRANSFER':
            phases = phases[1:]
        siblings = [s for s in geometry['strokes'] if s['character_index'] == stroke['character_index']]
        continuous = stroke['character'] == '工'
        needs_descent = not continuous or stroke['stroke_id'] == siblings[0]['stroke_id']
        needs_lift = not continuous or stroke['stroke_id'] == siblings[-1]['stroke_id']
        if needs_descent:
            if not phases or phases.pop(0) != 'DESCEND':
                raise ValueError('Pen-down run must begin with descent')
        if needs_lift:
            if not phases or phases.pop() != 'LIFT':
                raise ValueError('Pen-down run must end with a full lift')
        if not phases or set(phases) != {'DRAW'}:
            raise ValueError('Stroke must draw without unexpected pen lifts')


class ArcCurve:
    """高精度弧长参数化映射与 1~3 阶空间几何导数解析计算类。

    数学原理：
      样条曲线以参数 u 为自变量 p(u)。弧长定义为 s(u) = ∫_0^u ||p'(ξ)|| dξ。
      通过数值积分构建 u(s) 映射，并用牛顿迭代纠正残差达到机器精度。
      曲线关于弧长 s 的几何导数采用链式法则求得：
        p_s = dp/ds = a / ||a||                      (单位切向量，模长为 1)
        p_ss = d²p/ds² = b / v² - a (a·b) / v⁴      (主法向曲率向量)
        p_sss = d³p/ds³ (三阶挠率/加加速度几何导数)
    """
    def __init__(self, spec):
        self.spline = decode_spline(spec)
        # 预先计算样条曲线关于原始参数 u 的 1、2、3 阶导数
        self.derivatives = [self.spline.derivative(n) if n <= self.spline.k else None for n in (1, 2, 3)]
        u = np.unique(np.r_[np.linspace(0., 1., max(4097, len(self.spline.c) * 64 + 1)), self.spline.t])
        u = u[np.r_[True, np.diff(u) > 1e-12]]
        speed = np.linalg.norm(self.derivatives[0](u), axis=1)
        if speed.min() < 1e-7:
            raise ValueError('Curve has a cusp/zero tangent; split it into stopped pieces')
        # 辛普森数值累积积分计算弧长
        s = cumulative_simpson(speed, x=u, initial=0.)
        self.length = float(s[-1])
        if self.length < 1e-8 or np.any(np.diff(s) <= 0):
            raise ValueError('Invalid arc-length map')
        # 双向高保真插值器
        self.s_of_u = CubicSpline(u, s, bc_type=((1, speed[0]), (1, speed[-1])))
        self.u_of_s = PchipInterpolator(s, u)

    def at(self, distance, derivatives=False):
        """根据弧长计算空间坐标及 1~3 阶几何导数。

        Args:
            distance: 弧长距离标量或数组，单位 mm
            derivatives: 是否计算几何导数，默认为 False

        Returns:
            若 derivatives 为 False，返回三维点坐标 p (N, 3)；
            若 derivatives 为 True，返回 (p, p_s, p_ss, p_sss)，分别对应位置及对弧长的 1, 2, 3 阶导数。
        """
        distance = np.clip(np.atleast_1d(distance), 0., self.length)
        u = np.clip(self.u_of_s(distance), 0., 1.)
        # 2 次牛顿-拉夫逊迭代将残差推入机器精度
        for _ in range(2):
            u = np.clip(u - (self.s_of_u(u) - distance) / self.s_of_u(u, 1), 0., 1.)
        p = self.spline(u)
        if not derivatives:
            return p

        # a = dp/du, b = d²p/du², c = d³p/du³
        a, b, c = [d(u) if d is not None else np.zeros_like(p) for d in self.derivatives]
        v = np.linalg.norm(a, axis=1)[:, None]
        h = np.sum(a * b, axis=1)[:, None]
        hp = np.sum(b * b + a * c, axis=1)[:, None]
        # 解析计算关于弧长 s 的几何导数：
        ps = a / v
        pss = b / v ** 2 - a * h / v ** 4
        psss = c / v ** 3 - 3 * b * h / v ** 5 - a * hp / v ** 5 + 4 * a * h * h / v ** 7
        return p, ps, pss, psss

