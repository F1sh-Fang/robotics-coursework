"""校徽单字图像分割、真实书法骨架提取与笔画粗细信息生成模块。

主要功能：
  1. 从哈工大官方校徽 logo.png 中分割提取“哈”、“工”、“大”三个毛体书法单字；
  2. 采用欧氏距离变换 (Distance Transform) 计算字迹各处的物理笔宽；
  3. 采用 Zhang-Suen 形态学细化算法提取单像素宽的真实书法骨架；
  4. 利用骨架测地线图搜索 (Geodesic BFS) 沿汉字规范书写笔顺提取中心线点列；
  5. 进行等距弧长插值与粗细归一化，保存至 logo/hit_strokes.json 供轨迹规划使用。
"""
from collections import deque
from pathlib import Path
import json
import cv2
import numpy as np
from PIL import Image
from scipy.interpolate import CubicSpline

HERE = Path(__file__).resolve().parent.parent
LOGO_DIR = HERE / 'logo'
EXTRACTED_DIR = LOGO_DIR / 'extracted'
STROKES_JSON = LOGO_DIR / 'hit_strokes.json'


def extract_characters(logo_path=LOGO_DIR/'logo.png', out_dir=EXTRACTED_DIR):
    """从哈尔滨工业大学校徽图像中分割提取“哈”、“工”、“大”单字图像，并计算距离变换与骨架。

    处理步骤：
      1. 读取校徽透明度 Alpha 通道以获取高对比度文字轮廓；
      2. 根据像素坐标框裁剪“哈”、“工”、“大”区域；
      3. 连通域分析滤除孤立噪点，紧致外接矩形自适应裁剪加边距；
      4. 计算欧氏距离变换图（保存为 *_thickness.png，像素值代表笔画半径）；
      5. 细化提取单像素骨架（保存为 *_true_skel.png）；
      6. 输出单字黑白图像与三字拼接图。

    Args:
        logo_path: 校徽图片路径
        out_dir: 输出保存目录

    Returns:
        dict: 包含三字图像、尺寸、距离变换矩阵与骨架二值图的字典
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    img = Image.open(logo_path)
    alpha = np.array(img)[:, :, 3]

    # 校徽 logo.png 中“哈”、“工”、“大”所在像素坐标包围盒 (x1, x2, y1, y2, 汉字)
    char_boxes = {
        'ha': (845, 1155, 115, 430, '哈'),
        'gong': (1785, 2000, 195, 405, '工'),
        'da': (2345, 2615, 115, 425, '大'),
    }

    results = {}
    crops = []

    for key, (x1, x2, y1, y2, char_text) in char_boxes.items():
        region = alpha[y1:y2, x1:x2]
        mask = (region > 80).astype(np.uint8) * 255

        # 连通域分析滤除微小杂色噪点 (面积阈值 80 像素)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
        clean = np.zeros_like(mask)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] > 80:
                clean[labels == i] = 255

        # 紧致边缘裁剪并留出 pad 像素内边距
        coords = np.argwhere(clean > 0)
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        pad = 10
        h, w = clean.shape
        y_min = max(0, y_min - pad)
        y_max = min(h, y_max + pad + 1)
        x_min = max(0, x_min - pad)
        x_max = min(w, x_max + pad + 1)

        char_crop = clean[y_min:y_max, x_min:x_max]

        # 保存白底黑字的单字灰度图
        cv2.imwrite(str(out_dir / f'{key}.png'), 255 - char_crop)

        # 欧氏距离变换 (Distance Transform)：计算每个笔画内像素到轮廓的最短距离
        binary_ink = (char_crop > 0).astype(np.uint8)
        dist = cv2.distanceTransform(binary_ink, cv2.DIST_L2, 5)
        dist_vis = (dist / max(1e-3, dist.max()) * 255).astype(np.uint8)
        cv2.imwrite(str(out_dir / f'{key}_thickness.png'), dist_vis)

        # 形态学细化提取单像素真实中心线骨架
        try:
            skel = cv2.ximgproc.thinning(binary_ink * 255)
        except AttributeError:
            from skimage.morphology import skeletonize
            skel = (skeletonize(binary_ink) * 255).astype(np.uint8)
        cv2.imwrite(str(out_dir / f'{key}_true_skel.png'), skel)

        results[key] = {
            'character': char_text,
            'image_path': str(out_dir / f'{key}.png'),
            'size': (char_crop.shape[1], char_crop.shape[0]),
            'dist': dist,
            'skel': skel > 0,
        }
        crops.append(char_crop)

    # 拼接三字预览对比图像
    max_h = max(c.shape[0] for c in crops)
    padded_crops = []
    for c in crops:
        top = (max_h - c.shape[0]) // 2
        bottom = max_h - c.shape[0] - top
        padded = cv2.copyMakeBorder(c, top, bottom, 20, 20, cv2.BORDER_CONSTANT, value=0)
        padded_crops.append(padded)
    combined = np.hstack(padded_crops)
    cv2.imwrite(str(out_dir / 'ha_gong_da_combined.png'), 255 - combined)

    return results


def shortest_skeleton_path(skel, start_pt, end_pt):
    """在单像素骨架的 8 邻域连通图上，通过广度优先搜索 (BFS) 计算两点间的最短测地线路径。

    Args:
        skel: 骨架布尔掩膜 (H, W)
        start_pt: 起点坐标 (x, y)
        end_pt: 终点坐标 (x, y)

    Returns:
        list[tuple[int, int]]: 严格位于骨架上的连续像素坐标序列 [(x0, y0), (x1, y1), ...]
    """
    pts = set(map(tuple, np.argwhere(skel > 0)))
    pts_arr = np.array(list(pts))
    # 匹配离指定坐标最近的真实骨架像素
    d_start = np.hypot(pts_arr[:, 1] - start_pt[0], pts_arr[:, 0] - start_pt[1])
    d_end = np.hypot(pts_arr[:, 1] - end_pt[0], pts_arr[:, 0] - end_pt[1])
    s_y, s_x = pts_arr[np.argmin(d_start)]
    e_y, e_x = pts_arr[np.argmin(d_end)]

    # 8 邻域 BFS 搜索队列
    queue = deque([(s_y, s_x)])
    parent = {(s_y, s_x): None}
    while queue:
        curr = queue.popleft()
        if curr == (e_y, e_x):
            break
        cy, cx = curr
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                ny, nx = cy + dy, cx + dx
                if (ny, nx) in pts and (ny, nx) not in parent:
                    parent[(ny, nx)] = curr
                    queue.append((ny, nx))

    # 回溯父节点重构有序路径
    path = []
    curr = (e_y, e_x)
    while curr is not None:
        path.append((curr[1], curr[0]))  # 存储为 (x, y) 格式
        curr = parent.get(curr)
    path.reverse()
    return path


def smooth_resample(path, dist, num_samples=35):
    """沿骨架路径按弧长等距重采样，并从欧氏距离变换图查询局部笔画粗细。

    Args:
        path: 离散骨架像素点集 [(x, y), ...]
        dist: 距离变换矩阵 (H, W)，像素值即为墨迹半径
        num_samples: 重采样采样点数

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - pts: 等距平滑后的三维空间/平面点集 (num_samples, 2)
            - thickness: 归一化局部粗细分布序列 (num_samples,)，取值范围 [0.15, 1.0]
    """
    path = np.array(path, dtype=float)
    # 去除相邻重复像素
    diffs = np.diff(path, axis=0)
    keep = np.r_[True, np.hypot(diffs[:, 0], diffs[:, 1]) > 1e-6]
    path = path[keep]
    if len(path) < 2:
        return path, np.ones(len(path)) * 0.5

    # 累加线段弧长
    diffs = np.diff(path, axis=0)
    segs = np.hypot(diffs[:, 0], diffs[:, 1])
    dists = np.r_[0, np.cumsum(segs)]
    total_len = dists[-1]

    # 三次样条插值得到等距连续坐标
    sample_dists = np.linspace(0, total_len, num_samples)
    cs_x = CubicSpline(dists, path[:, 0])
    cs_y = CubicSpline(dists, path[:, 1])

    sampled_x = cs_x(sample_dists)
    sampled_y = cs_y(sample_dists)

    # 在距离变换矩阵中双线性查询每个采样点处的实际墨迹半径
    h, w = dist.shape
    thickness = []
    max_d = max(26.0, float(dist.max()))
    for px, py in zip(sampled_x, sampled_y):
        ix = int(np.clip(round(px), 0, w - 1))
        iy = int(np.clip(round(py), 0, h - 1))
        r = float(dist[iy, ix])
        # 归一化相对粗细至 [0.15, 1.0]，避免出锋处笔尖脱离纸面
        thickness.append(np.clip(r / max_d, 0.15, 1.0))

    pts = np.column_stack([sampled_x, sampled_y])
    return pts, np.array(thickness)


def build_logo_strokes():
    """提取哈工大校徽真实书法笔画数据库，并序列化保存为 JSON。

    笔顺定义：
      - “工”字 3 笔：上横 -> 顺势斜切中竖 -> 劲挺长挑底横（连笔运笔）；
      - “大”字 3 笔：长横 -> 舒展长撇 -> 沉稳右捺；
      - “哈”字 9 笔：左口 3 笔（左竖、横折、底横） + 右合 6 笔（人字撇、捺、中横、下口竖、横折、底横）。
    """
    chars_data = extract_characters()

    # 1. 工字 3 笔
    gong_skel = chars_data['gong']['skel']
    gong_dist = chars_data['gong']['dist']
    gong_strokes_def = [
        {'name': '工字上横', 'type': '横', 'raw_path': shortest_skeleton_path(gong_skel, (79, 53), (173, 36))},
        {'name': '工字中竖', 'type': '竖', 'raw_path': shortest_skeleton_path(gong_skel, (165, 40), (39, 170))},
        {'name': '工字下横', 'type': '横', 'raw_path': shortest_skeleton_path(gong_skel, (39, 170), (176, 117))},
    ]

    # 2. 大字 3 笔
    da_skel = chars_data['da']['skel']
    da_dist = chars_data['da']['dist']
    da_strokes_def = [
        {'name': '大字长横', 'type': '横', 'raw_path': shortest_skeleton_path(da_skel, (43, 209), (209, 124))},
        {'name': '大字长撇', 'type': '撇', 'raw_path': shortest_skeleton_path(da_skel, (136, 33), (24, 279))},
        {'name': '大字长捺', 'type': '捺', 'raw_path': shortest_skeleton_path(da_skel, (120, 197), (249, 261))},
    ]

    # 3. 哈字 9 笔（规范书法笔顺：左口 3 笔 + 右合 6 笔，横折内部不离纸）
    ha_skel = chars_data['ha']['skel']
    ha_dist = chars_data['ha']['dist']
    ha_strokes_def = [
        {'name': '左口左竖', 'type': '竖', 'raw_path': shortest_skeleton_path(ha_skel, (36, 117), (43, 184))},
        {'name': '左口横折', 'type': '横折', 'raw_path': shortest_skeleton_path(ha_skel, (80, 105), (118, 106)) + shortest_skeleton_path(ha_skel, (118, 106), (111, 163))},
        {'name': '左口底横', 'type': '横', 'raw_path': shortest_skeleton_path(ha_skel, (43, 184), (111, 163))},
        {'name': '右合人字撇', 'type': '撇', 'raw_path': shortest_skeleton_path(ha_skel, (207, 22), (28, 268))},
        {'name': '右合人字捺', 'type': '捺', 'raw_path': shortest_skeleton_path(ha_skel, (207, 22), (282, 174))},
        {'name': '右合中横', 'type': '横', 'raw_path': shortest_skeleton_path(ha_skel, (150, 180), (195, 181))},
        {'name': '下口左竖', 'type': '竖', 'raw_path': shortest_skeleton_path(ha_skel, (164, 237), (139, 283))},
        {'name': '下口横折', 'type': '横折', 'raw_path': shortest_skeleton_path(ha_skel, (164, 237), (206, 231)) + shortest_skeleton_path(ha_skel, (206, 231), (207, 276))},
        {'name': '下口底横', 'type': '横', 'raw_path': shortest_skeleton_path(ha_skel, (139, 283), (207, 276))},
    ]

    char_defs = {
        '哈': (ha_strokes_def, ha_dist),
        '工': (gong_strokes_def, gong_dist),
        '大': (da_strokes_def, da_dist),
    }

    normalized_library = {}
    for char, (strokes_def, dist) in char_defs.items():
        h, w = dist.shape
        scale = 1000.0 / max(w, h)
        char_medians = []
        char_thickness = []
        stroke_metas = []

        for s_def in strokes_def:
            samples = 55 if len(s_def['raw_path']) > 150 else 35
            pts, thick = smooth_resample(s_def['raw_path'], dist, num_samples=samples)
            # 像素坐标映射至字体归一化坐标系 (font_x 向右, font_y 向上)
            fx = pts[:, 0] * scale
            fy = (h - pts[:, 1]) * scale
            median_coords = np.column_stack([fx, fy]).tolist()
            char_medians.append(median_coords)
            char_thickness.append(thick.tolist())
            stroke_metas.append({
                'name': s_def['name'],
                'type': s_def['type'],
            })

        normalized_library[char] = {
            'character': char,
            'stroke_count': len(stroke_metas),
            'stroke_metas': stroke_metas,
            'medians': char_medians,
            'thickness': char_thickness,
        }

    STROKES_JSON.write_text(json.dumps(normalized_library, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Authentic logo strokes saved to {STROKES_JSON}')
    return normalized_library


def get_logo_character_strokes(char):
    """从本地字体数据库检索指定字符的骨架中心线、笔画粗细分布与笔画元数据。

    Args:
        char: 单字字符 ('哈', '工', 或 '大')

    Returns:
        tuple[list[np.ndarray], list[np.ndarray], list[dict]]:
            - medians: 笔画中心线点集列表
            - thickness: 笔画粗细分布序列列表
            - metas: 笔画元信息字典列表 (名称、笔画类型)
    """
    if not STROKES_JSON.exists():
        build_logo_strokes()
    data = json.loads(STROKES_JSON.read_text(encoding='utf-8'))
    if char not in data:
        raise ValueError(f'Character {char!r} not in logo database; supported: 哈, 工, 大')
    char_entry = data[char]
    medians = [np.asarray(p, dtype=float) for p in char_entry['medians']]
    thickness = [np.asarray(t, dtype=float) for t in char_entry['thickness']]
    metas = char_entry.get('stroke_metas', [])
    return medians, thickness, metas


if __name__ == '__main__':
    build_logo_strokes()

