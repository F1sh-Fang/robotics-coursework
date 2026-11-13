"""校徽单字图像分割、真实书法骨架提取与笔画粗细信息生成模块。

主要功能：
  1. 从哈工大官方校徽 logo.png 中分割提取“哈”、“工”、“大”三个毛体书法单字；
  2. 采用欧氏距离变换 (Distance Transform) 计算字迹各处的物理笔宽；
  3. 采用 Zhang-Suen 形态学细化算法提取单像素宽的真实书法骨架；
  4. 按规范笔顺，由沿墨迹中心选取的控制点生成各笔中心线（STROKE_DEFS）；
  5. 进行等距弧长插值与粗细归一化，保存至 logo/hit_strokes.json 供轨迹规划使用。
"""
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
STROKES_VERSION = 4  # 改动笔画定义后递增，旧缓存会自动重建


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


def smooth_resample(path, dist, num_samples=35):
    """沿中心线按弧长等距重采样，并从欧氏距离变换图查询局部笔画粗细。

    Args:
        path: 中心线像素点集 [(x, y), ...]
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


def guided_pieces(*pieces):
    """控制点 -> 约 1 像素间距的稠密中心线：段内三次样条平滑，段间相接处保留折角。"""
    dense = []
    for ctrl in pieces:
        ctrl = np.asarray(ctrl, dtype=float)
        chord = np.r_[0., np.cumsum(np.hypot(*np.diff(ctrl, axis=0).T))]
        count = max(2, int(np.ceil(chord[-1])) + 1)
        dense.append(CubicSpline(chord, ctrl)(np.linspace(0., chord[-1], count)))
    return dense


def resample_stroke(pieces, dist):
    """各段按长度分配采样点后拼接，折角处恰为采样点，几何层据此顿笔。"""
    dense = guided_pieces(*pieces)
    lengths = [float(np.hypot(*np.diff(p, axis=0).T).sum()) for p in dense]
    total = 55 if sum(lengths) > 150 else 35
    all_pts, all_thick = [], []
    for k, (piece, length) in enumerate(zip(dense, lengths)):
        count = max(8, int(round(total * length / sum(lengths))))
        pts, thick = smooth_resample(piece, dist, num_samples=count)
        if k:  # 与上一段共用折点
            pts, thick = pts[1:], thick[1:]
        all_pts.append(pts)
        all_thick.append(thick)
    return np.vstack(all_pts), np.concatenate(all_thick)


# 笔画定义：(名称, 类型, 分段控制点)。控制点为单字图像像素坐标，沿墨迹中心选取；
# 段内平滑，多段相接处为折角（顿笔）。不走骨架，避免交叉处折弯、拐角毛刺和重描。
STROKE_DEFS = {
    'ha': [
        ('左口左竖', '竖', [[(33, 101), (40, 120), (47, 145), (47, 170), (42, 201)]]),
        ('左口横折', '横折', [[(67, 104), (90, 105), (110, 106), (124, 107)],
                          [(124, 107), (113, 120), (104, 140), (99, 160)]]),
        ('左口底横', '横', [[(47, 169), (70, 169), (90, 167), (100, 160)]]),
        ('右合人字撇', '撇', [[(209, 22), (205, 61), (190, 100), (167, 143),
                            (141, 184), (112, 222), (76, 253), (28, 268)]]),
        ('右合人字捺', '捺', [[(186, 111), (204, 120), (219, 140), (233, 162),
                            (243, 176), (254, 180), (268, 178), (282, 175)]]),
        ('右合中横', '横', [[(145, 180), (168, 186), (185, 186), (207, 182)]]),
        ('下口左竖', '竖', [[(143, 246), (137, 262), (138, 280), (140, 293)]]),
        ('下口横折', '横折', [[(143, 246), (155, 241), (172, 237), (190, 236), (204, 234)],
                          [(204, 234), (196, 245), (193, 260), (192, 277)]]),
        ('下口底横', '横', [[(139, 281), (165, 282), (190, 279), (214, 277)]]),
    ],
    'gong': [  # 连笔：中竖从上横收笔处起笔，下横从中竖末端沿原路回笔
        ('工字上横', '横', [[(80, 57), (125, 47), (170, 37)]]),
        ('工字中竖', '竖', [[(170, 37), (155, 67), (137, 89), (120, 112), (108, 136),
                          (92, 150), (75, 164), (58, 171), (39, 172)]]),
        ('工字下横', '横', [[(39, 172), (58, 171), (75, 164), (92, 150), (108, 136),
                          (122, 130), (139, 127), (156, 123), (175, 116)]]),
    ],
    'da': [
        ('大字长横', '横', [[(44, 212), (64, 202), (84, 191), (104, 181), (125, 171),
                          (146, 163), (167, 156), (188, 148), (213, 138)]]),
        ('大字长撇', '撇', [[(138, 33), (140, 60), (137, 95), (132, 125), (127, 160), (121, 195),
                          (114, 223), (104, 244), (90, 261), (72, 275), (50, 283), (24, 281)]]),
        ('大字长捺', '捺', [[(121, 196), (137, 204), (152, 215), (162, 232), (173, 247),
                          (186, 262), (200, 275), (215, 278), (230, 273), (249, 262)]]),
    ],
}


def build_logo_strokes():
    """提取哈工大校徽真实书法笔画数据库，并序列化保存为 JSON。

    笔顺定义：
      - “工”字 3 笔：上横 -> 顺势斜切中竖 -> 劲挺长挑底横（连笔运笔）；
      - “大”字 3 笔：长横 -> 舒展长撇 -> 沉稳右捺；
      - “哈”字 9 笔：左口 3 笔（左竖、横折、底横） + 右合 6 笔（人字撇、捺、中横、下口竖、横折、底横）。
    """
    chars_data = extract_characters()

    normalized_library = {}
    for key, strokes_def in STROKE_DEFS.items():
        char, dist = chars_data[key]['character'], chars_data[key]['dist']
        h, w = dist.shape
        scale = 1000.0 / max(w, h)
        char_medians = []
        char_thickness = []
        stroke_metas = []

        for name, kind, pieces in strokes_def:
            pts, thick = resample_stroke(pieces, dist)
            # 像素坐标映射至字体归一化坐标系 (font_x 向右, font_y 向上)
            fx = pts[:, 0] * scale
            fy = (h - pts[:, 1]) * scale
            median_coords = np.column_stack([fx, fy]).tolist()
            char_medians.append(median_coords)
            char_thickness.append(thick.tolist())
            stroke_metas.append({'name': name, 'type': kind})

        normalized_library[char] = {
            'character': char,
            'stroke_count': len(stroke_metas),
            'stroke_metas': stroke_metas,
            'medians': char_medians,
            'thickness': char_thickness,
        }

    normalized_library['_version'] = STROKES_VERSION
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
    data = json.loads(STROKES_JSON.read_text(encoding='utf-8')) if STROKES_JSON.exists() else {}
    if data.get('_version') != STROKES_VERSION:  # 无缓存或缓存过期则重建
        data = build_logo_strokes()
    if char not in data:
        raise ValueError(f'Character {char!r} not in logo database; supported: 哈, 工, 大')
    char_entry = data[char]
    medians = [np.asarray(p, dtype=float) for p in char_entry['medians']]
    thickness = [np.asarray(t, dtype=float) for t in char_entry['thickness']]
    metas = char_entry.get('stroke_metas', [])
    return medians, thickness, metas


if __name__ == '__main__':
    build_logo_strokes()

