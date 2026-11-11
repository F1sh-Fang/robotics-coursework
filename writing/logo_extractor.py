"""Extract Ha-Gong-Da characters from HIT logo and generate authentic 3D brush stroke profiles."""
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
    """Segment '哈', '工', '大' from HIT logo image."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    img = Image.open(logo_path)
    alpha = np.array(img)[:, :, 3]

    # Bounding boxes inside logo.png for 哈, 工, 大
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

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
        clean = np.zeros_like(mask)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] > 80:
                clean[labels == i] = 255

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

        # Save single character image
        cv2.imwrite(str(out_dir / f'{key}.png'), 255 - char_crop)

        # Distance transform to compute stroke thickness
        binary_ink = (char_crop > 0).astype(np.uint8)
        dist = cv2.distanceTransform(binary_ink, cv2.DIST_L2, 5)
        dist_vis = (dist / max(1e-3, dist.max()) * 255).astype(np.uint8)
        cv2.imwrite(str(out_dir / f'{key}_thickness.png'), dist_vis)

        # Thinning skeleton
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

    # Combined 3-character preview image
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
    """Find the exact geodesic path on the skeleton between start_pt and end_pt."""
    pts = set(map(tuple, np.argwhere(skel > 0)))
    pts_arr = np.array(list(pts))
    d_start = np.hypot(pts_arr[:, 1] - start_pt[0], pts_arr[:, 0] - start_pt[1])
    d_end = np.hypot(pts_arr[:, 1] - end_pt[0], pts_arr[:, 0] - end_pt[1])
    s_y, s_x = pts_arr[np.argmin(d_start)]
    e_y, e_x = pts_arr[np.argmin(d_end)]

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

    path = []
    curr = (e_y, e_x)
    while curr is not None:
        path.append((curr[1], curr[0]))  # (x, y)
        curr = parent.get(curr)
    path.reverse()
    return path


def smooth_resample(path, dist, num_samples=35):
    """Resample skeleton path evenly by arc length and query distance transform for thickness."""
    path = np.array(path, dtype=float)
    # Deduplicate consecutive identical points
    diffs = np.diff(path, axis=0)
    keep = np.r_[True, np.hypot(diffs[:, 0], diffs[:, 1]) > 1e-6]
    path = path[keep]
    if len(path) < 2:
        return path, np.ones(len(path)) * 0.5

    diffs = np.diff(path, axis=0)
    segs = np.hypot(diffs[:, 0], diffs[:, 1])
    dists = np.r_[0, np.cumsum(segs)]
    total_len = dists[-1]

    sample_dists = np.linspace(0, total_len, num_samples)
    cs_x = CubicSpline(dists, path[:, 0])
    cs_y = CubicSpline(dists, path[:, 1])

    sampled_x = cs_x(sample_dists)
    sampled_y = cs_y(sample_dists)

    # Query local radius from distance transform
    h, w = dist.shape
    thickness = []
    max_d = max(26.0, float(dist.max()))
    for px, py in zip(sampled_x, sampled_y):
        ix = int(np.clip(round(px), 0, w - 1))
        iy = int(np.clip(round(py), 0, h - 1))
        r = float(dist[iy, ix])
        # Normalize relative thickness between 0.15 and 1.0
        thickness.append(np.clip(r / max_d, 0.15, 1.0))

    pts = np.column_stack([sampled_x, sampled_y])
    return pts, np.array(thickness)


def build_logo_strokes():
    """Extract authentic brush strokes directly from logo skeleton."""
    chars_data = extract_characters()

    # 1. Authentic Gong (工) - 3 strokes
    gong_skel = chars_data['gong']['skel']
    gong_dist = chars_data['gong']['dist']
    gong_strokes_def = [
        # 1. Top horizontal (上横)
        {'name': '工字上横', 'type': '横', 'raw_path': shortest_skeleton_path(gong_skel, (79, 53), (173, 36))},
        # 2. Slanted vertical stem (中竖)
        {'name': '工字中竖', 'type': '竖', 'raw_path': shortest_skeleton_path(gong_skel, (165, 40), (39, 170))},
        # 3. Bottom long rising horizontal (下长横)
        {'name': '工字下横', 'type': '横', 'raw_path': shortest_skeleton_path(gong_skel, (39, 170), (176, 117))},
    ]

    # 2. Authentic Da (大) - 3 strokes
    da_skel = chars_data['da']['skel']
    da_dist = chars_data['da']['dist']
    da_strokes_def = [
        # 1. Upward slanted long horizontal (长横)
        {'name': '大字长横', 'type': '横', 'raw_path': shortest_skeleton_path(da_skel, (43, 209), (209, 124))},
        # 2. Sweeping vertical pie (长撇)
        {'name': '大字长撇', 'type': '撇', 'raw_path': shortest_skeleton_path(da_skel, (136, 33), (24, 279))},
        # 3. Broad right-falling na (长捺)
        {'name': '大字长捺', 'type': '捺', 'raw_path': shortest_skeleton_path(da_skel, (120, 197), (249, 261))},
    ]

    # 3. Authentic Ha (哈) - 9 standard calligraphy strokes
    # Stroke sequence: Left '口' (竖 -> 横折 -> 横) + Right '合' (撇 -> 捺 -> 横 -> 下口竖 -> 下口横折 -> 下口横)
    ha_skel = chars_data['ha']['skel']
    ha_dist = chars_data['ha']['dist']
    ha_strokes_def = [
        # 1. Left 'kou': left vertical (左竖)
        {'name': '左口左竖', 'type': '竖', 'raw_path': shortest_skeleton_path(ha_skel, (36, 117), (43, 184))},
        # 2. Left 'kou': top horizontal & right turn (横折, 属于一个笔画，转折不抬笔)
        {'name': '左口横折', 'type': '横折', 'raw_path': shortest_skeleton_path(ha_skel, (80, 105), (118, 106)) + shortest_skeleton_path(ha_skel, (118, 106), (111, 163))},
        # 3. Left 'kou': bottom close (底横)
        {'name': '左口底横', 'type': '横', 'raw_path': shortest_skeleton_path(ha_skel, (43, 184), (111, 163))},
        # 4. Right 'he': top left sweeping pie (人字长撇，自顶端一气呵成扫至左下)
        {'name': '右合人字撇', 'type': '撇', 'raw_path': shortest_skeleton_path(ha_skel, (207, 22), (28, 268))},
        # 5. Right 'he': top right falling na (人字长捺)
        {'name': '右合人字捺', 'type': '捺', 'raw_path': shortest_skeleton_path(ha_skel, (207, 22), (282, 174))},
        # 6. Right 'he': middle horizontal stroke (中部独立一横，位于撇与捺之间)
        {'name': '右合中横', 'type': '横', 'raw_path': shortest_skeleton_path(ha_skel, (150, 180), (195, 181))},
        # 7. Lower 'kou': left vertical (下口左竖)
        {'name': '下口左竖', 'type': '竖', 'raw_path': shortest_skeleton_path(ha_skel, (164, 237), (139, 283))},
        # 8. Lower 'kou': top horizontal & right turn (下口横折, 属于一个笔画，转折不抬笔)
        {'name': '下口横折', 'type': '横折', 'raw_path': shortest_skeleton_path(ha_skel, (164, 237), (206, 231)) + shortest_skeleton_path(ha_skel, (206, 231), (207, 276))},
        # 9. Lower 'kou': bottom close (下口底横)
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
            # Map image (px, py) to font coordinates: font_x = px, font_y = h - py
            fx = pts[:, 0] * scale
            fy = (h - pts[:, 1]) * scale  # Font Y points upward
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
    """Retrieve authentic medians, thickness profiles and stroke metadata from logo database."""
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
