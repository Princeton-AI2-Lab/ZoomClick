import os
import json
import argparse
import shutil
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Any

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def write_json(path: str, data: Any):
    ensure_dir(os.path.dirname(path))
    with open(path, 'w') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def copy_image(src: str, dst_dir: str) -> str:
    ensure_dir(dst_dir)
    base = os.path.basename(src)
    dst = os.path.join(dst_dir, base)
    if not os.path.exists(dst):
        try:
            os.link(src, dst)
        except Exception:
            shutil.copy2(src, dst)
    return dst


def load_details(path: str) -> List[Dict[str, Any]]:
    with open(path, 'r') as f:
        data = json.load(f)
    if isinstance(data, dict) and 'details' in data:
        return data['details']
    elif isinstance(data, list):
        return data
    else:
        raise ValueError(f"Cannot read details list from {path}")


def index_by_task_id(details: List[Dict[str, Any]]) -> Dict[Tuple[str, Any], Dict[str, Any]]:
    idx: Dict[Tuple[str, Any], Dict[str, Any]] = {}
    for s in details:
        k = (s.get('task_filename'), s.get('id'))
        if k[0] is not None and k[1] is not None:
            idx[k] = s
    return idx


def index_by_task_basename(details: List[Dict[str, Any]]) -> Dict[Tuple[str, str], Dict[str, Any]]:
    idx: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for s in details:
        task = s.get('task_filename')
        p = s.get('img_filename') or s.get('img_path') or ''
        bn = os.path.basename(p)
        if task and bn:
            idx.setdefault((task, bn), s)
    return idx


def is_correct(entry: Dict[str, Any]) -> bool:
    return entry is not None and entry.get('correctness') == 'correct'


def decide_category(s1_ok: bool, s2_ok: bool, s3_ok: bool, s4_ok: bool) -> str:
    if s1_ok:
        return 'easy'
    if (not s1_ok) and (not s2_ok) and (not s3_ok) and (not s4_ok):
        return 'fail'
    return 'hard'


def draw_stats_png(counts: Dict[str, int], out_png: str):
    labels = list(counts.keys())
    values = [counts[k] for k in labels]
    plt.figure(figsize=(12, 6))
    colors = []
    for label in labels:
        if 'normal' in label:
            colors.append('#2ecc71')
        elif 'mislead' in label:
            colors.append('#e67e22')
        else:
            colors.append('#3498db')
    bars = plt.bar(labels, values, color=colors)
    plt.ylabel('Count')
    plt.title('ScreenSpot-Zoom Dataset Statistics')
    plt.xticks(rotation=20, ha='right')
    for bar, v in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2.0, bar.get_height(), str(v), ha='center', va='bottom')
    plt.tight_layout()
    ensure_dir(os.path.dirname(out_png))
    plt.savefig(out_png)
    plt.close()


def load_src_annotations(src_root: str) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[Tuple[str, Any], Dict[str, Any]]]:
    ann_dir = os.path.join(src_root, 'annotations')
    if not os.path.isdir(ann_dir):
        raise FileNotFoundError(f"Source annotations not found: {ann_dir}")
    task_to_items: Dict[str, List[Dict[str, Any]]] = {}
    id_index: Dict[Tuple[str, Any], Dict[str, Any]] = {}
    for name in os.listdir(ann_dir):
        if not name.endswith('.json'):
            continue
        task = os.path.splitext(name)[0]
        with open(os.path.join(ann_dir, name), 'r') as f:
            items = json.load(f)
        if not isinstance(items, list):
            continue
        task_to_items[task] = items
        for it in items:
            id_index[(task, it.get('id'))] = it
    return task_to_items, id_index


def main():
    parser = argparse.ArgumentParser(description='Build ScreenSpot-Zoom from four depth results (per-source-sample traversal).')
    parser.add_argument('--src_dataset_root', required=True, help='Original ScreenSpot-Pro root (must contain images/ and annotations/)')
    parser.add_argument('--depth1', required=True, help='venus_72b.json (depth=1)')
    parser.add_argument('--depth2', required=True, help='venus_72b_in_depth_2.json (depth=2)')
    parser.add_argument('--depth3', required=True, help='venus_72b_in.json (depth=3)')
    parser.add_argument('--depth4', required=True, help='venus_72b_in_depth_4.json (depth=4)')
    parser.add_argument('--out_dir', required=True, help='Output root for ScreenSpot-Zoom')
    parser.add_argument('--images_root', required=False, help='Optional extra images root for fallback matching')
    args = parser.parse_args()

    # Load depth results
    d1_list = load_details(args.depth1)
    d2_list = load_details(args.depth2)
    d3_list = load_details(args.depth3)
    d4_list = load_details(args.depth4)

    d1_by_id = index_by_task_id(d1_list)
    d2_by_id = index_by_task_id(d2_list)
    d3_by_id = index_by_task_id(d3_list)
    d4_by_id = index_by_task_id(d4_list)

    d1_by_bn = index_by_task_basename(d1_list)
    d2_by_bn = index_by_task_basename(d2_list)
    d3_by_bn = index_by_task_basename(d3_list)
    d4_by_bn = index_by_task_basename(d4_list)

    # Load source annotations
    task_to_items, id_index = load_src_annotations(args.src_dataset_root)

    # 5 splits: easy_normal, easy_mislead, hard_normal, hard_mislead, hard_est
    buckets: Dict[str, List[Tuple[str, Any]]] = {
        'easy_normal': [], 'easy_mislead': [],
        'hard_normal': [], 'hard_mislead': [],
        'hard_est': []
    }

    counts = Counter()

    def lookup(task: str, sid: Any, img_name: str, by_id, by_bn):
        s = by_id.get((task, sid))
        if s is not None:
            return s
        bn = os.path.basename(img_name) if img_name else None
        if bn:
            return by_bn.get((task, bn))
        return None

    # Traverse source dataset one-by-one
    for task, items in task_to_items.items():
        for it in items:
            sid = it.get('id')
            img_name = it.get('img_filename') or os.path.basename(it.get('img_path', ''))

            s1 = lookup(task, sid, img_name, d1_by_id, d1_by_bn)
            s2 = lookup(task, sid, img_name, d2_by_id, d2_by_bn)
            s3 = lookup(task, sid, img_name, d3_by_id, d3_by_bn)
            s4 = lookup(task, sid, img_name, d4_by_id, d4_by_bn)

            s1_ok, s2_ok, s3_ok, s4_ok = is_correct(s1), is_correct(s2), is_correct(s3), is_correct(s4)

            category = decide_category(s1_ok, s2_ok, s3_ok, s4_ok)

            if category == 'easy':
                split = 'easy_mislead' if (not s2_ok) else 'easy_normal'
            elif category == 'fail':
                split = 'hard_est'
            else:  # hard
                if (s2_ok and (not s3_ok)) or (s3_ok and (not s4_ok)):
                    split = 'hard_mislead'
                else:
                    split = 'hard_normal'

            buckets[split].append((task, sid))
            counts[split] += 1

    # Export per split in ScreenSpot-Pro style
    out_images_root = os.path.join(args.out_dir, 'images')
    out_ann_root = os.path.join(args.out_dir, 'annotations')
    ensure_dir(out_images_root)
    ensure_dir(out_ann_root)

    src_images_root = os.path.join(args.src_dataset_root, 'images')

    per_split_items: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    missing_total = 0

    def export_split(split_name: str, key_pairs: List[Tuple[str, Any]]):
        nonlocal missing_total
        images_dir = os.path.join(out_images_root, split_name)
        ensure_dir(images_dir)
        exported: List[Dict[str, Any]] = []
        missing = 0
        for task, sid in key_pairs:
            ann = id_index.get((task, sid))
            if ann is None:
                continue
            ann_out = dict(ann)
            # compute src image candidates
            img_name = ann.get('img_filename') or os.path.basename(ann.get('img_path', ''))
            base = os.path.basename(img_name) if img_name else None
            candidates = []
            if img_name:
                candidates.append(os.path.join(src_images_root, img_name))
                candidates.append(os.path.join(src_images_root, base))
                candidates.append(os.path.join(src_images_root, task, base))
            if args.images_root:
                candidates.append(os.path.join(args.images_root, img_name))
                candidates.append(os.path.join(args.images_root, base))
                candidates.append(os.path.join(args.images_root, task, base))

            src_img = None
            for c in candidates:
                if c and os.path.exists(c):
                    src_img = c
                    break
            if src_img is not None:
                copy_image(src_img, images_dir)
            else:
                print(f"[Warn] Missing image for {split_name}: task={task} id={sid} img={img_name}")
                missing += 1
                missing_total += 1

            # rewrite img_filename relative to new images root
            if base:
                ann_out['img_filename'] = f"{split_name}/" + base
            if 'img_path' in ann_out:
                del ann_out['img_path']
            exported.append(ann_out)

        write_json(os.path.join(out_ann_root, f"{split_name}.json"), exported)
        print(f"[Split] {split_name}: total={len(exported)} missing_images={missing}")
        per_split_items[split_name] = exported

    # Export new 5 splits
    for split_name in ['easy_normal','easy_mislead','hard_normal','hard_mislead','hard_est']:
        export_split(split_name, buckets[split_name])

    # Stats & manifest
    stats_png = os.path.join(args.out_dir, 'stats.png')
    draw_stats_png(counts, stats_png)
    manifest = {k: [(t, i) for (t, i) in v] for k, v in buckets.items()}
    write_json(os.path.join(args.out_dir, 'manifest.json'), manifest)
    print(f"[Done] Output: {args.out_dir}  Missing images total: {missing_total}")


if __name__ == '__main__':
    main()
