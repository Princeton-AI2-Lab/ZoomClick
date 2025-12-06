#!/usr/bin/env python3
import argparse
import json
import os
from collections import defaultdict


CATEGORIES = [
    "easy_normal",
    "easy_mislead",
    "hard_normal",
    "hard_mislead",
    "hard_est",
]


def load_annotations(dataset_root: str):
    annotations_dir = os.path.join(dataset_root, "annotations")
    basename_to_info = {}

    for category in CATEGORIES:
        ann_path = os.path.join(annotations_dir, f"{category}.json")
        if not os.path.isfile(ann_path):
            raise FileNotFoundError(f"Annotation file not found: {ann_path}")
        with open(ann_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for item in data:
            img_filename = item.get("img_filename")  # e.g., "easy_normal/screenshot_...png"
            if not img_filename:
                continue
            basename = os.path.basename(img_filename)
            basename_to_info[basename] = {
                "category": category,
                "id": item.get("id"),
                "instruction": item.get("instruction"),
                "bbox": item.get("bbox"),
            }

    return basename_to_info


def load_results(results_path: str):
    with open(results_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    details = data.get("details", [])
    return details


def compute_accuracy(details, basename_to_info):
    counters = {
        c: {
            "total": 0,
            "correct": 0,
        }
        for c in CATEGORIES
    }

    missing = 0
    for d in details:
        img_path = d.get("img_path")
        correctness = d.get("correctness")
        if not img_path:
            continue
        basename = os.path.basename(img_path)
        info = basename_to_info.get(basename)
        if not info:
            # 可能来自 ScreenSpot-Pro 中但未被纳入 Zoom，或文件名不匹配
            missing += 1
            continue

        category = info["category"]
        if category not in counters:
            # 非预期类别，跳过但计入缺失
            missing += 1
            continue

        counters[category]["total"] += 1
        if correctness == "correct":
            counters[category]["correct"] += 1

    # 计算准确率
    summary = {}
    grand_total = 0
    grand_correct = 0
    for c in CATEGORIES:
        total = counters[c]["total"]
        correct = counters[c]["correct"]
        acc = (correct / total) if total > 0 else 0.0
        summary[c] = {
            "total": total,
            "correct": correct,
            "accuracy": acc,
        }
        grand_total += total
        grand_correct += correct

    overall_acc = (grand_correct / grand_total) if grand_total > 0 else 0.0
    summary["overall"] = {
        "total": grand_total,
        "correct": grand_correct,
        "accuracy": overall_acc,
        "missing_unmatched": missing,
    }

    return summary


def main():
    parser = argparse.ArgumentParser(description="Collect per-category accuracy for ScreenSpot-Zoom.")
    parser.add_argument(
        "--results",
        required=True,
        help="Path to sspro result JSON (with 'details' list).",
    )
    parser.add_argument(
        "--dataset",
        default="/scratch/gpfs/sl8264/zhiyuan/datasets/ScreenSpot-Zoom",
        help="Path to ScreenSpot-Zoom dataset root.",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Optional path to save the summary JSON.",
    )
    args = parser.parse_args()

    basename_to_info = load_annotations(args.dataset)
    details = load_results(args.results)
    summary = compute_accuracy(details, basename_to_info)

    # 打印简表
    print("Per-category accuracy:")
    for c in CATEGORIES:
        s = summary[c]
        print(f"- {c}: accuracy={s['accuracy']:.4f} (correct={s['correct']}, total={s['total']})")
    overall = summary["overall"]
    print(
        f"- overall: accuracy={overall['accuracy']:.4f} (correct={overall['correct']}, total={overall['total']}); missing_unmatched={overall['missing_unmatched']}"
    )

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"Saved summary to: {args.output}")


if __name__ == "__main__":
    main()


