import argparse
import copy
import json
import logging
import math
import os
import sys
import tempfile

import torch
from PIL import Image
from tqdm import tqdm
from transformers.models.qwen2_vl.image_processing_qwen2_vl_fast import smart_resize

# ==================== Config & Constants ====================
logging.basicConfig(level=logging.INFO)
torch.manual_seed(114514)

GT_TYPES = ["positive", "negative"]
INSTRUCTION_STYLES = ["instruction", "action", "description"]
LANGUAGES = ["en", "cn"]

# ==================== Path Helpers ====================
# Resolve project roots so we can dynamically import code from sibling projects.
_current_dir = os.path.dirname(os.path.abspath(__file__))  # ZoomClick/grounding
_open_source_dir = os.path.dirname(_current_dir)  # ZoomClick
_project_root = os.path.dirname(_open_source_dir)  # zoomclick

_ui_venus_grounding_path = os.path.join(_project_root, "UI-Venus", "grounding")
_ui_tars_grounding_path = os.path.join(_project_root, "UI-TARS", "grounding")
_local_models_path = os.path.join(_current_dir, "models")

# ==================== Geometry & Image Utilities ====================
def _clamp(v, lo=0.0, hi=1.0):
    return max(lo, min(hi, v))

def _norm_to_pixel_point(p, img_size):
    x, y = p
    w, h = img_size
    return (int(round(x * w)), int(round(y * h)))

def _px_box_to_norm_box(px_box, img_size):
    x1, y1, x2, y2 = px_box
    w, h = img_size
    return (x1 / w, y1 / h, x2 / w, y2 / h)

def view_point_to_real_point(view_pt, viewport):
    vx1, vy1, vx2, vy2 = viewport
    vw, vh = max(1e-8, vx2 - vx1), max(1e-8, vy2 - vy1)
    px, py = view_pt
    rx = vx1 + vw * px
    ry = vy1 + vh * py
    return (_clamp(rx), _clamp(ry))

def view_box_to_real_box(view_box, viewport):
    vx1, vy1, vx2, vy2 = viewport
    vw, vh = max(1e-8, vx2 - vx1), max(1e-8, vy2 - vy1)
    x1, y1, x2, y2 = view_box
    rx1 = vx1 + vw * x1
    ry1 = vy1 + vh * y1
    rx2 = vx1 + vw * x2
    ry2 = vy1 + vh * y2
    rx1 = _clamp(rx1); ry1 = _clamp(ry1); rx2 = _clamp(rx2); ry2 = _clamp(ry2)
    if rx2 <= rx1: rx2 = min(1.0, rx1 + 1e-6)
    if ry2 <= ry1: ry2 = min(1.0, ry1 + 1e-6)
    return (rx1, ry1, rx2, ry2)

def crop_center_px(image, center_px, target_w_px, target_h_px, center_mode="shift"):
    W, H = image.size
    cx, cy = center_px
    tw = min(max(1, int(round(target_w_px))), W)
    th = min(max(1, int(round(target_h_px))), H)

    x1 = int(round(cx - tw / 2)); y1 = int(round(cy - th / 2))
    x2 = x1 + tw; y2 = y1 + th

    if center_mode == "clip":
        ix1 = max(0, x1); iy1 = max(0, y1)
        ix2 = min(W, x2); iy2 = min(H, y2)
        if ix2 <= ix1:
            ix1 = max(0, min(W - 1, int(round(cx))))
            ix2 = min(W, ix1 + 1)
        if iy2 <= iy1:
            iy1 = max(0, min(H - 1, int(round(cy))))
            iy2 = min(H, iy1 + 1)
        cropped = image.crop((ix1, iy1, ix2, iy2))
        return cropped, (ix1, iy1, ix2, iy2)

    if center_mode == "shrink":
        max_w = int(2 * min(cx, W - cx))
        max_h = int(2 * min(cy, H - cy))
        tw = max(1, min(tw, max_w))
        th = max(1, min(th, max_h))
        x1 = int(round(cx - tw / 2)); y1 = int(round(cy - th / 2))
        x2 = x1 + tw; y2 = y1 + th
        cropped = image.crop((x1, y1, x2, y2))
        return cropped, (x1, y1, x2, y2)

    # Default: shift behavior (keep window size, shift to stay inside bounds)
    if x1 < 0:
        x2 -= x1; x1 = 0
    if y1 < 0:
        y2 -= y1; y1 = 0
    if x2 > W:
        shift = x2 - W
        x1 -= shift; x2 = W
        if x1 < 0: x1 = 0
    if y2 > H:
        shift = y2 - H
        y1 -= shift; y2 = H
        if y1 < 0: y1 = 0
    if x2 <= x1: x2 = min(W, x1 + 1)
    if y2 <= y1: y2 = min(H, y1 + 1)
    cropped = image.crop((x1, y1, x2, y2))
    return cropped, (x1, y1, x2, y2)

def generate_patches(patch_size):
    if patch_size == 2:
        patches = [
            (0.0, 0.0, 0.5, 0.5), (0.5, 0.0, 1.0, 0.5),
            (0.0, 0.5, 0.5, 1.0), (0.5, 0.5, 1.0, 1.0),
        ]
    elif patch_size == 3:
        patches = [
            (0.0, 0.0, 1/3, 1/3), (1/3, 0.0, 2/3, 1/3), (2/3, 0.0, 1.0, 1/3),
            (0.0, 1/3, 1/3, 2/3), (1/3, 1/3, 2/3, 2/3), (2/3, 1/3, 1.0, 2/3),
            (0.0, 2/3, 1/3, 1.0), (1/3, 2/3, 2/3, 1.0), (2/3, 2/3, 1.0, 1.0),
        ]
    elif patch_size == 4:
        # Generate a 4x4 grid
        patches = []
        for r in range(4):
            for c in range(4):
                patches.append((c/4, r/4, (c+1)/4, (r+1)/4))
    else:
        raise ValueError(f"Unsupported patch_size: {patch_size}")
    return patches

def compute_pixel_distance(p1, p2, img_size):
    x1, y1 = p1
    x2, y2 = p2
    w, h = img_size
    px1 = x1 * w; py1 = y1 * h
    px2 = x2 * w; py2 = y2 * h
    return math.sqrt((px2 - px1) ** 2 + (py2 - py1) ** 2)

# ==================== Backend Adapter ====================

class BaseBackend:
    """Unified backend interface: given instruction and PIL Image, return (x_norm, y_norm) or None."""
    def infer_norm_point(self, instruction, pil_img):
        raise NotImplementedError

class Qwen3Backend(BaseBackend):
    def __init__(self, model):
        self.model = model
    
    def infer_norm_point(self, instruction, pil_img):
        res = self.model.ground_only_positive(instruction=instruction, image=pil_img)
        pt = res.get("point") if isinstance(res, dict) else None
        if not pt or len(pt) != 2:
            return None, res

        img_w, img_h = pil_img.size
        iproc = getattr(self.model, "processor", None)
        try:
            patch_size = iproc.image_processor.patch_size
            merge_size = iproc.image_processor.merge_size
            resized_h, resized_w = smart_resize(
                img_h,
                img_w,
                factor=patch_size * merge_size,
                min_pixels=patch_size * patch_size * merge_size * merge_size * 16,
                max_pixels=patch_size * patch_size * merge_size * merge_size * 6400,
            )
        except Exception:
            resized_w, resized_h = max(1, img_w), max(1, img_h)

        norm_x = float(pt[0]) / float(resized_w)
        norm_y = float(pt[1]) / float(resized_h)
        return (norm_x, norm_y), res

class UIVenusBackend(BaseBackend):
    def __init__(self, model):
        self.model = model
    
    def infer_norm_point(self, instruction, pil_img):
        # UI-Venus: inference expects an image path and returns normalized coordinates (x_norm, y_norm)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            pil_img.save(tmp.name)
            tmp_path = tmp.name
        
        try:
            res = self.model.inference(instruction=instruction, image_path=tmp_path)
        except Exception as e:
            return None, f"Error: {str(e)}"
        finally:
            try:
                os.remove(tmp_path)
            except:
                pass
        
        if res is None:
            return None, "Model returned None"
        
        pt = res.get("point") if isinstance(res, dict) else None
        if not pt or len(pt) != 2:
            return None, res
        
        # UI-Venus already returns normalized coordinates
        return (float(pt[0]), float(pt[1])), res

def build_backend(backend_type, model_type, model_name_or_path=None):
    """Build backend by adjusting sys.path and loading the corresponding model."""
    print(f"Building backend: {backend_type}, model_type: {model_type}, path: {model_name_or_path}")
    
    # Prefer local Open-Source/grounding/models
    if os.path.exists(_local_models_path) and _local_models_path not in sys.path:
        sys.path.insert(0, _local_models_path)
    # Remove potential conflicting paths from sibling projects to avoid shadowing local models
    if _ui_venus_grounding_path in sys.path:
        sys.path.remove(_ui_venus_grounding_path)
    if _ui_tars_grounding_path in sys.path:
        sys.path.remove(_ui_tars_grounding_path)

    if backend_type == 'qwen3':
        from models.qwen3vl import Qwen3VLModel
        model = Qwen3VLModel()
        if model_name_or_path:
            model.load_model(model_path=model_name_or_path)
        else:
            model.load_model()
        return Qwen3Backend(model)

    elif backend_type == 'uivenus':
        if model_type == "ui_venus_ground_7b":
            from models.ui_venus_ground_7b import UI_Venus_Ground_7B
            model = UI_Venus_Ground_7B()
        elif model_type == "ui_venus_ground_72b":
            from models.ui_venus_ground_72b import UI_Venus_Ground_72B
            model = UI_Venus_Ground_72B()
        else:
            raise ValueError(f"Unsupported UI-Venus model type: {model_type}")
        
        if model_name_or_path:
            model.load_model(model_name_or_path=model_name_or_path)
        else:
            model.load_model()
        
        if hasattr(model, "set_generation_config"):
            model.set_generation_config(temperature=0, max_new_tokens=256)
        
        return UIVenusBackend(model)
    
    else:
        raise ValueError(f"Unknown backend type: {backend_type}")

# ==================== ZoomClick / IN Logic ====================

def prezoom_first_iteration_distance(backend: BaseBackend,
                                     instruction,
                                     image: Image.Image,
                                     patch_size=2,
                                     prezoom_px_thresh=50.0):
    """
    Unified pre-zoom logic:
    1) run backend on full image to get a direct normalized point
    2) generate patches and run backend per patch (map back to original normalized coords)
    3) compute distances and choose patch vs direct using a pixel-distance threshold
    """
    original_size = image.size
    patches = generate_patches(patch_size)

    candidates = []
    point_direct_norm = None

    # 1) Full-image direct
    pt, _ = backend.infer_norm_point(instruction, image)
    if pt is not None:
        point_direct_norm = pt
        candidates.append([pt[0], pt[1], pt[0], pt[1]])
        print(f"[DEBUG] Direct point (norm): {point_direct_norm}")
    else:
        candidates.append([0.0, 0.0, 0.0, 0.0])
        print("[DEBUG] Direct point prediction failed")

    # 2) Patch points
    patch_points_norm = []
    for idx, (x1, y1, x2, y2) in enumerate(patches):
        patch_img = image.crop((
            int(x1 * original_size[0]),
            int(y1 * original_size[1]),
            int(x2 * original_size[0]),
            int(y2 * original_size[1])
        ))
        
        pt, _ = backend.infer_norm_point(instruction, patch_img)
        
        if pt is not None:
            # Patch-level normalized coords -> map back to original normalized coords
            px, py = pt
            orig_x_norm = x1 + px * (x2 - x1)
            orig_y_norm = y1 + py * (y2 - y1)
            patch_points_norm.append((orig_x_norm, orig_y_norm))
            candidates.append([orig_x_norm, orig_y_norm, orig_x_norm, orig_y_norm])
            print(f"[DEBUG] Patch {idx+1} point (norm in full image): ({orig_x_norm:.4f}, {orig_y_norm:.4f})")
        else:
            # Fallback: patch center
            center_x_norm = (x1 + x2) / 2
            center_y_norm = (y1 + y2) / 2
            patch_points_norm.append((center_x_norm, center_y_norm))
            candidates.append([center_x_norm, center_y_norm, center_x_norm, center_y_norm])
            print(f"[DEBUG] Patch {idx+1} prediction failed, use patch center: ({center_x_norm:.4f}, {center_y_norm:.4f})")

    # 3) Distance & Threshold Decision
    selected_candidate = None
    final_point = None
    point_type = None
    min_distance = None
    use_prezoom = True

    if point_direct_norm is not None and patch_points_norm:
        dists = []
        for i, pp in enumerate(patch_points_norm):
            d = compute_pixel_distance(point_direct_norm, pp, original_size)
            dists.append((d, i))
            print(f"[DEBUG] Distance from direct to patch {i+1}: {d:.1f}px")
        dists.sort(key=lambda x: x[0])
        min_distance, best_idx = dists[0]

        if min_distance < prezoom_px_thresh:
            selected_candidate = best_idx + 1
            final_point = patch_points_norm[best_idx]
            point_type = f"patch_{best_idx+1}"
            use_prezoom = True
            print(f"[DEBUG] Use prezoom: min_dist={min_distance:.1f}px < {prezoom_px_thresh}")
        else:
            selected_candidate = 0
            final_point = point_direct_norm
            point_type = "direct"
            use_prezoom = False
            print(f"[DEBUG] Drop prezoom: all dists >= {prezoom_px_thresh}px, use direct.")
    else:
        if point_direct_norm is not None:
            selected_candidate = 0
            final_point = point_direct_norm
            point_type = "direct"
            use_prezoom = False
            print("[DEBUG] Only direct point available, no patch points. Use direct (no prezoom).")
        elif patch_points_norm:
            selected_candidate = 1
            final_point = patch_points_norm[0]
            point_type = "patch_1"
            use_prezoom = True
            print("[DEBUG] No direct point, use first patch point as fallback (prezoom).")
        else:
            selected_candidate = 0
            final_point = (0.5, 0.5)
            point_type = "fallback_center"
            use_prezoom = False
            print("[DEBUG] No valid points at all, fallback to image center (no prezoom).")

    num_candidates = len(candidates)
    num_model_calls = 1 + len(patches)

    print(f"[DEBUG] prezoom_first_iteration_distance: num_candidates={num_candidates}, num_model_calls={num_model_calls}")
    if min_distance is not None:
        print(f"[DEBUG] Selected candidate={selected_candidate} ({point_type}), "
              f"point={final_point}, min_dist={min_distance:.1f}px, use_prezoom={use_prezoom}")
    else:
        print(f"[DEBUG] Selected candidate={selected_candidate} ({point_type}), "
              f"point={final_point}, use_prezoom={use_prezoom}")

    return {
        "point": final_point,
        "selected_candidate": selected_candidate,
        "point_type": point_type,
        "candidates": candidates,
        "num_candidates": num_candidates,
        "min_distance": min_distance,
        "use_prezoom": use_prezoom,
        "num_model_calls": num_model_calls
    }


class PrezoomIterativeNarrowingRunner:
    def __init__(self, backend: BaseBackend,
                 max_search_depth=3,
                 shrink_ratio=0.5,
                 min_crop_size=768,
                 debug=True,
                 patch_size=2,
                 center_mode='shift',
                 prezoom_px_thresh=50.0):
        self.backend = backend
        self.max_search_depth = max_search_depth
        self.shrink_ratio = shrink_ratio
        self.min_crop_size = min_crop_size
        self.debug = debug
        self.patch_size = patch_size
        self.center_mode = center_mode
        self.prezoom_px_thresh = float(prezoom_px_thresh)
        self.logs = []

    def _log(self, s):
        self.logs.append(s)
        if self.debug:
            print(s)

    def ground_only_positive(self, instruction: str, image):
        self.logs.clear()
        if isinstance(image, str):
            assert os.path.exists(image) and os.path.isfile(image)
            image = Image.open(image).convert('RGB')
        assert isinstance(image, Image.Image)

        orig_W, orig_H = image.size
        viewport = (0.0, 0.0, 1.0, 1.0)
        curr_img = image
        last_pt_real = None

        for it in range(self.max_search_depth):
            self._log(f"[PIN] Iter {it+1}/{self.max_search_depth} viewport={viewport}, curr_size={curr_img.size}")

            if it == 0:
                self._log("[PIN] First iteration: prezoom")
                prezoom_result = prezoom_first_iteration_distance(
                    self.backend,
                    instruction,
                    curr_img,
                    patch_size=self.patch_size,
                    prezoom_px_thresh=self.prezoom_px_thresh,
                )
                
                # Check point
                if prezoom_result and prezoom_result.get("point") is not None:
                    pt_view = prezoom_result["point"]
                    self._log(
                        f"[PIN] Prezoom choice: {prezoom_result['point_type']}, "
                        f"use_prezoom={prezoom_result['use_prezoom']}, "
                        f"min_dist={prezoom_result.get('min_distance')}, "
                        f"num_candidates={prezoom_result.get('num_candidates')}, "
                        f"num_model_calls={prezoom_result.get('num_model_calls')}"
                    )
                else:
                    self._log("[PIN] Prezoom failed, fallback to direct.")
                    pt_view, _ = self.backend.infer_norm_point(instruction, curr_img)
                    if pt_view is None:
                        self._log("[PIN] Direct inference failed.")
                        break
            else:
                self._log(f"[PIN] Iteration {it+1}: direct inference")
                pt_view, _ = self.backend.infer_norm_point(instruction, curr_img)
                if pt_view is None:
                    self._log("[PIN] model returned None (no click).")
                    break

            # Map view point -> real point (normalized)
            pt_real = view_point_to_real_point(pt_view, viewport)
            last_pt_real = pt_real
            self._log(f"[PIN] pt_view={pt_view} -> pt_real={pt_real}")

            if it == self.max_search_depth - 1:
                self._log("[PIN] reached max depth, stop.")
                break

            # Crop for next iteration
            curr_W, curr_H = curr_img.size
            target_w = max(int(round(curr_W * self.shrink_ratio)), self.min_crop_size)
            target_h = max(int(round(curr_H * self.shrink_ratio)), self.min_crop_size)
            target_w = min(target_w, curr_W)
            target_h = min(target_h, curr_H)
            
            if target_w == curr_W and target_h == curr_H:
                self._log("[PIN] window cannot shrink further; stop early.")
                break

            center_px = _norm_to_pixel_point(pt_view, (curr_W, curr_H))
            
            curr_img, crop_px = crop_center_px(curr_img, center_px, target_w, target_h, center_mode=self.center_mode)
            norm_box = _px_box_to_norm_box(crop_px, (curr_W, curr_H))
            viewport = view_box_to_real_box(norm_box, viewport)
            
            self._log(f"[PIN] next_viewport={viewport}, next_size={curr_img.size}")

        if last_pt_real is None:
            return {"result": "negative", "point": None, "raw_response": "\n".join(self.logs)}

        click_px = _norm_to_pixel_point(last_pt_real, (orig_W, orig_H))
        return {"result": "positive", "point": [click_px[0], click_px[1]], "raw_response": "\n".join(self.logs)}

# ==================== Metrics & Evaluation ====================

def collect_results_to_eval(results, platform=None, group=None, application=None, language=None, gt_type=None, instruction_style=None, ui_type=None):
    filtered_results = []
    for sample in results:
        if (platform is None or sample.get("platform") == platform) and \
           (group is None or sample.get("group") == group) and \
           (application is None or sample.get("application") == application) and \
           (language is None or sample.get("language") == language) and \
           (gt_type is None or sample.get("gt_type") == gt_type) and \
           (instruction_style is None or sample.get("instruction_style") == instruction_style) and \
           (ui_type is None or sample.get("ui_type") == ui_type):
            filtered_results.append(sample)
    return filtered_results

def make_combinations(results, platform=False, group=None, application=False, language=False, gt_type=False, instruction_style=False, ui_type=False):
    import itertools
    unique_values = {"platform": set(), "group": set(), "application": set(), "language": set(), "gt_type": set(), "instruction_style": set(), "ui_type": set()}
    for sample in results:
        if platform: unique_values["platform"].add(sample.get("platform"))
        if group: unique_values["group"].add(sample.get("group"))
        if application: unique_values["application"].add(sample.get("application"))
        if language: unique_values["language"].add(sample.get("language"))
        if gt_type: unique_values["gt_type"].add(sample.get("gt_type"))
        if instruction_style: unique_values["instruction_style"].add(sample.get("instruction_style"))
        if ui_type: unique_values["ui_type"].add(sample.get("ui_type"))
    filtered_values = {k: list(v) for k, v in unique_values.items() if v}
    if not filtered_values: return []
    attribute_combinations = list(itertools.product(*filtered_values.values()))
    return [dict(zip(filtered_values.keys(), combo)) for combo in attribute_combinations]

def calc_metric_for_result_list(results):
    num_total = len(results)
    correct_num = sum(1 for r in results if r["correctness"] == "correct")
    wrong_format_num = sum(1 for r in results if r["correctness"] == "wrong_format")
    text_results = collect_results_to_eval(results, ui_type="text")
    icon_results = collect_results_to_eval(results, ui_type="icon")
    text_correct = sum(1 for r in text_results if r["correctness"] == "correct")
    text_total = len(text_results)
    icon_correct = sum(1 for r in icon_results if r["correctness"] == "correct")
    icon_total = len(icon_results)
    return {
        "num_correct_action": correct_num,
        "num_total": num_total,
        "wrong_format_num": wrong_format_num,
        "action_acc": correct_num / num_total if num_total > 0 else 0,
        "text_acc": text_correct / text_total if text_total > 0 else 0,
        "icon_acc": icon_correct / icon_total if icon_total > 0 else 0
    }

def eval_sample_positive_gt(sample, response):
    bbox = sample["bbox"]
    pt = response["point"]
    if pt is None: return "wrong_format"
    x, y = pt
    if bbox[0] <= x <= bbox[2] and bbox[1] <= y <= bbox[3]:
        print(f"[Correct] {pt} in {bbox}")
        return "correct"
    print(f"[Wrong] {pt} not in {bbox}")
    return "wrong"

def eval_sample_negative_gt(sample, response):
    if response["result"] == "negative": return "correct"
    elif response["result"] == "positive": return "wrong"
    else: return "wrong_format"

def evaluate_fine_grained(results):
    evaluation_result = {}
    for combo in make_combinations(results, platform=True, application=True, instruction_style=True, gt_type=True):
        filtered = collect_results_to_eval(results, platform=combo.get("platform"), application=combo.get("application"),
                                           instruction_style=combo.get("instruction_style"), gt_type=combo.get("gt_type"))
        metrics = calc_metric_for_result_list(filtered)
        if metrics['num_total'] == 0: continue
        key = f"plat:{combo.get('platform')} app:{combo.get('application')} inst_style:{combo.get('instruction_style')} gt_type:{combo.get('gt_type')}"
        evaluation_result[key] = metrics
    return evaluation_result

def evaluate_overall(results):
    return calc_metric_for_result_list(results)

def evaluate(results):
    return {
        "details": results,
        "metrics": {
            "fine_grained": evaluate_fine_grained(results),
            "overall": evaluate_overall(results),
        }
    }

# ==================== Main ====================

def main():
    parser = argparse.ArgumentParser()
    # Backend Selection
    parser.add_argument('--backend', type=str, required=True, choices=['qwen3', 'uivenus'],
                        help="Backend: 'qwen3' (for Qwen3-VL) or 'uivenus' (for UI-Venus)")
    parser.add_argument('--model_type', type=str, required=True,
                        help="Specific model type, e.g. 'ui_venus_ground_7b', 'qwen3_ground_8b'")
    parser.add_argument('--model_name_or_path', type=str, required=False)
    
    # Dataset & Task
    parser.add_argument('--screenspot_imgs', type=str, required=True)
    parser.add_argument('--screenspot_test', type=str, required=True)
    parser.add_argument('--task', type=str, required=True)
    parser.add_argument('--log_path', type=str, required=True)
    
    # Filters
    parser.add_argument('--inst_style', type=str, default='all', choices=INSTRUCTION_STYLES + ['all'])
    parser.add_argument('--language', type=str, default='en', choices=LANGUAGES + ['all'])
    parser.add_argument('--gt_type', type=str, default='all', choices=GT_TYPES + ['all'])

    # Algorithm Hyperparams
    parser.add_argument('--in_depth', type=int, default=3)
    parser.add_argument('--in_ratio', type=float, default=0.5)
    parser.add_argument('--in_min_crop', type=int, default=768)
    parser.add_argument('--patch_size', type=int, default=2, choices=[2, 3, 4])
    parser.add_argument('--center_mode', type=str, default='shift', choices=['shift', 'clip', 'shrink'])
    parser.add_argument('--prezoom_px_thresh', type=float, default=50.0)

    args = parser.parse_args()

    # 1. Build Backend & Runner
    backend = build_backend(args.backend, args.model_type, args.model_name_or_path)
    print(f"Backend {args.backend} built successfully.")

    runner = PrezoomIterativeNarrowingRunner(
        backend=backend,
        max_search_depth=args.in_depth,
        shrink_ratio=args.in_ratio,
        min_crop_size=args.in_min_crop,
        debug=True,
        patch_size=args.patch_size,
        center_mode=args.center_mode,
        prezoom_px_thresh=args.prezoom_px_thresh
    )

    # 2. Prepare Tasks
    if args.task == "all":
        task_filenames = [os.path.splitext(f)[0] for f in os.listdir(args.screenspot_test) if f.endswith(".json")]
    else:
        task_filenames = args.task.split(",")

    inst_styles = INSTRUCTION_STYLES if args.inst_style == "all" else args.inst_style.split(",")
    languages = LANGUAGES if args.language == "all" else args.language.split(",")
    gt_types = GT_TYPES if args.gt_type == "all" else args.gt_type.split(",")

    tasks_to_run = []
    for task_filename in task_filenames:
        path = os.path.join(args.screenspot_test, task_filename + ".json")
        if not os.path.exists(path):
            print(f"[Warn] Task file not found: {path}")
            continue
            
        with open(path, 'r') as f:
            task_data = json.load(f)
            
        for inst_style in inst_styles:
            for gt_type in gt_types:
                for lang in languages:
                    for task_instance in task_data:
                        ti = copy.deepcopy(task_instance)
                        ti["task_filename"] = task_filename
                        ti["gt_type"] = gt_type
                        ti["instruction_style"] = inst_style
                        ti["language"] = lang
                        
                        # CN support check
                        if lang == "cn":
                            if inst_style != 'instruction' or gt_type != 'positive':
                                continue # Skip unsupported CN combination
                            ti["prompt_to_evaluate"] = ti.get("instruction_cn", "")
                        else:
                            ti["prompt_to_evaluate"] = ti.get("instruction", "")
                        
                        tasks_to_run.append(ti)
    
    print(f"Total tasks to run: {len(tasks_to_run)}")

    # 3. Run Inference
    results = []
    for sample in tqdm(tasks_to_run):
        img_path = os.path.join(args.screenspot_imgs, sample["img_filename"])
        if not os.path.exists(img_path):
            print(f"[Error] Image not found: {img_path}")
            continue
            
        image = Image.open(img_path).convert('RGB')
        
        # Inference
        # Grounding
        if sample["gt_type"] == "positive":
            response = runner.ground_only_positive(sample["prompt_to_evaluate"], image)
        else:
            r = runner.ground_only_positive(sample["prompt_to_evaluate"], image)
            # For negative sample: result is correct if point is None (negative)
            # But here we just return what we got, evaluation logic handles correctness
            response = {"result": ("negative" if r["point"] is None else "positive"),
                        "point": r["point"],
                        "raw_response": r.get("raw_response","")}

        if response is None:
            continue

        point_px = response["point"]
        
        sample_result = {
            "id": sample.get("id"),
            "img_path": img_path,
            "group": sample.get("group"),
            "platform": sample.get("platform"),
            "application": sample.get("application"),
            "language": sample.get("language"),
            "instruction_style": sample.get("instruction_style"),
            "prompt_to_evaluate": sample.get("prompt_to_evaluate"),
            "gt_type": sample.get("gt_type"),
            "ui_type": sample.get("ui_type"),
            "task_filename": sample["task_filename"],
            "pred": point_px,
            "raw_response": response.get("raw_response","")
        }

        # 4. Correctness Check (On-the-fly)
        if sample["gt_type"] == "positive":
            correctness = eval_sample_positive_gt(sample, response)
            sample_result["bbox"] = sample.get("bbox")
        else:
            correctness = eval_sample_negative_gt(sample, response)

        sample_result["correctness"] = correctness
        results.append(sample_result)

    # 5. Evaluate & Dump
    report = evaluate(results)
    os.makedirs(os.path.dirname(args.log_path), exist_ok=True)
    with open(args.log_path, 'w') as f:
        json.dump(report, f, indent=4)
    
    print(f"Evaluation finished. Results saved to {args.log_path}")

if __name__ == "__main__":
    main()
