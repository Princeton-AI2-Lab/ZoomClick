# models/qwen3_vl.py
import json
import base64
from io import BytesIO
from typing import Tuple, Optional
from PIL import Image

import torch
from transformers import AutoProcessor, AutoModelForVision2Seq
from transformers.models.qwen2_vl.image_processing_qwen2_vl_fast import smart_resize

# ============== helpers ==============
def _img2b64(img: Image.Image) -> str:
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

def _build_tools_blob(screen_w: int, screen_h: int):
    return {
        "type": "function",
        "function": {
            "name_for_human": "computer_use",
            "name": "computer_use",
            "description": (
                "Use a mouse and keyboard to interact with a computer, and take screenshots.\n"
                f"* The screen's resolution is {screen_w}x{screen_h}.\n"
                "* Always consult the screenshot to determine coordinates before clicking."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {"type": "string",
                               "enum": ["key","type","mouse_move","left_click","left_click_drag",
                                        "right_click","middle_click","double_click","scroll","wait","terminate"]},
                    "keys": {"type": "array"},
                    "text": {"type": "string"},
                    "coordinate": {"type": "array", "description": "[x,y] or [x1,y1,x2,y2] in pixels"},
                    "pixels": {"type": "number"},
                    "time": {"type": "number"},
                    "status": {"type": "string", "enum": ["success","failure"]}
                },
                "required": ["action"]
            },
            "args_format": "Format the arguments as a JSON object."
        }
    }

def _build_messages(image: Image.Image, instruction: str, screen_w: int, screen_h: int):
    tools_blob = _build_tools_blob(screen_w, screen_h)
    sys_content = [
        {"type": "text", "text": "You are a helpful assistant that can click on elements in screenshots."},
        {"type": "text", "text":
            "You may call one or more functions to assist with the user query.\n"
            "You are provided with function signatures within <tools></tools> XML tags:\n"
            "<tools>\n" + json.dumps(tools_blob) + "\n</tools>\n\n"
            "IMPORTANT: You MUST respond with a tool call to click on the requested element.\n"
            "For each function call, return a json object within <tool_call>...</tool_call> tags:\n"
            "<tool_call>\n"
            "{\"name\": <function-name>, \"arguments\": <args-json-object>}\n"
            "</tool_call>\n\n"
            "Example: To click at coordinates (500, 300), respond with:\n"
            "<tool_call>\n"
            "{\"name\": \"computer_use\", \"arguments\": {\"action\": \"left_click\", \"coordinate\": [500, 300]}}\n"
            "</tool_call>"
        }
    ]
    user_content = [
        {"type": "image_url", "image_url": {"url": "data:image/png;base64," + _img2b64(image)}},
        {"type": "text", "text": f"Please click on the element described as: {instruction}. Respond with a tool call containing the exact pixel coordinates."},
    ]
    return [
        {"role": "system", "content": sys_content},
        {"role": "user", "content": user_content},
    ]

def _trim_to_json(s: str) -> str:
    last = s.rfind("}")
    return s[:last+1] if last != -1 else s

# ============== Qwen3-VL (cookbook 风格) ==============
class Qwen3VLModel:
    def __init__(self):
        self.model = None
        self.processor = None
        self.generation_config = {
            "max_new_tokens": 256,
            "do_sample": False,
            "temperature": 0.0,
        }

    def load_model(self, model_path: str = "Qwen/Qwen3-VL-4B-Instruct"):
        self.processor = AutoProcessor.from_pretrained(model_path)
        if getattr(self.processor.tokenizer, "pad_token_id", None) is None:
            self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token

        self.model, _ = AutoModelForVision2Seq.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map="auto",
            output_loading_info=True
        )
        self.model.eval()

    def set_generation_config(self, **kwargs):
        self.generation_config.update(**kwargs)

    @torch.inference_mode()
    def ground_only_positive(self, instruction: str, image) -> dict:
        """
        返回:
        {
          "result": "positive",
          "point": [x_px, y_px] | None,
          "raw_response": "<tool_call>...json...</tool_call>",
          "format": "x1y1x2y2"
        }
        """
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        assert isinstance(image, Image.Image)

        # 使用官方的 smart_resize 函数
        patch_size = self.processor.image_processor.patch_size
        merge_size = self.processor.image_processor.merge_size
        resized_height, resized_width = smart_resize(
            image.height,
            image.width,
            factor=patch_size * merge_size,
            min_pixels=patch_size * patch_size * merge_size * merge_size * 16,
            max_pixels=patch_size * patch_size * merge_size * merge_size * 6400,
        )
        
        resized = image.resize((resized_width, resized_height))

        messages = _build_messages(resized, instruction, resized_width, resized_height)

        guide_prefix = '<tool_call>\n{"name": "computer_use", "arguments": {"action": "left_click", "coordinate": ['

        prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        ) + guide_prefix
        
        inputs = self.processor(
            text=[prompt],
            images=[resized],
            padding=True,
            return_tensors="pt"
        ).to(self.model.device)

        gen_ids = self.model.generate(**inputs, **self.generation_config)
        trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, gen_ids)]
        resp = self.processor.batch_decode(
            trimmed, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )[0]

        resp = guide_prefix + resp
        resp = _trim_to_json(resp)
        if not resp.endswith("</tool_call>"):
            resp += "\n</tool_call>"

        result = {
            "result": "positive",
            "format": "x1y1x2y2",
            "raw_response": resp,
            "point": None,
            "bbox": None,
        }

        try:
            inner = resp.split("<tool_call>")[1].split("</tool_call>")[0]
            obj = json.loads(_trim_to_json(inner))
            coords = obj["arguments"]["coordinate"]
            if len(coords) == 2:
                x, y = coords
            elif len(coords) == 4:
                x1, y1, x2, y2 = coords
                x, y = (x1 + x2) / 2, (y1 + y2) / 2
            else:
                raise ValueError("coordinate must be [x,y] or [x1,y1,x2,y2]")
            
            # Convert from 1000x1000 normalized coordinates to actual pixel coordinates
            # Model outputs coordinates normalized to 1000x1000, need to scale to actual image size
            actual_x = float(x) / 1000.0 * resized_width
            actual_y = float(y) / 1000.0 * resized_height
            result["point"] = [actual_x, actual_y]
        except Exception:
            pass

        return result

    def ground_allow_negative(self, instruction: str, image):
        r = self.ground_only_positive(instruction, image)
        if r.get("point") is None:
            return {"result": "negative", "point": None, "raw_response": r.get("raw_response","")}
        return r
    
    def simple_text_response(self, instruction: str, image) -> dict:
        """
        用于候选点选择，返回简单的文本响应而不是tool_call格式
        """
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        assert isinstance(image, Image.Image)

        # 使用官方的 smart_resize 函数
        patch_size = self.processor.image_processor.patch_size
        merge_size = self.processor.image_processor.merge_size
        resized_height, resized_width = smart_resize(
            image.height,
            image.width,
            factor=patch_size * merge_size,
            min_pixels=patch_size * patch_size * merge_size * merge_size * 16,
            max_pixels=patch_size * patch_size * merge_size * merge_size * 6400,
        )
        
        resized = image.resize((resized_width, resized_height))

        # 构建简单的消息，不使用tool_call格式
        messages = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "You are a helpful assistant that can analyze images and respond with simple text."}
                ]
            },
            {
                "role": "user", 
                "content": [
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64," + _img2b64(resized)}},
                    {"type": "text", "text": instruction}
                ]
            }
        ]

        prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        inputs = self.processor(
            text=[prompt],
            images=[resized],
            padding=True,
            return_tensors="pt"
        ).to(self.model.device)

        gen_ids = self.model.generate(**inputs, **self.generation_config)
        trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, gen_ids)]
        resp = self.processor.batch_decode(
            trimmed, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )[0]
        
        return {
            "raw_response": resp,
            "point": None  # 这个方法不返回坐标点
        }