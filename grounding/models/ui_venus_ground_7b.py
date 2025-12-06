import os
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor,AutoTokenizer
from transformers.generation import GenerationConfig

from qwen_vl_utils import process_vision_info, smart_resize


class UI_Venus_Ground_7B():
    def load_model(self, model_name_or_path="/root/ckpt/huggingface/"):
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name_or_path, 
            device_map="auto", # "cuda"
            trust_remote_code=True, 
            dtype=torch.bfloat16,
            local_files_only=True,
            # attn_implementation="flash_attention_2"
        ).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True, local_files_only=True)
        self.processor = AutoProcessor.from_pretrained(model_name_or_path, local_files_only=True)

        # Setting default generation config
        self.generation_config = GenerationConfig.from_pretrained(model_name_or_path, trust_remote_code=True, local_files_only=True).to_dict()
        self.set_generation_config(
            max_length=2048,
            do_sample=False,
            temperature=0.0
        )

    def set_generation_config(self, **kwargs):
        self.generation_config.update(**kwargs)
        self.model.generation_config = GenerationConfig(**self.generation_config)

    def set_top_k_generation(self, k=2):
        self.set_generation_config(
            num_beams=k,
            num_return_sequences=k,
            do_sample=False,
            temperature=0.0,
            early_stopping=True
        )

    def inference(self, instruction, image_path, use_top_k=False, k=2):
        """
        Do inference with top-k generation.
        Args:
            instruction: instruction text
            image_path: image path
            use_top_k: whether to use top-k generation, default False (keep original behavior)
            k: when use_top_k=True, the number of candidates to generate
        """
        assert os.path.exists(image_path) and os.path.isfile(image_path), "Invalid input image path."
        
        prompt_origin = 'Outline the position corresponding to the instruction: {}. The output should be only [x1,y1,x2,y2].'
        full_prompt = prompt_origin.format(instruction)

        min_pixels = 2000000
        max_pixels = 4800000
        
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image", 
                        "image": image_path,
                        "min_pixels": min_pixels,
                        "max_pixels": max_pixels
                    },
                    {"type": "text", "text": full_prompt},
                ],
            }
        ]

        # Preparation for inference
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)

        if use_top_k:
            original_config = self.model.generation_config
            self.set_top_k_generation(k)

        # Inference: Generation of the output
        generated_ids = self.model.generate(**inputs, max_new_tokens=128)
        
        # restore original config
        if use_top_k:
            self.model.generation_config = original_config
            
        if use_top_k and k > 1:
            # for top-k (k>1), generated_ids' shape is (k, sequence_length)
            # inputs.input_ids' shape is (1, input_length), need to expand to match
            input_ids_expanded = inputs.input_ids.repeat(k, 1)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(input_ids_expanded, generated_ids)
            ]
        else:
            # for k=1 or no top-k, use original logic
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
                
        print(output_text)

        input_height = inputs['image_grid_thw'][0][1]*14
        input_width = inputs['image_grid_thw'][0][2]*14

        if use_top_k:
            # process multiple outputs (k>=1)
            results = []
            for i, text in enumerate(output_text):
                try:
                    box = eval(text)
                    abs_y1 = float(box[1]/input_height)
                    abs_x1 = float(box[0]/input_width)
                    abs_y2 = float(box[3]/input_height)
                    abs_x2 = float(box[2]/input_width)
                    box = [abs_x1,abs_y1,abs_x2,abs_y2]
                except:
                    box = [0,0,0,0]

                point = [(box[0]+box[2])/2,(box[1]+box[3])/2]
                result_dict = {
                    "result": "positive",
                    "format": "x1y1x2y2",
                    "raw_response": text,
                    "bbox": box,
                    "point": point,
                    "rank": i + 1  # add ranking info
                }
                results.append(result_dict)
            
            # return all candidates
            return {
                "num_candidates": len(results),
                "candidates": results,
                "raw_responses": output_text
            }
        else:
            # single output (original logic or k=1)
            try:
                box = eval(output_text[0])
                abs_y1 = float(box[1]/input_height)
                abs_x1 = float(box[0]/input_width)
                abs_y2 = float(box[3]/input_height)
                abs_x2 = float(box[2]/input_width)
                box = [abs_x1,abs_y1,abs_x2,abs_y2]
            except:
                box = [0,0,0,0]

            point = [(box[0]+box[2])/2,(box[1]+box[3])/2]
            result_dict = {
                "result": "positive",
                "format": "x1y1x2y2",
                "raw_response": output_text,
                "bbox": box,
                "point": point
            }
            
            return result_dict

    def pick(self, instruction, image_path, candidates: list[list[float]]) -> list[float]:
        prompt_for_pick_origin = """You are given a UI screenshot and a user command: {instruction}.There are {num_candidates} candidate UI elements: {candidates}.Based on the command and the description of candidates,choose which candidate better matches the command.Output only the most preferred coordinate in the format [x1,x2,y1,y2]."""
        prompt_for_pick = prompt_for_pick_origin.format(instruction=instruction, num_candidates=len(candidates), candidates=candidates)
        
        min_pixels = 2000000
        max_pixels = 4800000
        
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image", 
                        "image": image_path,
                        "min_pixels": min_pixels,
                        "max_pixels": max_pixels
                    },
                    {"type": "text", "text": prompt_for_pick},
                ],
            }
        ]

        # Preparation for inference
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)

        generated_ids = self.model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
                
        print(output_text)

        input_height = inputs['image_grid_thw'][0][1]*14
        input_width = inputs['image_grid_thw'][0][2]*14
        try:
            box = eval(output_text[0])
            abs_y1 = float(box[1]/input_height)
            abs_x1 = float(box[0]/input_width)
            abs_y2 = float(box[3]/input_height)
            abs_x2 = float(box[2]/input_width)
            box = [abs_x1,abs_y1,abs_x2,abs_y2]
        except:
            box = [0,0,0,0]

        point = [(box[0]+box[2])/2,(box[1]+box[3])/2]
        result_dict = {
            "result": "positive",
            "format": "x1y1x2y2",
            "raw_response": output_text,
            "bbox": box,
            "point": point
        }
        
        return result_dict
