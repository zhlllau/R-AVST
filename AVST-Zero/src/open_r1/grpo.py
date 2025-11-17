# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
import numpy as np
import os
import re
from collections import defaultdict
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional

from datasets import load_dataset, load_from_disk, Dataset, DatasetDict
from transformers import Qwen2VLForConditionalGeneration

from math_verify import parse, verify
from src.open_r1.trainer import Qwen2VLGRPOTrainer_Video_GQA as Qwen2VLGRPOTrainer
from trl import GRPOConfig, GRPOTrainer, ModelConfig, ScriptArguments, TrlParser, get_peft_config
from src.open_r1.my_qwen_utils import process_vision_info
from tqdm import tqdm
import torch
import json
import random
import ast

import gensim.models.keyedvectors as kv
from gensim.utils import simple_preprocess
# Load checkpoints
from deepspeed.runtime.zero.config import ZeroStageEnum
torch.serialization.add_safe_globals([ZeroStageEnum])

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'iou', 'format', 'answer'.
    """
    reward_funcs: list[str] = field(
        default_factory=lambda: ["iou", "format", "answer"],
        metadata={"help": "List of reward functions. Possible values: 'iou', 'format'"},
    )
    max_pixels: Optional[int] = field(
        default=12845056,
        metadata={"help": "Maximum number of pixels for the image"},
    )
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image"},
    )
    train_data_path: str = field(
        default="",
        metadata={"help": "Path to the training data JSON file."},
    )
    eval_data_path: str = field(
        default="",
        metadata={"help": "Path to the evaluation data JSON file."},
    )
    video_folder: str = field(
        default="",
        metadata={"help": "Path to the folder containing video files."},
    )

def is_valid_bbox_line(line):
    pattern = r'^\s*\d+(\.\d+)?:\s*\[\s*\d+(\.\d+)?,\s*\d+(\.\d+)?,\s*\d+(\.\d+)?,\s*\d+(\.\d+)?\s*\]\s*$'
    return re.match(pattern, line.strip()) is not None

class ARIGGRPOReward:
    def __init__(self):
        """
        Reward based on IoU between multiple bounding boxes.
        """

    @staticmethod
    def compute_iou(gt_bbox, pred_bbox):
        """
        Compute 2D IoU between two bounding boxes.

        Args:
            gt_bbox (tuple): Ground truth bounding box as (x1, y1, x2, y2)
            pred_bbox (tuple): Predicted bounding box as (x1, y1, x2, y2)

        Returns:
            float: IoU score.
        """
        x1_gt, y1_gt, x2_gt, y2_gt = gt_bbox
        x1_pred, y1_pred, x2_pred, y2_pred = pred_bbox

        ix1 = max(x1_gt, x1_pred)
        iy1 = max(y1_gt, y1_pred)
        ix2 = min(x2_gt, x2_pred)
        iy2 = min(y2_gt, y2_pred)

        inter_width = max(0, ix2 - ix1)
        inter_height = max(0, iy2 - iy1)
        intersection_area = inter_width * inter_height

        gt_area = (x2_gt - x1_gt) * (y2_gt - y1_gt)
        pred_area = (x2_pred - x1_pred) * (y2_pred - y1_pred)
        union_area = gt_area + pred_area - intersection_area
        iou = intersection_area / union_area if union_area > 0 else 0.0
        
        return iou
        

    def group_by_object(self, data):
        """
        Group bounding boxes by object name and aggregate their intervals.
        """
        grouped = defaultdict(list)
        for item in data:
            key = item['object_name']
            grouped[key].append((item['x1'], item['y1'], item['x2'], item['y2']))
        return grouped

    def compute_reward(self, preds, refs):
        """
        Compute IoU reward for each object and average.

        Args:
            preds (list of dict): Predicted bounding boxes.
            refs (list of dict): Ground truth bounding boxes.

        Returns:
            float: Final reward score (average IoU).
        """
        if len(preds) == 0 and len(refs) == 0:
            return 1.0
        pred_groups = self.group_by_object(preds)
        ref_groups = self.group_by_object(refs)

        scores = []
        for object_name in ref_groups:
            gt_bboxes = ref_groups[object_name]
            pred_bboxes = pred_groups.get(object_name, [])
            if not gt_bboxes:
                continue

            ious = [max([self.compute_iou(gt_bbox, pred) for pred in pred_bboxes], default=0.0)
                    for gt_bbox in gt_bboxes]
            scores.append(np.mean(ious))

        return float(np.mean(scores)) if scores else 0.0

class VTGGRPOReward:
    def __init__(self, ignore_type=False):
        """
        Reward based on multi-segment event-level temporal IoU.
        """
        self.ignore_type = ignore_type

    @staticmethod
    def merge_intervals(intervals):
        """
        Merge overlapping intervals.
        """
        if not intervals:
            return []

        intervals = sorted(intervals, key=lambda x: x[0])
        merged = [intervals[0]]
        for current in intervals[1:]:
            last = merged[-1]
            if current[0] <= last[1]: 
                merged[-1] = (last[0], max(last[1], current[1]))
            else:
                merged.append(current)
        return merged

    @staticmethod
    def compute_tiou(gt_intervals, pred_intervals):
        """
        Compute tIoU between two sets of segments.
        """
        if not gt_intervals and not pred_intervals:
            return 1.0  
        if not gt_intervals or not pred_intervals:
            return 0.0

        gt_merged = VTGGRPOReward.merge_intervals(gt_intervals)
        pred_merged = VTGGRPOReward.merge_intervals(pred_intervals)

        inter = 0.0
        for gs, ge in gt_merged:
            for ps, pe in pred_merged:
                inter += max(0, min(ge, pe) - max(gs, ps))

        all_intervals = gt_merged + pred_merged
        union = 0.0
        for s, e in VTGGRPOReward.merge_intervals(all_intervals):
            union += e - s

        return inter / union if union > 0 else 0.0

    def group_by_event(self, data):
        """
        Group segments by (type, event) or (all, event) and collect all intervals.
        """
        grouped = defaultdict(list)
        for item in data:
            key = (item['type'] if not self.ignore_type else 'all', item['event'])
            grouped[key].append((item['start'], item['end']))
        return grouped

    def compute_reward(self, preds, refs):
        """
        Compute multi-segment tIoU per event and average.
        """
        pred_groups = self.group_by_event(preds)
        ref_groups = self.group_by_event(refs)

        scores = []
        for key in ref_groups:
            gt_intervals = ref_groups[key]
            pred_intervals = pred_groups.get(key, [])
            tiou = self.compute_tiou(gt_intervals, pred_intervals)
            scores.append(tiou)

        return float(np.mean(scores)) if scores else 0.0

def object_reward(prediction_objects, ground_truth_objects):
    prediction_set = set(prediction_objects)
    ground_truth_set = set(ground_truth_objects)
    correct_objects = prediction_set.intersection(ground_truth_set)
    total_objects = len(ground_truth_set)
    return len(correct_objects) / total_objects if total_objects > 0 else 0.0


def normalize_object_name(name: str) -> str:
    name = name.lower().strip()
    name = re.sub(r'^(a |an |the )', '', name)
    name = re.sub(r'[^\w\s]', '', name)
    return name.strip()

def extract_ground_truth(solution):
    ground_truth = defaultdict(list)

    object_pattern = re.compile(r'<object>(.*?)</object>', re.DOTALL)
    where_pattern = re.compile(r'<where>(.*?)</where>', re.DOTALL)
    
    object_matches = object_pattern.findall(solution)
    where_matches = where_pattern.findall(solution)
    
    for obj_name, where_data in zip(object_matches, where_matches):
        obj_name = normalize_object_name(obj_name.strip())
        lines = where_data.strip().split('\n')
        for line in lines:
            if not line:
                continue
            parts = line.split(': ')
            if len(parts) != 2:
                continue 
            time_str, bbox_str = parts
            try:
                time_val = round(float(time_str))
                bbox_vals = [float(x) for x in bbox_str.strip('[]').split(',')]
                ground_truth['S_spatial'].append({
                    'object': obj_name,
                    'time': time_val,
                    'bbox': bbox_vals
                })
            except ValueError:
                continue  

    temporal_pattern = re.compile(r'<object>(.*?)</object>\n<when>\[(.*?)\]</when>', re.DOTALL)
    temporal_matches = temporal_pattern.findall(solution)
    
    for obj_name, when_data in temporal_matches:
        obj_name = normalize_object_name(obj_name.strip())
        start_time, end_time = map(float, when_data.split(','))
        ground_truth['T_temporal'].append({
            'object': obj_name,
            'start_time': start_time,
            'end_time': end_time
        })

    st_pattern = re.compile(r'<object>(.*?)</object>\n<when>\[(.*?)\]</when>\n<where>(.*?)</where>', re.DOTALL)
    st_matches = st_pattern.findall(solution)
    
    for obj_name, when_data, where_data in st_matches:
        obj_name = normalize_object_name(obj_name.strip())
        start_time, end_time = map(float, when_data.split(','))
        lines = where_data.strip().split('\n')
        for line in lines:
            if not line:
                continue
            parts = line.split(': ')
            if len(parts) != 2:
                continue  
            time_str, bbox_str = parts
            try:
                time_val = round(float(time_str))
                bbox_vals = [float(x) for x in bbox_str.strip('[]').split(',')]
                ground_truth['ST_spatial'].append({
                    'object': obj_name,
                    'time': time_val,
                    'bbox': bbox_vals
                })
            except ValueError:
                continue  
        ground_truth['ST_temporal'].append({
            'object': obj_name,
            'start_time': start_time,
            'end_time': end_time
        })

    return dict(ground_truth)

def encode_object_name(word2vec_model, object_name):
    words = simple_preprocess(object_name)
    embeddings = [word2vec_model[word] for word in words if word in word2vec_model]
    if embeddings:
        return np.mean(embeddings, axis=0)
    else:
        return np.zeros(word2vec_model.vector_size)

def map_object_name_to_gt(name):
    norm_name = normalize_object_name(name)
    if not norm_name or not isinstance(norm_name, str):
        return norm_name
    word2vec_model = object_mapping_context["word2vec_model"]

    obj_emb = encode_object_name(word2vec_model, norm_name)
    obj_emb_tensor = torch.tensor(obj_emb, dtype=torch.float32).unsqueeze(0) 
    gt_embeddings = torch.tensor(object_mapping_context["gt_embeddings"], dtype=torch.float32)  
    sim = torch.cosine_similarity(obj_emb_tensor, gt_embeddings) 
    best_score, best_idx = torch.max(sim, dim=0)
    try:
        if best_score.item() >= object_mapping_context.get("threshold", 0.75):
            new_name = object_mapping_context["gt_list"][best_idx.item()]
            return new_name
        return norm_name
    except Exception as e:
        return norm_name

def extract_prediction(prompt_answer):
    prediction = defaultdict(list)

    try:
        object_pattern = re.compile(r'<object>(.*?)</object>', re.DOTALL)
        where_pattern = re.compile(r'<where>(.*?)</where>', re.DOTALL)

        object_matches = object_pattern.findall(prompt_answer)
        where_matches = where_pattern.findall(prompt_answer)

        if len(object_matches) != len(where_matches):
            print("[Warning] <object> and <where> count mismatch.")
        
        for obj_name, where_data in zip(object_matches, where_matches):
            try:
                raw_obj = obj_name.strip()
                if object_mapping_context:
                    obj_name = map_object_name_to_gt(raw_obj)
                    print(f"[Map] Object '{raw_obj}' mapped to '{obj_name}'")
                else:
                    obj_name = normalize_object_name(raw_obj)
                    
                lines = where_data.strip().split('\n')
                for line in lines:
                    if not line:
                        continue
                    parts = line.split(': ')
                    if len(parts) != 2:
                        print(f"[Warning] Invalid line format: {line}")
                        continue
                    time_str, bbox_str = parts
                    time_val = round(float(time_str))
                    bbox_vals = [float(x) for x in bbox_str.strip('[]').split(',')]
                    if len(bbox_vals) != 4:
                        print(f"[Warning] Invalid bbox length: {bbox_vals}")
                        continue
                    prediction['S_spatial'].append({
                        'object': obj_name,
                        'time': time_val,
                        'bbox': bbox_vals
                    })
                    prediction['ST_spatial'].append({
                        'object': obj_name,
                        'time': time_val,
                        'bbox': bbox_vals
                    })
            except Exception as e:
                print(f"[Error] Failed to parse <where> block for object '{obj_name}': {e}")
                continue

        when_match = re.search(r'<when>\[(.*?)\]</when>', prompt_answer)
        if when_match:
            try:
                start_time, end_time = map(float, when_match.group(1).split(','))
                prediction['T_temporal'].append({
                    'start_time': start_time,
                    'end_time': end_time
                })
                prediction['ST_temporal'].append({
                    'start_time': start_time,
                    'end_time': end_time
                })
            except Exception as e:
                print(f"[Error] Failed to parse <when> value: {e}")
    except Exception as e:
        print(f"[Error] extract_prediction failed: {e}")

    return dict(prediction)


def spatial_temporal_reward(completions, solution, durations, reasoning_type, object_mapping_context=None):
    arig_reward = ARIGGRPOReward()
    vtgg_reward = VTGGRPOReward()
    
    rewards = []

    for completion, sol, reason_type in zip(completions, solution, reasoning_type):
        match = re.search(r"'answer':\s*'(.*?)'}", completion, re.DOTALL)
        if match:
            completion = match.group(1)
        import codecs
        completion = codecs.decode(completion, 'unicode_escape')

        prediction = extract_prediction(completion)
        ground_truth = extract_ground_truth(sol["answer"])
        reward = 0.0

        prediction_objects = [x['object'] for x in prediction.get('S_spatial', [])]
        ground_truth_objects = [normalize_object_name(x['object']) for x in ground_truth.get('S_spatial', [])]
        object_reward_score = object_reward(prediction_objects, ground_truth_objects)

        if reason_type == 'AV Temporal Reasoning':
            temporal_reward = vtgg_reward.compute_reward(
                [{'type': 'temporal', 'event': 'event1', 'start': pred['start_time'], 'end': pred['end_time']} 
                 for pred in prediction.get('T_temporal', [])],
                [{'type': 'temporal', 'event': 'event1', 'start': gt['start_time'], 'end': gt['end_time']} 
                 for gt in ground_truth.get('T_temporal', [])]
            )
            reward = temporal_reward

        elif reason_type == 'AV Spatial Reasoning':
            pred_spatial = prediction.get('S_spatial', [])
            gt_spatial = ground_truth.get('S_spatial', [])

            pred_dict = defaultdict(lambda: defaultdict(list))
            gt_dict = defaultdict(lambda: defaultdict(list))

            for pred in pred_spatial:
                pred_dict[pred['time']][pred['object']].append(pred['bbox'])
            for gt in gt_spatial:
                gt_dict[gt['time']][gt['object']].append(gt['bbox'])

            spatial_rewards = []
            for time in set(pred_dict.keys()).intersection(gt_dict.keys()):
                for obj in set(pred_dict[time].keys()).intersection(gt_dict[time].keys()):
                    spatial_reward = arig_reward.compute_reward(
                        [{'object_name': obj, 'x1': b[0], 'y1': b[1], 'x2': b[2], 'y2': b[3]} for b in pred_dict[time][obj]],
                        [{'object_name': obj, 'x1': b[0], 'y1': b[1], 'x2': b[2], 'y2': b[3]} for b in gt_dict[time][obj]],
                    )
                    spatial_rewards.append(spatial_reward)

            if spatial_rewards:
                spatial_score = sum(spatial_rewards) / len(spatial_rewards)
                reward = (object_reward_score + spatial_score) / 2
            else:
                reward = object_reward_score / 2

        elif reason_type == 'AV ST Reasoning':
            temporal_reward = vtgg_reward.compute_reward(
                [{'type': 'temporal', 'event': 'event1', 'start': pred['start_time'], 'end': pred['end_time']} 
                 for pred in prediction.get('ST_temporal', [])],
                [{'type': 'temporal', 'event': 'event1', 'start': gt['start_time'], 'end': gt['end_time']} 
                 for gt in ground_truth.get('ST_temporal', [])]
            )

            pred_spatial = prediction.get('ST_spatial', [])
            gt_spatial = ground_truth.get('ST_spatial', [])

            pred_dict = defaultdict(lambda: defaultdict(list))
            gt_dict = defaultdict(lambda: defaultdict(list))

            for pred in pred_spatial:
                pred_dict[pred['time']][pred['object']].append(pred['bbox'])
            for gt in gt_spatial:
                gt_dict[gt['time']][gt['object']].append(gt['bbox'])

            spatial_rewards = []
            for time in set(pred_dict.keys()).intersection(gt_dict.keys()):
                for obj in set(pred_dict[time].keys()).intersection(gt_dict[time].keys()):
                    spatial_reward = arig_reward.compute_reward(
                        [{'object_name': obj, 'x1': b[0], 'y1': b[1], 'x2': b[2], 'y2': b[3]} for b in pred_dict[time][obj]],
                        [{'object_name': obj, 'x1': b[0], 'y1': b[1], 'x2': b[2], 'y2': b[3]} for b in gt_dict[time][obj]],
                    )
                    spatial_rewards.append(spatial_reward)

            if spatial_rewards:
                spatial_score = sum(spatial_rewards) / len(spatial_rewards)
                reward = (temporal_reward + spatial_score) / 2
            else:
                reward = temporal_reward / 2
        rewards.append(reward)
    return rewards

def format_reward(completions):
    reward_list = []
    for content in completions:
        content = content.strip()

        answer_match = re.search(r'<answer>(.*?)</answer>', content, re.DOTALL)
        if not answer_match:
            reward_list.append(0.0)
            continue

        answer_text = answer_match.group(1).strip()

        when_match = re.search(r'<when>\s*\[\s*(\d+(\.\d+)?),\s*(\d+(\.\d+)?)\s*\]\s*</when>', answer_text)
        if not when_match:
            reward_list.append(0.0)
            continue

        object_where_pattern = re.compile(r'<object>(.*?)</object>\s*<where>(.*?)</where>', re.DOTALL)
        object_where_matches = object_where_pattern.findall(answer_text)

        if not object_where_matches:
            reward_list.append(0.0)
            continue

        valid = True
        for obj_name, where_block in object_where_matches:
            lines = where_block.strip().splitlines()
            for line in lines:
                if not is_valid_bbox_line(line):
                    valid = False
                    break
            if not valid:
                break

        reward_list.append(1.0 if valid else 0.0)

    return reward_list

def iou_glue_reward(completions, solution, durations):
    pass

def answer_reward(completions, solution):
    pass

reward_funcs_registry = {
    "iou": iou_glue_reward,
    "answer": answer_reward,
    "format": format_reward,
    "spatial-temporal": spatial_temporal_reward
    
}

def zl_load_json_dataset(train_data_path, eval_data_path, video_folder):
    def create_dataset_from_json(file_path, split_name):
        with open(file_path, 'r', encoding="utf-8") as f:
            data = json.load(f)
        examples = []
        
        for info in data:
            video_name = info['video_name']
            video_path = os.path.join(video_folder, video_name)
            video_path = video_path + ".mp4"
            
            for event in info['annotation']:
                event_num = event['event_num']
                
                for reasoning_type in ['AV Spatial Reasoning', 'AV Temporal Reasoning', 'AV ST Reasoning']:
                    if reasoning_type in event:
                        for question in event[reasoning_type]:
                            q = question['Q']
                            a = question['A']
                            
                            example = {
                                "problem": {"question": q}, 
                                "solution": {"answer": a},  
                                "video_path": video_path,
                                "durations": info['duration'],
                                "event_num": event_num,
                                "reasoning_type": reasoning_type
                            }
                            
                            examples.append(example)
        
        random.shuffle(examples)
        dataset = Dataset.from_list(examples)
        dataset.client = None
        
        def __getitem__(self, idx):
            example = self[idx]
            data_to_return = {k: v for k, v in example.items()} 
            
            try:
                messages = [{"role": "user", "content": [{"type": "video", "video": example["video_path"][0], "total_pixels": 3584 * 28 * 28, "min_pixels": 16 * 28 * 28,}]}]
                image_inputs, video_inputs, video_kwargs = process_vision_info([messages], return_video_kwargs=True, client=self.client)
                fps_inputs = video_kwargs['fps']
                data_to_return["video_inputs"] = [video_inputs]
                data_to_return["video_kwargs"] = [video_kwargs]
          
            except Exception as e:
                print(f"Warning: Error loading preprocessed data from {example['video_path']}, falling back to video_path. Error: {e}")
                idx = idx + 1
                return self.__getitem__(idx)
            
            return data_to_return
        
        dataset.__getitem__ = __getitem__.__get__(dataset, Dataset) 
        
        return dataset
    
    train_dataset = create_dataset_from_json(train_data_path, "train")
    eval_dataset = create_dataset_from_json(eval_data_path, "eval")
    
    return DatasetDict({"train": train_dataset, "eval": eval_dataset})

# If you want to use this function, you need to first process the video files with generate_npy.py to generate npy files. This will speed up the video loading process.
# def zl_load_json_dataset(train_data_path, eval_data_path, video_folder, npy_folder="./video_npy", default_fps=0.25):
#     def create_dataset_from_json(file_path, split_name):
#         with open(file_path, 'r', encoding="utf-8") as f:
#             data = json.load(f)
#         examples = []

#         for info in data:
#             video_name = info['video_name']
#             video_id = os.path.splitext(video_name)[0]
#             npy_path = os.path.join(npy_folder, f"{video_id}.npy")

#             for event in info['annotation']:
#                 event_num = event['event_num']

#                 for reasoning_type in ['AV Spatial Reasoning', 'AV Temporal Reasoning', 'AV ST Reasoning']:
#                     if reasoning_type in event:
#                         for question in event[reasoning_type]:
#                             q = question['Q']
#                             a = question['A']

#                             example = {
#                                 "problem": {"question": q},
#                                 "solution": {"answer": a},
#                                 "video_npy_path": npy_path,
#                                 "durations": info['duration'],
#                                 "event_num": event_num,
#                                 "reasoning_type": reasoning_type,
#                                 "video_id": video_id
#                             }

#                             examples.append(example)

#         random.shuffle(examples)
#         dataset = Dataset.from_list(examples)
#         dataset.client = None

#         def __getitem__(self, idx):
#             example = self[idx]
#             data_to_return = {k: v for k, v in example.items()}
#             npy_path = example["video_npy_path"][0]
#             if not os.path.exists(npy_path):
#                 idx[0] = idx[0] + 1
#                 return self.__getitem__(idx)
#             else:
#                 print(f"[Info] Loading npy from {npy_path} for idx={idx}")
                
#             try:
#                 video_array = np.load(npy_path) 
#                 video_tensor = torch.tensor(video_array)
#                 video_tensor = torch.tensor(video_array, dtype=torch.float32)
#                 video_tensor.clamp_(0, 255)
#                 data_to_return["video_inputs"] = [video_tensor]
#                 data_to_return["video_kwargs"] = [{"fps": [default_fps]}]
#             except Exception as e:
#                 idx = idx + 1
#                 return self.__getitem__(idx)

#             return data_to_return

#         dataset.__getitem__ = __getitem__.__get__(dataset, Dataset)
#         return dataset

#     train_dataset = create_dataset_from_json(train_data_path, "train")
#     eval_dataset = create_dataset_from_json(eval_data_path, "eval")

#     return DatasetDict({"train": train_dataset, "eval": eval_dataset})



def main(script_args, training_args, model_args):
    logger.info("Starting the training process")
    
    # Load GT vocab
    gt_vocab = torch.load("./models/Train_gt_object_vocab.pt")
    gt_object_list = gt_vocab["gt_object_list"]
    gt_embeddings = gt_vocab["gt_embeddings"]

    # Load pre-tarined Word2Vec model
    word2vec_model_path = "./models/GoogleNews-vectors-negative300.bin"
    word2vec_model = kv.KeyedVectors.load_word2vec_format(word2vec_model_path, binary=True)
    global object_mapping_context
    object_mapping_context = {
        "gt_list": gt_object_list,
        "gt_embeddings": gt_embeddings,
        "word2vec_model": word2vec_model,
        "threshold": 0.75
    }
    
    # Reward funcs
    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]
    # Load dataset
    dataset = zl_load_json_dataset(
        script_args.train_data_path,
        script_args.eval_data_path,
        script_args.video_folder
    )
    if not training_args.use_vllm:
        trainer_cls = Qwen2VLGRPOTrainer
    else:
        raise NotImplementedError
    logger.info("Initializing the GRPO trainer")
    trainer = trainer_cls(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        peft_config=get_peft_config(model_args),
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels
    )
    logger.info("Starting training")
    trainer.train()
    logger.info("Saving the model")
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        logger.info("Pushing the model to the Hub")
        trainer.push_to_hub(dataset_name=script_args.dataset_name)

if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    
    gpu_env = os.environ.get("CUDA_VISIBLE_DEVICES", "All")
    print(f"Running on CUDA_VISIBLE_DEVICES={gpu_env}")
    
    main(script_args, training_args, model_args)
