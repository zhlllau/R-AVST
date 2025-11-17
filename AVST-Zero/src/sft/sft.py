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

"""
Supervised fine-tuning script for decoder language models.

Usage:

# One 1 node of 8 x H100s
accelerate launch --config_file=configs/zero3.yaml src/open_r1/sft.py \
    --model_name_or_path Qwen/Qwen2.5-1.5B-Instruct \
    --dataset_name HuggingFaceH4/Bespoke-Stratos-17k \
    --learning_rate 2.0e-5 \
    --num_train_epochs 1 \
    --packing \
    --max_seq_length 4096 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --gradient_checkpointing \
    --bf16 \
    --logging_steps 5 \
    --eval_strategy steps \
    --eval_steps 100 \
    --output_dir data/Qwen2.5-1.5B-Open-R1-Distill
"""

import logging
import os
import sys
import json
import random
from dataclasses import dataclass, field
from typing import Optional
import trl
import torch
import datasets
from datasets import Dataset, DatasetDict
import transformers
from transformers import (
    AutoProcessor, AutoTokenizer, set_seed, 
    Qwen2_5OmniForConditionalGeneration,
    Qwen2_5OmniThinkerForConditionalGeneration,
    Qwen2_5OmniProcessor,
)
from transformers.trainer_utils import get_last_checkpoint

from trl import (
    ModelConfig, ScriptArguments, SFTTrainer, TrlParser,
    get_kbit_device_map, get_peft_config, get_quantization_config,
)

from tqdm import tqdm
from src.open_r1.my_qwen_utils import process_vision_info

logger = logging.getLogger(__name__)

@dataclass
class SFTScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'iou', 'format'.
    """

    train_data_path: str = field(
        default="./Charades/charades_annotation/train.json",
        metadata={"help": "Path to the training data JSON file."},
    )
    eval_data_path: str = field(
        default="./Charades/charades_annotation/val.json",
        metadata={"help": "Path to the evaluation data JSON file."},
    )

    video_folder: str = field(
        default="./Charades/Charades_v1",  # Replace with your actual video folder path
        metadata={"help": "Path to the folder containing video files."},
    )
    preprocessed_data_path: Optional[str] = field( # Add preprocessed_data_path argument
        default="",
        metadata={"help": "Path to the preprocessed dataset directory. If provided, load preprocessed data instead of raw videos."},
    )

@dataclass
class SFTConfig(trl.SFTConfig):
    """
    args for callbacks, benchmarks etc
    """

    benchmarks: list[str] = field(
        default_factory=lambda: [], metadata={"help": "The benchmarks to run after training."}
    )
    callbacks: list[str] = field(
        default_factory=lambda: [], metadata={"help": "The callbacks to run during training."}
    )
    system_prompt: Optional[str] = field(
        default=None,
        metadata={"help": "The optional system prompt to use for benchmarking."},
    )
    hub_model_revision: Optional[str] = field(
        default="main",
        metadata={"help": "The Hub model branch to push the model to."},
    )
    overwrite_hub_revision: bool = field(default=False, metadata={"help": "Whether to overwrite the Hub revision."})
    push_to_hub_revision: bool = field(default=False, metadata={"help": "Whether to push to a Hub revision/branch."})


def zl_load_json_dataset(train_data_path, eval_data_path, video_folder, preprocessed_data_path=None):
    def create_dataset_from_json(file_path, split_name):
        with open(file_path, 'r', encoding="utf-8") as f:
            data = json.load(f)

        examples = []

        for info in data:
            video_name = info['video_name']
            video_path = os.path.join(video_folder, video_name + ".mp4")

            for event in info.get('annotation', []):
                event_num = event['event_num']

                for reasoning_type in ['AV Spatial Reasoning', 'AV Temporal Reasoning', 'AV ST Reasoning']:
                    if reasoning_type in event:
                        for question in event[reasoning_type]:
                            q = question['Q']
                            a = question['A']

                            example = {
                                "problem": {"question": q},
                                "answer": {"answer": a},
                                "video_path": video_path,
                                "durations": info['duration'],
                                "event_num": event_num,
                                "reasoning_type": reasoning_type,
                                "preprocessed_path": video_path
                            }
                            examples.append(example)

        random.shuffle(examples)
        dataset = Dataset.from_list(examples)
        dataset.client = None

        def __getitem__(self, idx):
            example = self[idx]
            data_to_return = {k: v for k, v in example.items()}
            messages = convert_example(dataset[idx[0]])['messages']

            image_inputs, video_inputs, video_kwargs = process_vision_info(
                [messages], return_video_kwargs=True, client=self.client
            )

            data_to_return["video_inputs"] = [video_inputs]
            data_to_return["video_kwargs"] = [video_kwargs]
            data_to_return["use_preprocessed"] = [True]

            return data_to_return

        dataset.__getitem__ = __getitem__.__get__(dataset, Dataset)
        return dataset

    train_dataset = create_dataset_from_json(train_data_path, "train")
    eval_dataset = create_dataset_from_json(eval_data_path, "eval")
    return DatasetDict({"train": train_dataset, "eval": eval_dataset})


processor = None

SYSTEM_PROMPT = (
                '''You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech.
                You are an expert in audio-visual video grounding.

                For each question, output within <answer></answer>:
                - <object>: grounded object(s), e.g. 'man', 'dog', separated by commas
                - <when>: [start, end] in seconds, 1 decimal
                - <where> for each object by each second within the object appearing duration:
                timestamp: [x1, y1, x2, y2]

                Only include tags <object>, <when> and <where>. No extra text.

                Example:
                <answer>
                <when>[10.0,20.5]</when>
                <object>dog</object>
                <where>
                10.0: [100,200,300,400]
                11.0: [109,280,320,432]
                12.0: [100,200,300,400]
                </where>
                <object>cat</object>
                <where>
                12.5: [50,60,150,160]
                13.5: [55,62,140,150]
                </where>
                </answer>'''
)


QUESTION_TEMPLATE = """
    'Q: [QUESTION]\n\n'
    'Please answer based on the video content using the specified XML-style format.'
"""

def convert_example(example):
    messages = []
    messages.append({"role": "system", "content": SYSTEM_PROMPT})
    example_prompt = QUESTION_TEMPLATE.replace("[QUESTION]", example["problem"]["question"])
    messages.append({
        "role": "user",
        "content": [
            {"type": "text", "text": example_prompt},
            {"type": "video",
             "video": example["video_path"],
             "total_pixels": 512 * 28 * 28,
             "min_pixels": 4 * 28 * 28}
        ]
    })
    answer_text = f'<answer>{example["answer"]}</answer>'
    messages.append({"role": "assistant", "content": answer_text})
    example["messages"] = messages
    return example

# -------------------------------
# 关键修改：Omni 的 collate_fn
# -------------------------------
processor: Qwen2_5OmniProcessor | AutoTokenizer | None = None

def collate_fn(examples):
    conversations = [convert_example(examples[0])["messages"]]

    batch_inputs = processor.apply_chat_template(
        conversations,
        load_audio_from_video=True,    
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        video_fps=1,
        padding=True,
        use_audio_in_video=True,       
    )

    labels = batch_inputs["input_ids"].clone()
    pad_id = processor.tokenizer.pad_token_id
    labels[labels == pad_id] = -100
    
    if hasattr(processor, "video_token"):
        vid_id = processor.tokenizer.convert_tokens_to_ids(processor.video_token)
        labels[labels == vid_id] = -100
    if hasattr(processor, "audio_token"):
        aud_id = processor.tokenizer.convert_tokens_to_ids(processor.audio_token)
        labels[labels == aud_id] = -100

    batch_inputs["labels"] = labels
    return batch_inputs

# -------------------------------
# main function
# -------------------------------
def main(script_args, training_args, model_args):
    set_seed(training_args.seed)

    # logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint and training_args.resume_from_checkpoint is None:
            logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")

    dataset = zl_load_json_dataset(
        script_args.train_data_path,
        script_args.eval_data_path,
        script_args.video_folder,
        script_args.preprocessed_data_path
    )

    global processor
    processor = Qwen2_5OmniProcessor.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code
    )
    logger.info("Using Qwen2_5OmniProcessor for multimodal (video/audio/image/text).")

    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    load_thinker_only = True

    logger.info("*** Initializing model kwargs ***")
    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    quantization_config = get_quantization_config(model_args)

    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )

    if load_thinker_only:
        model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            **model_kwargs
        )
        logger.info("Loaded Qwen2_5OmniThinkerForConditionalGeneration (text-only).")
    else:
        model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            **model_kwargs
        )
        logger.info("Loaded Qwen2_5OmniForConditionalGeneration (text + speech).")

    peft_config = get_peft_config(model_args)

    # -------- SFTTrainer --------
    training_args.dataset_kwargs = {"skip_prepare_dataset": True}
    training_args.remove_unused_columns = False

    if getattr(training_args, "eos_token", None) is None:
        try:
            training_args.eos_token = processor.tokenizer.eos_token
        except Exception:
            pass

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["eval"] if training_args.eval_strategy != "no" else None,
        processing_class=processor,         
        data_collator=collate_fn,
        peft_config=peft_config,
    )

    # train
    logger.info("*** Train ***")
    checkpoint = training_args.resume_from_checkpoint or last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)

    metrics = train_result.metrics
    metrics["train_samples"] = len(dataset["train"])
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    # save
    logger.info("*** Save model ***")
    trainer.save_model(training_args.output_dir)
    processor.save_pretrained(training_args.output_dir)
    logger.info(f"Model & processor saved to {training_args.output_dir}")

    if trainer.accelerator.is_main_process:
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)
        kwargs = {"dataset_name": getattr(script_args, "dataset_name", "custom-video-qa"), "tags": ["SFT", "Qwen2.5-Omni"]}
        trainer.create_model_card(**kwargs)

    if training_args.push_to_hub:
        logger.info("Pushing to hub...")
        trainer.push_to_hub(**kwargs)
        processor.push_to_hub(training_args.hub_model_id)

if __name__ == "__main__":
    parser = TrlParser((SFTScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)