export TRANSFORMERS_CACHE=/hdd0/zl/huggingface
export HF_HOME=/hdd0/zl/huggingface
export MODELSCOPE_CACHE=/hdd0/zl/modelscope_cache
export USE_AUDIO_IN_VIDEO=true

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
nproc_per_node=1
CUDA_VISIBLE_DEVICES="7" \
NPROC_PER_NODE=$nproc_per_node \
VIDEO_MAX_PIXELS=21952 \
FPS_MAX_FRAMES=6 \
MAX_PIXELS=21952 \
ENABLE_AUDIO_OUTPUT=0 \
swift sft \
    --model Qwen/Qwen2.5-Omni-7B \
    --dataset '/R-AVST_code/Annotations/xml_train_QA.json' \  # You need to change this
    --split_dataset_ratio 0.01 \
    --train_type lora \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-4 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --freeze_vit true \
    --gradient_accumulation_steps 8 \
    --eval_steps 20 \
    --save_steps 20 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --max_length 8192 \
    --output_dir output \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --deepspeed zero3 
    