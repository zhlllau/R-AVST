cd src/r1-v

export DEBUG_MODE="true" 
export LOG_PATH="./debug_log-r1.txt"
export WANDB_NAME=TEST
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=DETAIL

export WANDB_MODE=disabled
export LOGLEVEL=INFO

CUDA_VISIBLE_DEVICES=1,2,3,0 torchrun --nproc_per_node="4" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12360" \
    src/open_r1/grpo.py \
    --output_dir ./log_SFT_GRPO_THINK/$WANDB_NAME \
    --model_name_or_path Qwen/Qwen2.5-Omni-7B \
    --dataset_name ./Annotations/xml_train_QA.json \
    --deepspeed ./local_scripts/zero3_offload.json \
    --max_prompt_length 2048 \
    --max_completion_length 2048 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate 1e-6 \
    --lr_scheduler_type "cosine" \
    --weight_decay 0.01 \
    --bf16 \
    --logging_steps 1 \
    --gradient_checkpointing true \
    --len_control false \
    --weighted_reward false\
    --attn_implementation flash_attention_2 \
    --max_pixels 200704 \
    --num_train_epochs 1 \
    --run_name $WANDB_NAME \
    --save_steps 20 \
    --beta 0.001 \
    --max_grad_norm 5 \
    --save_only_model false \
    --num_generations 6 \
    --model_type omni \
    --use_audio_in_video true \
    --reward_funcs "spatial-temporal" "format" \
    --train_data_path  ./Annotations/xml_train_QA.json \
    --eval_data_path ./Annotations/xml_test_QA.json \
    --video_folder /zl/Videos   # You need to change this file