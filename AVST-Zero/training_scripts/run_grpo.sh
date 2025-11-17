# export WANDB_PROJECT=Video-GRPO
export WANDB_NAME=$(basename $0)_$(date +"%Y%m%d_%H%M%S")

export PYTHONPATH=".:$PYTHONPATH"
OUTDIR=./checkpoints/$WANDB_NAME

export DEBUG_MODE="true"
export LOG_PATH="./logs/${WANDB_NAME}.log"

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=DETAIL

MASTER_PORT=$((10000 + RANDOM % 2000))

export CUDA_VISIBLE_DEVICES=2,3,4,5

torchrun --nproc_per_node="4" \
    --nnodes="1" \
    --node_rank="0"\
    --master_addr="127.0.0.1" \
    --master_port=$MASTER_PORT  \
    ./src/open_r1/grpo.py \
    --deepspeed ./training_scripts/zero3_offload.json \
    --output_dir $OUTDIR \
    --model_name_or_path ./models/Qwen2.5-VL-7B-Instruct \
    --train_data_path  ./Annotations/xml_train_QA.json \
    --eval_data_path ./Annotations/xml_test_QA.json \
    --video_folder /zl/Videos \  # You need to change to your video folder path
    --dataset_name R_AVST \
    --max_prompt_length 512 \
    --max_completion_length 1024 \
    --num_generations 6 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --logging_steps 1 \
    --bf16 \
    --torch_dtype bfloat16 \
    --data_seed 42 \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --num_train_epochs 1 \
    --run_name $WANDB_NAME \
    --report_to tensorboard \
    --save_steps 50 \
    --save_total_limit 1 \
    --save_only_model false \
    --reward_funcs "spatial-temporal" "format"