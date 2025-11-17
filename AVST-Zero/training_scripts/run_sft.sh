
# export WANDB_PROJECT=Video-GRPO
# export OMP_NUM_THREADS=1
# export DISABLE_ADDMM_CUDA_LT=1
# export TORCH_CUDNN_USE_HEURISTIC_MODE_B=1
# export NCCL_SOCKET_IFNAME=bond0
# export NCCL_DEBUG="INFO"
# export NCCL_IB_HCA=mlx5_0

export HF_ENDPOINT=https://hf-mirror.com
export PATH=/usr/local/cuda-12.6/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:$LD_LIBRARY_PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

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
export CUDA_VISIBLE_DEVICES=4,5,6,7
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

torchrun --nproc_per_node="1" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port=$MASTER_PORT \
    ./src/sft/sft.py \
    --deepspeed ./training_scripts/zero3_offload.json \
    --model_name_or_path ./models/Qwen2.5Omni \
    --preprocessed_data_path ./Annotations/xml_train_QA.json \
    --train_data_path ./Annotations/xml_train_QA.json \
    --eval_data_path ./Annotations/xml_test_QA.json \
    --video_folder /zl/Videos \  # You need to change to your video folder path
    --dataset_name xxx \
    --learning_rate 2.0e-5 \
    --num_train_epochs 1 \
    --max_seq_length 1024 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --gradient_checkpointing \
    --bf16 \
    --torch_dtype bfloat16 \
    --logging_steps 5 \
    --eval_strategy no \
    --report_to tensorboard \
    --output_dir $OUTDIR \
    --save_steps 300 \
    --save_only_model true