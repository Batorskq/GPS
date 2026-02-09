export DATASET=gen
export NUMBER_OF_SAMPLES=5
export NUMBER_OF_PROMPTS=10
export ADVERSARIAL=0
export REASONING=True
export AUX=False
export SAMPLINGPROB=0.2
export LLMREGUL=False
export PSP=True
export USE_BOTH_REG=False

CUDA_VISIBLE_DEVICES=0,1 \
NPROC_PER_NODE=1 \
MASTER_PORT=19506 \
swift rlhf \
    --rlhf_type grpo \
    --model Qwen/Qwen2.5-7B-Instruct \
    --model_type qwen2_5 \
    --dataset datasets/original/gen_train.jsonl  \
    --val_dataset datasets/original/gen_train.jsonl \
    --reward_funcs accuracy format \
    --torch_dtype bfloat16 \
    --gradient_checkpointing_kwargs '{"use_reentrant": false}' \
    --use_lmdeploy false \
    --train_type lora \
    --lora_rank 8 \
    --lora_alpha 32 \
    --seed 5 \
    --max_completion_length 1024 \
    --num_train_epochs 1000 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --learning_rate 5e-5 \
    --gradient_accumulation_steps 1 \
    --eval_steps 1000 \
    --save_steps 500 \
    --save_total_limit 20 \
    --max_length 2048 \
    --output_dir output \
    --warmup_ratio 0 \
    --dataloader_num_workers 1 \
    --dataset_num_proc 4 \
    --num_generations 4 \
    --temperature 0.9 \
    --report_to wandb \
    --logging_steps 5 \
    --system 'examples/train/grpo/prompt.txt' \
    --log_completions true \
    --num_iterations 1 \
    --num_infer_workers 1
