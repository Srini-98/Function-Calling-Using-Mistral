pred_type='function_calling'
model_name='mistral7B'
training_type='glaiveai_function_calling'
dataset_type='glaiveai_function_calling'
batch_size=1
val_batch_size=2
gradient_accumulation_steps=2
num_train_epochs=1
num_gpus= # number of GPU's for DDP training
device= #GPU IDS in your clustser
total_batch_size=$(($batch_size * $gradient_accumulation_steps * $num_gpus))
learning_rate=1e-5
logfile_name= #enter log file to track training
run_name= #enter run name for wandb
data_path=glaiveai/glaive-function-calling-v2
model_path= #enter model path
cache_dir= #enter model path
output_dir= #enter directory to save model

echo '''starting to finetune'''

CUDA_VISIBLE_DEVICES=$device nohup torchrun --nproc_per_node $num_gpus train.py \
    --ddp_find_unused_parameters False  \
    --gradient_accumulation_steps $gradient_accumulation_steps \
    --cache_dir $cache_dir \
    --model_path $model_path \
    --num_train_epochs $num_train_epochs \
    --data_path $data_path \
    --bf16 True \
    --load_best_model_at_end True \
    --output_dir $output_dir \
    --per_device_train_batch_size $batch_size \
    --per_device_eval_batch_size $val_batch_size \
    --evaluation_strategy "steps" \
    --eval_steps 1000 \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 3 \
    --learning_rate $learning_rate \
    --weight_decay 0. \
    --run_name $run_name \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 > ./training_logs/${logfile_name}