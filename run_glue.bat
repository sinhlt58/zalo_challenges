@echo off

set GLUE_DIR=./glue_data
set TASK_NAME=qnli

python ./transformers-repo/examples/run_glue.py^
    --model_type roberta^
    --model_name_or_path roberta-large^
    --cache_dir models/roberta-large^
    --task_name %TASK_NAME%^
    --do_train^
    --do_eval^
    --data_dir %GLUE_DIR%/%TASK_NAME%^
    --max_seq_length 300^
    --per_gpu_eval_batch_size=8^
    --per_gpu_train_batch_size=8^
    --learning_rate 2e-5^
    --num_train_epochs 3.0^
    --output_dir ./tmp/%TASK_NAME%/
