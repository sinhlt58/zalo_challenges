@echo off

set GLUE_DIR=./qna_data/glue_data/en
set TASK_NAME=qnli
set MODEL_TYPE=roberta
set MODEL_NAME=roberta_base

python run_glue.py^
    --model_type %MODEL_TYPE%^
    --model_name_or_path models/%MODEL_NAME%^
    --task_name %TASK_NAME%^
    --do_test^
    --data_dir %GLUE_DIR%^
    --test_file_name test^
    --dev_file_name dev^
    --max_seq_length 300^
    --per_gpu_eval_batch_size=8^
    --per_gpu_train_batch_size=8^
    --learning_rate 2e-5^
    --num_train_epochs 3.0^
    --output_dir models/%MODEL_NAME%
