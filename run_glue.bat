@echo off

set GLUE_DIR=./qna_data/glue_data/vi
set TASK_NAME=qnli
set MODEL_TYPE=bert
set MODEL_NAME=bert-base-multilingual-cased-domain

python run_glue.py^
    --model_type %MODEL_TYPE%^
    --model_name_or_path models/%MODEL_NAME%^
    --task_name %TASK_NAME%^
    --do_eval^
    --do_test^
    --data_dir %GLUE_DIR%^
    --test_file_name test^
    --test_pids_file_name test_pids^
    --dev_file_name dev^
    --dev_pids_file_name dev_pids^
    --max_seq_length 512^
    --per_gpu_eval_batch_size=8^
    --per_gpu_train_batch_size=8^
    --learning_rate 2e-5^
    --num_train_epochs 3.0^
    --output_dir models/%MODEL_NAME%
