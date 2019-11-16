@echo off

set BERT_MODEL=bert-base-multilingual-cased
set BERT_MODEL_FOLDER=models/%BERT_MODEL%
set BERT_MODEL_FOLDER_TORCH=models/%BERT_MODEL%

python convert_bert_original_tf_checkpoint_to_pytorch.py^
    --tf_checkpoint_path %BERT_MODEL_FOLDER%/bert_model.ckpt^
    --bert_config_file %BERT_MODEL_FOLDER%/config.json^
    --pytorch_dump_path %BERT_MODEL_FOLDER_TORCH%/pytorch_model.bin^
