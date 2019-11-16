python bert/create_pretraining_data.py^
  --input_file=D:/works/zalo_challenges/qna_data/wiki_data/domain.txt^
  --output_file=D:/works/zalo_challenges/qna_data/wiki_data/domain.tfrecord^
  --vocab_file=D:/works/zalo_challenges/models/bert/multi_cased_L-12_H-768_A-12/vocab.txt^
  --do_lower_case=False^
  --max_seq_length=512^
  --max_predictions_per_seq=76^
  --masked_lm_prob=0.15^
  --random_seed=12345^
  --dupe_factor=5
