#! /usr/bin/env bash
CUDA_VISIBLE_DEVICES=$1 \
python run_smooth.py \
  --data_dir '../input' \
  --bert_model_dir '/home/liujian/NLP/corpus/chinese_L-12_H-768_A-12' \
  --output_dir '../output/models' \
  --max_seq_length 300 \
  --do_train \
  --do_lower_case \
  --train_batch_size 60 \
  --gradient_accumulation_steps 3 \
  --predict_batch_size 15 \
  --learning_rate 2e-5 \
  --num_train_epochs 3
