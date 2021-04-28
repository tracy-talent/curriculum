CUDA_VISIBLE_DEVICES=0 \
python main.py \
    --dataset renmin1998 \
    --compress_seq \
    --tagscheme bmoe \
    --only_test \
    --use_lstm \
    --use_crf \
    --embedding_size 300 \
    --hidden_size 300 \
    --batch_size 64 \
    --dropout_rate 0.1 \
    --lr 1e-3 \
    --max_length 100 \
    --max_epoch 20 \
    --warmup_epoches 0 \
    --optimizer adam \
    --metric micro_f1 \
    # --only_test \
