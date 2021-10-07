#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0 python train.py \
    data-bin-joined-dict/iwslt14.tokenized.de-en \
    --task translation_thor --user-dir thor \
    --num-experts 2 --consistency-alpha 5.0 --inference-level 1 \
    --arch thor_transformer_iwslt_de_en --share-all-embeddings \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 0.001 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 4096 \
    --log-format json --log-interval 10 \
    --num-workers 10 \
    --update-freq 1 \
    --ddp-backend no_c10d \
    --fp16 \
    --seed 1 \
