#!/bin/bash
nohup python -u run.py \
    --test_file ./data/booking.jsonl\
    --api_key "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" \
    --max_iter 15 \
    --max_attached_imgs 3 \
    --temperature 1 \
    --fix_box_color \
    --window_width 1920 \
    --window_height 1080 \
    --seed 42 > test_tasks.log &
    