#!/bin/bash

MODEL_PATH=xxx.pth
python scripts/evaluate_pclmodel.py NoBRS \
    --gpus=2 \
    --checkpoint=${MODEL_PATH} \
    --datasets=xxx \
    --cf-n=0 \
    --acf \
    --iou-analysis \
    --save-ious \
    --print-ious
