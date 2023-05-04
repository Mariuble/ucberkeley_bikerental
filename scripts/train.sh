#!/bin/bash

mkdir -p ../models

python train.py \
    --data_path=../data/train.csv \
    --model_path=../models/gbr.joblib