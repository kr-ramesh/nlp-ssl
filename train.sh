#!/bin/bash

#SBATCH --job-name="train-classifier"
#SBATCH --time=12:00:00
#SBATCH --account=a100acct
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-a100
#SBATCH --output=test.txt

export HF_DATASETS_CACHE="/export/fs06/kramesh3/nlp-hw/.cache"

python ft-models.py \
            --model_name "roberta-base" \
            --dataset_name "sst2" \
            --path_to_model "/export/fs06/kramesh3/nlp-hw/base-model-sst2" \
            --is_train \
            --n_labels 2

python ft-models.py \
            --model_name "roberta-base" \
            --dataset_name "sst2" \
            --path_to_model "/export/fs06/kramesh3/nlp-hw/lora-model-sst2" \
            --is_train \
            --n_labels 2 \
            --lora

python ft-models.py \
            --model_name "roberta-base" \
            --dataset_name "sst2" \
            --path_to_model "/export/fs06/kramesh3/nlp-hw/bitfit-model-sst2" \
            --is_train \
            --n_labels 2 \
            --bitfit