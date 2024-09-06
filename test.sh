export HF_DATASETS_CACHE="/export/fs06/kramesh3/nlp-hw/.cache"

python ft-models.py \
            --model_name "roberta-base" \
            --dataset_name "sst2" \
            --path_to_model "/export/fs06/kramesh3/nlp-hw/lora-model-sst2-final" \
            --is_test \
            --n_labels 2 \
            --lora

python ft-models.py \
            --model_name "roberta-base" \
            --dataset_name "sst2" \
            --path_to_model "/export/fs06/kramesh3/nlp-hw/bitfit-model-sst2-final" \
            --is_test \
            --n_labels 2 \
            --bitfit

python ft-models.py \
            --model_name "roberta-base" \
            --dataset_name "sst2" \
            --path_to_model "/export/fs06/kramesh3/nlp-hw/base-model-sst2-final" \
            --is_test \
            --n_labels 2