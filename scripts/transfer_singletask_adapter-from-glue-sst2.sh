cd ../src

TASKS="rotten_tomatoes yelp_polarity amazon_polarity"
DATA_DIR=../data
TUNE_METHOD=PET_mc_stage2
ADAPTER_SIZE=12
LORA_SIZE=10
PREFIX_R=24
PREFIX_NUM=120
LOW_DIMENSION=4
SAVE_PATH=../models
IDENTIFIER=full_data_PET_mc_stage2
PRETRAINED_MODEL_PATH=../pretrained_models
SOURCE_TASK=glue-sst2
PET_NAME=adapter
GPU=3

for TASK in $TASKS
do

echo "Task: $TASK, Identifier: $IDENTIFIER"

CUDA_VISIBLE_DEVICES=${GPU} \
MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python tune_hps_singletask_PET_mode_connectivity_stage2.py \
--task_dir ${DATA_DIR}/${TASK} \
--do_train \
--do_predict \
--learning_rate_list 1e-2 5e-2 \
--bsz_list 8 \
--train_iters 5000 \
--model ${PRETRAINED_MODEL_PATH}/t5.1.1.lm100k.base \
--tokenizer_path ${PRETRAINED_MODEL_PATH}/t5-v1_1-base \
--output_dir ${SAVE_PATH}/${IDENTIFIER}/${TASK}-${PET_NAME}-r_${LOW_DIMENSION} \
--predict_batch_size 40 \
--one_prefix \
--tune_method ${TUNE_METHOD} \
--valid_interval 500 \
--output_interval 10000 \
--log_interval 100 \
--early_stop -1 \
--quiet \
--apply_adapter \
--adapter_type houlsby \
--adapter_size ${ADAPTER_SIZE} \
--load_PET_enc_dec_path ../models/full_data_PET_mc/${SOURCE_TASK}-r_4/lr_0.0001_bsz_8_seed_42/checkpoint-best.pt \
--low_dimension ${LOW_DIMENSION} \

done