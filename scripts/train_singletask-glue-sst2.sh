cd ../src

TASKS="glue-sst2"
DATA_DIR=data
TUNE_METHOD=PET_mc
ADAPTER_SIZE=12
LORA_SIZE=10
PREFIX_R=24
PREFIX_NUM=120
LOW_DIMENSION=4
SAVE_PATH=../models
IDENTIFIER=full_data_PET_mc_singleTask
PRETRAINED_MODEL_PATH=../pretrained_models
GPU=3
ALPHA=0.1

for TASK in $TASKS
do

echo "Task: $TASK, Identifier: $IDENTIFIER"

CUDA_VISIBLE_DEVICES=${GPU} \
MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python tune_hps_singletask_PET_mode_connectivity.py \
--task_dir ${DATA_DIR}/${TASK} \
--do_train \
--do_predict \
--learning_rate_list 1e-4 \
--bsz_list 8 \
--train_iters 100000 \
--model ${PRETRAINED_MODEL_PATH}/t5.1.1.lm100k.base \
--tokenizer_path ${PRETRAINED_MODEL_PATH}/t5-v1_1-base \
--output_dir ${SAVE_PATH}/${IDENTIFIER}/${TASK}-r_${LOW_DIMENSION}-recon_alpha_${ALPHA} \
--predict_batch_size 40 \
--one_prefix \
--tune_method ${TUNE_METHOD} \
--valid_interval 1000 \
--output_interval 1000000 \
--log_interval 100 \
--early_stop -1 \
--quiet \
--apply_adapter \
--adapter_type houlsby \
--adapter_size ${ADAPTER_SIZE} \
--apply_lora \
--lora_alpha 16 \
--lora_r ${LORA_SIZE} \
--apply_prefix \
--prefix_r ${PREFIX_R} \
--prefix_num ${PREFIX_NUM} \
--load_stage1_adapter_path_list ../models/full_data_adapter/${TASK}-adapter_size_12-seed_42/lr_0.0005_bsz_16_seed_42/checkpoint-best.pt \
--load_stage1_lora_path_list ../models/full_data_lora/${TASK}-lora_size_10-seed_42/lr_0.0005_bsz_16_seed_42/checkpoint-best.pt \
--load_stage1_prefix_path_list ../models/full_data_prefix/${TASK}-r_24-num_120-SGD_noise_seed_42/lr_0.0005_bsz_16_seed_42/checkpoint-best.pt \
--low_dimension ${LOW_DIMENSION} \
--reconstruct_alpha ${ALPHA} \
--choose_valid \
--choose_valid_lines 1000 \

done
