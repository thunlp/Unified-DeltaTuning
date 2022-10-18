cd ../src

TASKS="ai2_arc blimp-anaphor_number_agreement eli5-asks ethos-gender qasc quartz-no_knowledge rotten_tomatoes superglue-wsc yelp_polarity"
DATA_DIR="data"
TUNE_METHOD=PET_mc_stage2
ADAPTER_SIZE=12
LORA_SIZE=10
PREFIX_R=24
PREFIX_NUM=120
LOW_DIMENSION=100
SAVE_PATH=models
IDENTIFIER=full_data_PET_mc_multitask_stage2
PRETRAINED_MODEL_PATH=pretrained_models
PET_NAME=adapter
GPU=0

for TASK in $TASKS
do

echo "Task: $TASK, Identifier: $IDENTIFIER"

CUDA_VISIBLE_DEVICES=${GPU} \
MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python tune_hps_singletask_PET_mode_connectivity_stage2.py \
--task_dir data_full_new/${TASK} \
--do_train \
--do_predict \
--learning_rate_list 1e-2 5e-2 \
--bsz_list 8 \
--train_iters 5000 \
--model ${PRETRAINED_MODEL_PATH}/t5.1.1.lm100k.base \
--tokenizer_path ${PRETRAINED_MODEL_PATH}/t5-v1_1-base \
--output_dir ${SAVE_PATH}/${IDENTIFIER}/${TASK}-${PET_NAME}-r_${LOW_DIMENSION}_multitask-${DATA_DIR}-best-from_train_60 \
--predict_batch_size 80 \
--one_prefix \
--tune_method ${TUNE_METHOD} \
--valid_interval 100 \
--output_interval 100 \
--log_interval 100 \
--early_stop -1 \
--quiet \
--apply_adapter \
--adapter_type houlsby \
--adapter_size ${ADAPTER_SIZE} \
--load_PET_enc_dec_path models/full_data_PET_mc_multitask_multigpu/data_full_new_train_60_unseen_rest_4gpu-lr_1e-4-delta_R-train_2w_lines-new/lr_0.0001_bsz_4_seed_42/checkpoint-best.pt \
--low_dimension ${LOW_DIMENSION} \
--choose_valid \
--choose_valid_lines 1000 \

done
