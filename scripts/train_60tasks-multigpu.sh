cd ../src

DATA_DIR=../data
TUNE_METHOD=PET_mc
ADAPTER_SIZE=12
LORA_SIZE=10
PREFIX_R=24
PREFIX_NUM=120
LOW_DIMENSION=100
SAVE_PATH=../models
IDENTIFIER=full_data_PET_mc_multitask_multigpu
PRETRAINED_MODEL_PATH=../pretrained_models
TASK_SPLIT=train_60_unseen_rest
NPROC_PER_NODE=4
GPU=1,2,3,5


echo "Task: $TASK, Identifier: $IDENTIFIER"

CUDA_VISIBLE_DEVICES=${GPU} \
MKL_NUM_THREADS=4 OMP_NUM_THREADS=1 \
python -m torch.distributed.launch --master_port 88888 --nproc_per_node ${NPROC_PER_NODE} tune_hps_multitask_PET_mode_connectivity_multigpu.py \
--task_dir ${DATA_DIR} \
--do_train \
--do_predict \
--custom_tasks_splits dataloader/custom_tasks_splits_PET_mc/${TASK_SPLIT}.json \
--learning_rate_list 1e-4 \
--bsz_list 4 \
--train_epochs 1 \
--train_lines_each_task 20000 \
--dev_lines_each_task 240 \
--model ${PRETRAINED_MODEL_PATH}/t5.1.1.lm100k.base \
--tokenizer_path ${PRETRAINED_MODEL_PATH}/t5-v1_1-base \
--output_dir ${SAVE_PATH}/${IDENTIFIER}/${DATA_DIR}_${TASK_SPLIT}_${NPROC_PER_NODE}gpu-lr_1e-4-delta_R-train_2w_lines-recon_alpha_10 \
--predict_batch_size 40 \
--one_prefix \
--tune_method ${TUNE_METHOD} \
--valid_interval 1000 \
--output_interval 10000000 \
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
--low_dimension ${LOW_DIMENSION} \
--load_PET_dir ../models \
--gpu_num ${NPROC_PER_NODE} \
--reconstruct_alpha 10 \
