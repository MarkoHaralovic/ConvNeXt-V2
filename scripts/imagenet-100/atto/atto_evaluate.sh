#!/bin/bash

#environment variables
RUN_NAME="evaluation"
MODEL_NAME="atto"
DATASET_NAME="imagenet100"
NUM_GPU=1

# ConvNeXt parameters
BATCH_SIZE=32

# EMA related parameters
MODEL_EMA=False
MODEL_EMA_DECAY=0.9999
MODEL_EMA_FORCE_CPU=False
MODEL_EMA_EVAL=False

# Optimization parameters
# CLIP_GRAD=''
WEIGHT_DECAY=0.3
BLR=0.00015 
LAYER_DECAY=0.9
MIN_LR=0.000001
WARMUP_EPOCHS=0

WARMUP_STEPS=-1
OPT='adamw'
OPT_EPS=0.00000001
# OPT_BETAS=''
MOMENTUM=0.9
# WEIGHT_DECAY_END=''

#model parameters
MODEL="convnextv2_atto"
INPUT_SIZE=64
DROP_PATH=0.0
LAYER_DECAY_TYPE='single' 

#dataset parameters
DATA_PATH="/imagenet100"
NB_CLASSES=100
OUTPUT_DIR="/ConvNeXt-V2/log_dir/${RUN_NAME}_${MODEL_NAME}_${DATASET_NAME}_bs${BATCH_SIZE}_ep${EPOCHS}_inputsize${INPUT_SIZE}"
LOG_DIR="/ConvNeXt-V2/log_dir/${RUN_NAME}_${MODEL_NAME}_${DATASET_NAME}_bs${BATCH_SIZE}_ep${EPOCHS}_inputsize${INPUT_SIZE}/tensorboard_log.txt"
DEVICE="cuda"
SEED=0
RESUME='log_dir/checkpoint-183.pth'

IMAGENET_DEFAULT_MEAN_AND_STD=True
DATA_SET='IMAGENET100'
EVAL=True
NUM_WORKERS=4
PIN_MEM=False

SCRIPT_DIR="$(dirname "$(realpath "$0")")"
PYTHON_SCRIPT="$SCRIPT_DIR/../../../main_finetune.py"

python -m torch.distributed.launch --nproc_per_node="$NUM_GPU" "$PYTHON_SCRIPT" \
 --batch_size "$BATCH_SIZE" \
 --model "$MODEL"  --input_size "$INPUT_SIZE"\
 --data_path "$DATA_PATH" --nb_classes "$NB_CLASSES"  --output_dir "$OUTPUT_DIR"  --log_dir "$LOG_DIR" \
 --device "$DEVICE"  --seed "$SEED"  --resume "$RESUME"  \
 --imagenet_default_mean_and_std "$IMAGENET_DEFAULT_MEAN_AND_STD"  --data_set "$DATA_SET" \
 --num_workers "$NUM_WORKERS" --pin_mem "$PIN_MEM" --eval "$EVAL"\
 --model_ema "$MODEL_EMA" --model_ema_decay "$MODEL_EMA_DECAY" --model_ema_force_cpu "$MODEL_EMA_FORCE_CPU" --model_ema_eval "$MODEL_EMA_EVAL" \
 --weight_decay "$WEIGHT_DECAY" --blr "$BLR"  --layer_decay "$LAYER_DECAY" --min_lr "$MIN_LR" --warmup_epochs "$WARMUP_EPOCHS"  \
 --warmup_steps "$WARMUP_STEPS"  --opt "$OPT"  --opt_eps "$OPT_EPS"  --momentum "$MOMENTUM"
