#!/bin/bash

#environment variables
RUN_NAME="finetune"
MODEL_NAME="tiny"
DATASET_NAME="imagenet100"
NUM_GPU=1

# ConvNeXt parameters
BATCH_SIZE=128
EPOCHS=300
UPDATE_FREQ=1

#model parameters
MODEL="convnextv2_tiny"
INPUT_SIZE=224
MASK_RATIO=0.6
DROP_PATH=0.0
LAYER_DECAY_TYPE='single' 

# EMA related parameters
MODEL_EMA=False
MODEL_EMA_DECAY=0.9999
MODEL_EMA_FORCE_CPU=False
MODEL_EMA_EVAL=False

# Optimization parameters
# CLIP_GRAD=''
WEIGHT_DECAY=0.05
BLR=0.0008 
LAYER_DECAY=0.9
MIN_LR=0.000001
WARMUP_EPOCHS=40

WARMUP_STEPS=-1
OPT='adamw'
OPT_EPS=0.00000001
# OPT_BETAS=''
MOMENTUM=0.9
# WEIGHT_DECAY_END=''

# Augmentation parameters
# COLOR_JITTER=''
AA='rand-m9-mstd0.5-inc1'
SMOOTHING=0.1
TRAIN_INTERPOLATION='bicubic'

# * Random Erase params
REPROB=0.25
REMODE='pixel'
RECOUNT=1
RESPLIT=False

# * Mixup params
MIXUP=0.8
CUTMIX=1.0
# CUTMIX_MINMAX=''
MIXUP_PROB=1.0
MIXUP_SWITCH_PROB=0.5
MIXUP_MODE='batch'

# * Finetuning params
FINETUNE=''
HEAD_INIT_SCALE=0.001
MODEL_KEY='model|module'
MODEL_PREFIX=''

#dataset parameters
DATA_PATH="/imagenet100"
NB_CLASSES=200
OUTPUT_DIR="/ConvNeXt-V2/log_dir/${RUN_NAME}_${MODEL_NAME}_${DATASET_NAME}_bs${BATCH_SIZE}_ep${EPOCHS}_inputsize${INPUT_SIZE}"
LOG_DIR="/ConvNeXt-V2/log_dir/${RUN_NAME}_${MODEL_NAME}_${DATASET_NAME}_bs${BATCH_SIZE}_ep${EPOCHS}_inputsize${INPUT_SIZE}/tensorboard_log.txt"
DEVICE="cuda"
SEED=0
RESUME=''

EVAL_DATA_PATH=''
IMAGENET_DEFAULT_MEAN_AND_STD=True
DATA_SET='IMNET'
AUTO_RESUME=True
SAVE_CKPT=True
SAVE_CKPT_FREQ=1
SAVE_CKPT_NUM=3

START_EPOCH=0
EVAL=False
DIST_EVAL=True
DISABLE_EVAL=False
NUM_WORKERS=4
PIN_MEM=False
CONVERT_TO_FFCV=False 
BETON_PATH="/imagenet100/imagenet100.beton"
VALIDATION_BETON_PATH="/imagenet1k/imagenet1k_validation.beton"

# distributed training parameters
WORLD_SIZE=1
LOCAL_RANK=1
DIST_ON_ITP=False
DIST_URL='env://'
FIND_UNUSED_PARAMETERS=False
USE_AMP=False

SCRIPT_DIR="$(dirname "$(realpath "$0")")"
PYTHON_SCRIPT="$SCRIPT_DIR/../../../main_finetune.py"

mkdir -p "$OUTPUT_DIR"
touch "$OUTPUT_DIR/config.txt"
cp "$0" "$OUTPUT_DIR/config.txt"

python -m torch.distributed.launch --nproc_per_node="$NUM_GPU" "$PYTHON_SCRIPT" \
 --batch_size "$BATCH_SIZE"  --epochs  "$EPOCHS"  --update_freq  "$UPDATE_FREQ"  \
 --model "$MODEL"  --input_size "$INPUT_SIZE"  --mask_ratio "$MASK_RATIO"  --drop_path "$DROP_PATH"  --layer_decay_type "$LAYER_DECAY_TYPE"  \
 --model_ema "$MODEL_EMA" --model_ema_decay "$MODEL_EMA_DECAY" --model_ema_force_cpu "$MODEL_EMA_FORCE_CPU" --model_ema_eval "$MODEL_EMA_EVAL" \
 --weight_decay "$WEIGHT_DECAY"  --lr "$LR"  --blr "$BLR"  --layer_decay "$LAYER_DECAY" --min_lr "$MIN_LR" --warmup_epochs "$WARMUP_EPOCHS"  \
 --warmup_steps "$WARMUP_STEPS"  --opt "$OPT"  --opt_eps "$OPT_EPS"  --momentum "$MOMENTUM"\
 --aa "$AA"  --smoothing "$SMOOTHING"  --train_interpolation "$TRAIN_INTERPOLATION"\
 --reprob "$REPROB"  --remode "$REMODE"  --recount "$RECOUNT" --resplit "$RESPLIT"\
 --mixup "$MIXUP"  --cutmix "$CUTMIX"  --mixup_prob "$MIXUP_PROB" --mixup_switch_prob "$MIXUP_SWITCH_PROB" --mixup_mode "$MIXUP_MODE"\
 --finetune "$FINETUNE"  --head_init_scale "$HEAD_INIT_SCALE"  --model_key "$MODEL_KEY" --model_prefix "$MODEL_PREFIX"\
 --data_path "$DATA_PATH" --nb_classes "$NB_CLASSES"  --output_dir "$OUTPUT_DIR"  --log_dir "$LOG_DIR"  --device "$DEVICE"  --seed "$SEED"  --resume "$RESUME"  \
 --eval_data_path "$EVAL_DATA_PATH" --imagenet_default_mean_and_std "$IMAGENET_DEFAULT_MEAN_AND_STD"  --data_set "$DATA_SET" \
 --auto_resume "$AUTO_RESUME"  --save_ckpt "$SAVE_CKPT" --save_ckpt_freq "$SAVE_CKPT_FREQ" --save_ckpt_num "$SAVE_CKPT_NUM"\
 --start_epoch "$START_EPOCH" --eval "$EVAL" --dist_eval "$DIST_EVAL" --disable_eval "$DISABLE_EVAL" --num_workers "$NUM_WORKERS" --pin_mem "$PIN_MEM" \
 --convert_to_ffcv "$CONVERT_TO_FFCV" --beton_path "$BETON_PATH" --validation_beton_path "$VALIDATION_BETON_PATH" \
 --world_size  "$WORLD_SIZE" --local_rank  "$LOCAL_RANK" --dist_on_itp  "$DIST_ON_ITP" --dist_url  "$DIST_URL"   --find_unused_parameters "$FIND_UNUSED_PARAMETERS"  --use_amp "$USE_AMP" 
