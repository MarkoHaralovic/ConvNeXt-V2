#!/bin/bash

#environment variables
RUN_NAME="pretrain"
MODEL_NAME="atto"
DATASET_NAME="tiny_imagenet"
NUM_GPU=1

# ConvNeXt parameters
BATCH_SIZE=8 #1024
EPOCHS=1 #800
WARMUP_EPOCHS=40 
UPDATE_FREQ=1

#model parameters
MODEL="convnextv2_atto"
INPUT_SIZE=64
MASK_RATIO=0.6
NORM_PIX_LOSS=True
DECODER_DEPTH=1
DECODER_EMBED_DIM=512 

#optimizer parameters 
WEIGHT_DECAY=0.05
BLR=0.00015
MIN_LR=0.0

#dataset parameters
DATA_PATH="/tiny_imagenet"
OUTPUT_DIR="/ConvNeXt-V2/log_dir/${RUN_NAME}_${MODEL_NAME}_${DATASET_NAME}_bs${BATCH_SIZE}_ep${EPOCHS}_inputsize${INPUT_SIZE}"
LOG_DIR="/ConvNeXt-V2/log_dir/${RUN_NAME}_${MODEL_NAME}_${DATASET_NAME}_bs${BATCH_SIZE}_ep${EPOCHS}_inputsize${INPUT_SIZE}/tensorboard_log.txt"
DEVICE="cuda"
SEED=0
RESUME=''

AUTO_RESUME=True
SAVE_CKPT=True
SAVE_CKPT_FREQ=1
SAVE_CKPT_NUM=3

START_EPOCH=0
NUM_WORKERS=4
PIN_MEM=False
CONVERT_TO_FFCV=True 
BETON_PATH="/tiny_imagenet/tiny_imagenet.beton"

# distributed training parameters
WORLD_SIZE=1
LOCAL_RANK=1
DIST_ON_ITP=False
DIST_URL='env://'
FIND_UNUSED_PARAMETERS=False

SCRIPT_DIR="$(dirname "$(realpath "$0")")"
PYTHON_SCRIPT="$SCRIPT_DIR/../../../main_pretrain.py"

mkdir -p "$OUTPUT_DIR"
touch "$OUTPUT_DIR/config.txt"
cp "$0" "$OUTPUT_DIR/config.txt"

python -m torch.distributed.launch --nproc_per_node="$NUM_GPU" "$PYTHON_SCRIPT" \
 --batch_size "$BATCH_SIZE"  --epochs  "$EPOCHS" --warmup_epochs "$WARMUP_EPOCHS"  --update_freq  "$UPDATE_FREQ"  \
 --model "$MODEL"  --input_size "$INPUT_SIZE" \
 --data_path "$DATA_PATH"  --output_dir "$OUTPUT_DIR"  --log_dir "$LOG_DIR"  --device "$DEVICE"  --seed "$SEED"  --resume "$RESUME"  \
 --auto_resume "$AUTO_RESUME"  --save_ckpt "$SAVE_CKPT" --save_ckpt_freq "$SAVE_CKPT_FREQ" --save_ckpt_num "$SAVE_CKPT_NUM" \
 --start_epoch "$START_EPOCH" --num_workers "$NUM_WORKERS" --pin_mem "$PIN_MEM" --convert_to_ffcv "$CONVERT_TO_FFCV" --beton_path "$BETON_PATH" \
 --weight_decay "$WEIGHT_DECAY" --blr "$BLR"  --min_lr "$MIN_LR"  \
 --mask_ratio "$MASK_RATIO"  --decoder_depth  "$DECODER_DEPTH" --decoder_embed_dim "$DECODER_EMBED_DIM" \
 --world_size  "$WORLD_SIZE" --local_rank  "$LOCAL_RANK" --dist_on_itp  "$DIST_ON_ITP" --dist_url  "$DIST_URL"   --find_unused_parameters "$FIND_UNUSED_PARAMETERS" 
