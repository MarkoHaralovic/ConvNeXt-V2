#!/bin/bash

#environment variables
RUN_NAME="evaluation"
MODEL_NAME="atto"
DATASET_NAME="imagenet1k"
NUM_GPU=1

# ConvNeXt parameters
BATCH_SIZE=256

#model parameters
MODEL="convnextv2_atto"
INPUT_SIZE=224

#dataset parameters
DATA_PATH="/imagenet100"
NB_CLASSES=1000
OUTPUT_DIR="/ConvNeXt-V2/log_dir/${RUN_NAME}_${MODEL_NAME}_${DATASET_NAME}_bs${BATCH_SIZE}_ep${EPOCHS}_inputsize${INPUT_SIZE}"
LOG_DIR="/ConvNeXt-V2/log_dir/${RUN_NAME}_${MODEL_NAME}_${DATASET_NAME}_bs${BATCH_SIZE}_ep${EPOCHS}_inputsize${INPUT_SIZE}/tensorboard_log.txt"
DEVICE="cuda"
SEED=0
RESUME='pretrained_weights/convnextv2_atto_1k_224_ema.pt'

IMAGENET_DEFAULT_MEAN_AND_STD=True
DATA_SET='IMAGENET100'
EVAL=True
NUM_WORKERS=4
PIN_MEM=False

SCRIPT_DIR="$(dirname "$(realpath "$0")")"
PYTHON_SCRIPT="$SCRIPT_DIR/../../../main_finetune.py"

mkdir -p "$OUTPUT_DIR"
touch "$OUTPUT_DIR/config.txt"
cp "$0" "$OUTPUT_DIR/config.txt"

python -m torch.distributed.launch --nproc_per_node="$NUM_GPU" "$PYTHON_SCRIPT" \
 --batch_size "$BATCH_SIZE" \
 --model "$MODEL"  --input_size "$INPUT_SIZE"\
 --data_path "$DATA_PATH" --nb_classes "$NB_CLASSES"  --output_dir "$OUTPUT_DIR"  --log_dir "$LOG_DIR" \
 --device "$DEVICE"  --seed "$SEED"  --resume "$RESUME"  \
 --imagenet_default_mean_and_std "$IMAGENET_DEFAULT_MEAN_AND_STD"  --data_set "$DATA_SET" \
 --num_workers "$NUM_WORKERS" --pin_mem "$PIN_MEM" --eval "$EVAL"
