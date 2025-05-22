# Warmup-Train
DATA_DIR=./data
MODEL_PATH=meta-llama/Llama-2-7b-hf
PERCENTAGE=0.05 # percentage of the full data to train, you can specify the training file you want to use in the script
DATA_SEED=0
JOB_NAME=llama2-7b-p${PERCENTAGE}-seed${DATA_SEED}-warmup-4epochs

CUDA_VISIBLE_DEVICES=7 bash ./nice/scripts/train/warmup_lora_train.sh "$DATA_DIR" "$MODEL_PATH" "$PERCENTAGE" "$DATA_SEED" "$JOB_NAME"


# TRAIN gradients
conda activate ./myenv
# Run all CKPT: 105; 211; 317; 420
# TRAINING_DATA_NAME=cot; oasst1; flan_v2; dolly
CKPT="105"
MODEL_NAME="llama2-7b-p0.05-seed0-warmup-4epochs"
TRAINING_DATA_NAME=cot
TRAINING_DATA_FILE=./data/train/processed/${TRAINING_DATA_NAME}/${TRAINING_DATA_NAME}_data.jsonl # when changing data name, change the data path accordingly
GRADIENT_TYPE="adam"
MODEL_PATH=/home/NICE/out/${MODEL_NAME}/checkpoint-${CKPT}
OUTPUT_PATH=/home/NICE/grads/${MODEL_NAME}/${TRAINING_DATA_NAME}-ckpt${CKPT}-${GRADIENT_TYPE}
DIMS="8192"

CUDA_VISIBLE_DEVICES=0 ./nice/scripts/get_info/grad/get_train_lora_grads.sh "$TRAINING_DATA_FILE" "$MODEL_PATH" "$OUTPUT_PATH" "$DIMS" "$GRADIENT_TYPE"


# VAL generations
# Run all CKPT: 105; 211; 317; 420
# GRADIENT_TYPE: policy_alpaca; policy_tldr; policy_reward; policy_codex
# Task: alpaca; tldr; hh_rlhf; codex (HumanEval)
CKPT="105"
GRADIENT_TYPE="policy_reward"
TASK=hh_rlhf
MODEL_NAME="llama2-7b-p0.05-seed0-warmup-4epochs"
MODEL_PATH=/home/NICE/out/${MODEL_NAME}/checkpoint-${CKPT}
TMP=1.2
OUTPUT_PATH=/home/NICE/generations/${MODEL_NAME}/${TASK}-val-ckpt${CKPT}-${GRADIENT_TYPE}-${TMP}
DATA_DIR=./data
MC=20
CUDA_VISIBLE_DEVICES=0 bash ./nice/scripts/get_info/grad/get_eval_generations.sh "$TASK" "$DATA_DIR" "$MODEL_PATH" $OUTPUT_PATH "$GRADIENT_TYPE" "$MC" "$TMP"


# VAL gradients
# Run all CKPT: 105; 211; 317; 420
# GRADIENT_TYPE: policy_alpaca; policy_tldr; policy_reward; policy_codex
# Task: alpaca; tldr; hh_rlhf; codex (HumanEval)
CKPT="105"
GRADIENT_TYPE="policy_reward"
TASK=hh_rlhf
MODEL_NAME="llama2-7b-p0.05-seed0-warmup-4epochs"
MODEL_PATH=/home/NICE/out/${MODEL_NAME}/checkpoint-${CKPT}
POLICY="vanilla"
MC=20
TMP=1.2
OUTPUT_PATH=/home/NICE/grads/${MODEL_NAME}/${TASK}-val-ckpt${CKPT}-${GRADIENT_TYPE}-${TMP}_${POLICY}_${MC}
DATA_DIR=./data
CUDA_VISIBLE_DEVICES=0 ./nice/scripts/get_info/grad/get_eval_lora_grads.sh "$TASK" "$DATA_DIR" "$MODEL_PATH" "$OUTPUT_PATH" "$GRADIENT_TYPE" "$POLICY" "$MC"


# NICE score
# GRADIENT_TYPE_VAL: {GRADIENTTYPE}_{TMP}_vanilla_MC
# TARGET_TASK_NAMES: alpaca; tldr; hh_rlhf; codex (HumanEval)
MODEL_NAME="llama2-7b-p0.05-seed0-warmup-4epochs"
DIM=8192 # decide which dimension to use
GRADIENT_PATH=/home/NICE/grads/${MODEL_NAME}/{}-ckpt{}-adam/dim${DIM}
TRAIN_FILE_NAMES="flan_v2 cot dolly oasst1"
CKPTS="105 211 317 420"
CHECKPOINT_WEIGHTS="1.7069e-05 1.2875e-05 7.6658e-06 2.3660e-06"

GRADIENT_TYPE_VAL="policy_reward-1.2_vanilla_20"
VALIDATION_GRADIENT_PATH=/home/NICE/grads/${MODEL_NAME}/{}-val-ckpt{}-${GRADIENT_TYPE_VAL}/dim${DIM}
TARGET_TASK_NAMES="hh_rlhf"
SELECTED_DATA_OUTPUT_PATH="/home/NICE/selected_data/${MODEL_NAME}-${GRADIENT_TYPE_VAL}-${DIM}/"
CUDA_VISIBLE_DEVICES=0 bash ./nice/scripts/data_selection/matching_nice.sh "$GRADIENT_PATH" "$TRAIN_FILE_NAMES" "$CKPTS" "$CHECKPOINT_WEIGHTS" "$VALIDATION_GRADIENT_PATH" "$TARGET_TASK_NAMES" "$SELECTED_DATA_OUTPUT_PATH"


# SELECT top K
# TARGET_TASK_NAMES: alpaca; tldr; hh_rlhf; codex (HumanEval)
SELECTED_DATA_OUTPUT_PATH="/home/NICE/selected_data/${MODEL_NAME}-${GRADIENT_TYPE_VAL}-${DIM}/"
TARGET_TASK_NAMES="hh_rlhf"
TRAIN_FILE_NAMES="flan_v2 cot dolly oasst1"
CUDA_VISIBLE_DEVICES=0 python3 -m nice.data_selection.write_selected_data \
--target_task_names ${TARGET_TASK_NAMES} \
--train_file_names ${TRAIN_FILE_NAMES} \
--train_files ./data/train/processed/flan_v2/flan_v2_data.jsonl ./data/train/processed/cot/cot_data.jsonl ./data/train/processed/dolly/dolly_data.jsonl ./data/train/processed/oasst1/oasst1_data.jsonl \
--output_path $SELECTED_DATA_OUTPUT_PATH \
--percentage 0.05

# Retrain
PERCENTAGE=0.05
SEED=0
TRAIN_FILES="/home/NICE/selected_data/${MODEL_NAME}-${GRADIENT_TYPE_VAL}-${DIM}/hh_rlhf/top_p0.05.jsonl"
MODEL_PATH=meta-llama/Llama-2-7b-hf
JOB_NAME=llama2-7b-p${PERCENTAGE}-lora-seed0-policy_reward-1.2_vanilla_20
CUDA_VISIBLE_DEVICES=0 bash ./nice/scripts/train/lora_train.sh "$TRAIN_FILES" "$MODEL_PATH" "$JOB_NAME" "$SEED"

# Evaluation
cd evaluation
# PROMPT_TYPE: llama; mistral; llama3
bash eval_rlhf_cv.sh MODEL_PATH PROMPT_TYPE CUDA