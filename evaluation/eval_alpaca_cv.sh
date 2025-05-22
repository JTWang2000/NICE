#!/bin/bash
source eval.sh
# Directory containing the checkpoints
CHECKPOINT_DIR=$1  # Directory where checkpoints are stored
MODEL=$2           # Model name (e.g., llama)
CUDA=$3

# Set environment variables
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=$CUDA

# Temporary file to store results
TEMP_FILE="$CHECKPOINT_DIR/alpaca_val_results.txt"
echo $TEMP_FILE

# main evaluation function for alpaca
eval_alpaca() {
    mdir=$1
    model=$2
    split=$3
    set_save_dir $mdir "alpaca_$split"
    mkdir -p $save_dir

    if [[ "$model" == "llama" ]]; then
        chat_formatting_function="eval.templates.create_prompt_with_tulu_chat_format"
    elif [[ "$model" == "mistral" ]]; then
        chat_formatting_function="eval.templates.create_prompt_with_llama2_chat_format"
    else
        echo "Unsupported model type"
        return 1
    fi

    if [[ "$split" == "test" ]]; then
        data_file="$DATA_DIR/alpaca_eval/test/alpaca_test_data_diverse.json"
    elif [[ "$split" == "eval" ]]; then
        data_file="$DATA_DIR/alpaca_eval/eval/alpaca_eval_data_diverse.json"
    fi

    cmd="python -m eval.alpaca_eval.run_eval \
    --reference_path $data_file \
    --annotators_config weighted_alpaca_eval_gpt4_turbo \
    --max_new_tokens 512 \
    --save_dir $save_dir \
    --model $mdir \
    --tokenizer $mdir \
    --eval_batch_size 20 \
    --use_chat_format \
    --convert_to_bf16 \
    --chat_formatting_function $chat_formatting_function"
    eval "$cmd" 2>&1 | tee $save_dir/log.txt
}


# Define the function to extract evaluation results for Alpaca eval
extract_alpaca() {
    mdir=$1
    set_save_dir $mdir "alpaca_eval"
    # Extract the floating-point number following "win_rate"
    result=$(grep -oP '"length_controlled_winrate":\s*\K\d+\.\d+' $save_dir/win_rate.txt | head -n 1)
    echo $result
}

# Check if alpaca_val_results.txt already exists
if  [ -f "$TEMP_FILE" ]; then
    echo "alpaca_val_results.txt exists, finding the best checkpoint directly."
    # Find the checkpoint with the highest BLEURT score
    best_checkpoint=$(sort -k2 -n -r $TEMP_FILE | head -n 1 | awk '{print $1}')
    echo "Best checkpoint from existing results: $best_checkpoint"
else
    echo "alpaca_val_results.txt does not exist, running evaluation on checkpoints."

    # Iterate through all checkpoints in the directory with the pattern 'checkpoint-*'
    for checkpoint in "$CHECKPOINT_DIR"/checkpoint-*; do
        echo "Evaluating checkpoint: $checkpoint"
        eval_alpaca $checkpoint $MODEL eval

        # Extract the evaluation performance (e.g., reward score)
        reward_score=$(extract_alpaca $checkpoint)

        # Save checkpoint and its reward score
        echo "$checkpoint $reward_score" >> $TEMP_FILE
    done

    # Find the checkpoint with the highest reward score
    best_checkpoint=$(sort -k2 -n -r $TEMP_FILE | head -n 1 | awk '{print $1}')

    echo "Best checkpoint: $best_checkpoint"

fi
# Run final test on the best checkpoint
eval_alpaca $best_checkpoint $MODEL test

