#!/bin/bash
source /opt/rh/devtoolset-10/enable

# model_name=$1
# behaviors_path=$2
# test_cases_path=$3
# save_path=$4
# max_new_tokens=$5
# incremental_update=$6

# Make sure your environment has enough GPUs for the model
base_save_dir="./results"

# Often, the experiment name will be the same as the model name. It can be different for transfer attacks.
method_name="PAIR"
experiment_name="qwen-2.5-7b" 
model_name="qwen-2.5-7b" # from configs/model_configs/models.yaml
behaviors_path="./data/behavior_datasets/harmbench_behaviors_text_val.csv"
max_new_tokens=512
incremental_update="True"

experiment_dir="$base_save_dir/$method_name/$experiment_name"
test_cases_path="$experiment_dir/test_cases/test_cases.json"
save_path="$experiment_dir/completions/${model_name}.json"

echo "model_name=$model_name"
echo "behaviors_path=$behaviors_path"
echo "test_cases_path=$test_cases_path"
echo "max_new_tokens=$max_new_tokens"
echo "save_path=$save_path"
echo "incremental_update=$incremental_update"

# Construct the base command
cmd="python -u generate_completions.py \
    --model_name $model_name \
    --behaviors_path $behaviors_path \
    --test_cases_path $test_cases_path \
    --save_path $save_path \
    --max_new_tokens $max_new_tokens \
    --generate_with_vllm"

# Check if incremental_update is provided and equals "True"
if [[ ! -z "$incremental_update" && "$incremental_update" == "True" ]]; then
    cmd="$cmd --incremental_update"
fi

# Execute the command
eval $cmd
