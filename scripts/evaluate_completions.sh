#!/bin/bash

source /opt/rh/devtoolset-10/enable

# cls_path="$1"
# behaviors_path=$2
# completions_path=$3
# save_path=$4

# Make sure your environment has enough GPUs for the classifier (Llama 2 13B)
base_save_dir="./results"

method="PAIR"
# Often, the experiment name will be the same as the model name. It can be different for transfer attacks.
experiment="qwen-2.5-7b" 
model="qwen-2.5-7b" # from configs/model_configs/models.yaml

# cls_path="cais/HarmBench-Llama-2-13b-cls"
cls_path="gemini-2.5-flash" # from configs/model_configs/models.yaml
behaviors_path="./data/behavior_datasets/harmbench_behaviors_text_val.csv"
completions_path="$base_save_dir/$method/$experiment/completions/$model.json" # output file from Step 2
save_path="$base_save_dir/$method/$experiment/results/$model.json"

echo "cls_path=$cls_path"
echo "behaviors_path=$behaviors_path"
echo "completions_path=$completions_path"
echo "save_path=$save_path"


python -u evaluate_completions.py \
    --cls_path $cls_path \
    --behaviors_path $behaviors_path \
    --completions_path $completions_path \
    --save_path $save_path \
    --include_advbench_metric
