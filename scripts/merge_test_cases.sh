#!/bin/bash
base_save_dir="/home/bob/learn_LLM/HarmBench/results"
experiment_name="qwen-3-4b" 
method_name="PAIR"
save_dir="$base_save_dir/$method_name/$experiment_name/test_cases"


# method_name="$1"
# save_dir="$2"

echo "method_name=$method_name"
echo "save_dir=$save_dir"

python -u merge_test_cases.py --method_name $method_name --save_dir $save_dir  