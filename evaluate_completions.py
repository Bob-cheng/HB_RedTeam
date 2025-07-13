import transformers
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import json
import argparse
import os
import csv
from tqdm import tqdm 
import numpy as np
from api_models import api_models_map
import yaml

from vllm import LLM, SamplingParams

from eval_utils import compute_results_hashing, compute_results_classifier, compute_results_advbench, compute_results_api
# Set this to disable warning messages in the generation mode.
transformers.utils.logging.set_verbosity_error()

def parse_args():
    parser = argparse.ArgumentParser(description="Running red teaming with baseline methods.")
    parser.add_argument("--cls_path", type=str, default='cais/HarmBench-Llama-2-13b-cls',
                        help="The name or path of the classifier for computing ASR")
    parser.add_argument("--behaviors_path", type=str, default='./data/behavior_datasets/harmbench_behaviors_text_all.csv',
                        help="The path to the behaviors file")
    parser.add_argument("--completions_path", type=str,
                        help="The path to the completions to evaluate")
    parser.add_argument("--save_path", type=str,
                        help="The path for saving results")
    parser.add_argument("--include_advbench_metric", action="store_true",
                        help="Whether to include the AdvBench refusal metric")
    parser.add_argument("--num_tokens", type=int, default=512,
                        help="The number of tokens to evaluate")
    args = parser.parse_args()
    return args


def main():
    # ========== load arguments and config ========== #
    args = parse_args()
    print(args)
    
    # ========== load behaviors (for tags and context strings) ========== #
    with open(args.behaviors_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        behaviors = list(reader)
    
    # convert to dictionary mapping behavior ID field to behavior dict
    behaviors = {b['BehaviorID']: b for b in behaviors}

    # ========== initialize results ========== #
    with open(args.completions_path, 'r') as f:
        completions = json.load(f)

    config_file = f"configs/model_configs/models.yaml"
    with open(config_file) as file:
        model_configs = yaml.full_load(file)
    if args.cls_path in model_configs and model_configs[args.cls_path]['model_type'] == 'closed_source':
        model_name_or_path = model_configs[args.cls_path]['model']['model_name_or_path']
        api_key = model_configs[args.cls_path]['model']['token']
        api_model = api_models_map(model_name_or_path, api_key)
    else:
        api_model = None
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.cls_path)
        tokenizer.truncation_side="right"

        # Clip the 'generation' field of the completions to have a maximum of num_tokens tokens
        for behavior_id, completion_list in completions.items():
            for completion in completion_list:
                generation = completion['generation']
                tokenized_text = tokenizer.encode(generation, max_length=args.num_tokens, truncation=True)
                clipped_generation = tokenizer.decode(tokenized_text, skip_special_tokens=True)
                completion['generation'] = clipped_generation
        cls = LLM(model=args.cls_path, tensor_parallel_size=1)
        cls.llm_engine.tokenizer.tokenizer.truncation_side="left"

    # ========== evaluate completions ========== #
    cls_params = SamplingParams(temperature=0.0, max_tokens=1)

    results = {}
    for behavior_id, data in tqdm(completions.items()):
        if behavior_id not in behaviors:
            print(f"Behavior {behavior_id} not found in behaviors file; skipping")
            continue
        behavior_dict = behaviors[behavior_id]
        tags = behavior_dict['Tags'].split(', ')

        if 'hash_check' in tags:
            current_results = compute_results_hashing(behavior_dict, data)
        elif api_model is not None:
            current_results = compute_results_api(behavior_dict, data, api_model, cls_params)
        else:
            current_results = compute_results_classifier(behavior_dict, data, cls, cls_params)
        
        if args.include_advbench_metric:
            current_results = compute_results_advbench(current_results, data)

        results[behavior_id] = current_results

    success_rates = []

    for i, (behavior_id, data) in enumerate(results.items()):
        success_mean = np.mean([d['label'] == 1 for d in data])
        print(f"{i}. {behavior_id} === average asr: {success_mean}")
        success_rates.append(success_mean)
    
    print("Average ASR: ", np.mean(success_rates))

    # Make dirs to output_file if not yet exist
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True) if os.path.dirname(args.save_path) else None
    with open(args.save_path, 'w') as file:
        json.dump(results, file, indent=4)

if __name__ == "__main__":
    main()
