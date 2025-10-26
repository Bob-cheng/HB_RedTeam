# Import necessary libraries and modules
import transformers  # Hugging Face transformers library, used for loading and using pre-trained models
import json  # JSON processing library, used for reading and saving data
import argparse  # Command-line argument parsing library
import os  # Operating system interface library, used for file and directory operations
import csv  # CSV file processing library
from tqdm import tqdm  # Progress bar library, used to display processing progress
import torch  # PyTorch deep learning framework
from baselines import get_template, load_model_and_tokenizer, load_vllm_model  # Import baseline-related utility functions
from api_models import api_models_map  # Import API model mapping
from functools import partial  # Used to create partial functions
import multimodalmodels  # Multimodal model module
from PIL import Image  # PIL image processing library
from torchvision import transforms  # torchvision image transformation module
from vllm import SamplingParams  # vLLM sampling parameters
from accelerate.utils import find_executable_batch_size  # Used to automatically find the executable batch size

import yaml  # YAML file processing library, used for reading configuration files

# Disable warning messages
# Set this to disable warning messages in the generation mode.
# Set the logging level of the transformers library to ERROR to disable warning messages in generation mode.
transformers.utils.logging.set_verbosity_error()

def parse_args():
    """
    Parse command-line arguments
    
    Returns:
        argparse.Namespace: An object containing all command-line arguments
    """
    # Create an argument parser for parsing command-line arguments
    parser = argparse.ArgumentParser(description="运行基线方法的红队测试。")
    # Add model name argument, used to specify the model to use
    parser.add_argument("--model_name", type=str,
                        help="模型配置文件中模型的名称")
    # Add model configuration file path argument, used to specify the location of the model configuration file
    parser.add_argument("--models_config_file", type=str, default='./configs/model_configs/models.yaml',
                        help="包含模型超参数的配置文件路径")
    # Add behaviors file path argument, used to specify the location of the behaviors dataset
    parser.add_argument("--behaviors_path", type=str, default='./data/behavior_datasets/harmbench_behaviors_text_all.csv',
                        help="行为文件的路径")
    # Add test cases file path argument, used to specify the path to the test cases file for generating completions
    parser.add_argument("--test_cases_path", type=str,
                        help="用于生成补全的测试用例文件路径")
    # Add save path argument, used to specify the location to save the generated results
    parser.add_argument("--save_path", type=str,
                        help="用于保存补全结果的路径")
    # Add max new tokens argument, used to control the maximum length of generated text
    parser.add_argument("--max_new_tokens", type=int, default=256,
                        help="补全的最大新标记数量")
    # Add vLLM generation flag argument, used to choose whether to use vLLM for generation
    parser.add_argument("--generate_with_vllm", action="store_true",
                        help="是否使用 vLLM 生成补全（如果适用）")
    # Add incremental update flag argument, used to choose whether to incrementally update results
    parser.add_argument("--incremental_update", action="store_true",
                        help="是否增量更新补全结果或从头开始生成新的补全文件")
    # Parse command-line arguments
    args = parser.parse_args()
    return args


def main():
    """
    Main function, responsible for generating model completions
    """
    # ========== Load Parameters and Configuration ========== #
    # Parse command-line arguments
    args = parse_args()
    print(args)

    # Load model configuration file
    config_file = f"configs/model_configs/models.yaml" if not args.models_config_file else args.models_config_file
    with open(config_file) as file:
        model_configs = yaml.full_load(file)
    
    # Get the number of GPUs from the model configuration and check if the available GPU count matches
    num_gpus = model_configs[args.model_name].get('num_gpus', 0)
    # Check if the number of available GPUs matches the number of GPUs in the configuration
    num_gpus_available = torch.cuda.device_count()
    if num_gpus_available != num_gpus:
        print(f"Warning: num_gpus in config ({num_gpus}) does not match num_gpus available ({num_gpus_available}). Using {num_gpus_available} GPUs.")
        num_gpus = num_gpus_available
    model_config = model_configs[args.model_name]['model']
    model_config['num_gpus'] = num_gpus

    print("model_config", model_config)
    
    # ========== Generate Test Cases ========== #
    # Load behaviors CSV file
    with open(args.behaviors_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        behaviors = list(reader)[1:]  # Ignore header
   
    # Load test cases file
    with open(args.test_cases_path, 'r') as f:
        test_cases_data = json.load(f)

    # Load generation function, choose appropriate generation method based on configuration
    generation_function = load_generation_function(model_config,
                                                args.max_new_tokens,
                                                test_cases_path=args.test_cases_path,
                                                generate_with_vllm=args.generate_with_vllm)

    # Initialize returned data dictionary
    returned_data = {}

    # If incremental update is enabled and save path exists, load existing data
    if args.incremental_update and os.path.exists(args.save_path):
        with open(args.save_path, 'r') as file:
            returned_data = json.load(file)
        
        # Remove already generated test cases
        new_test_cases_data = {}
        new_returned_data = {}
        for behavior_id, _test_cases in test_cases_data.items():
            # Check if behavior ID is in returned data and if completions have been generated for all test cases
            if behavior_id in returned_data and len(returned_data[behavior_id]) == len(_test_cases):
                continue
            print(behavior_id, len(_test_cases))
            new_test_cases_data[behavior_id] = _test_cases
            new_returned_data[behavior_id] = []  # If not all completions exist, we will regenerate all completions
        test_cases_data = new_test_cases_data

    # ==== Expand dictionary to array ======
    test_cases = []
    for behavior_id, _test_cases in test_cases_data.items():
        test_cases.extend([dict(behavior_id=behavior_id, test_case=t) for t in _test_cases])

    # Check if there are test cases for which completions need to be generated
    if len(test_cases) == 0:
        print('No test cases to generate completions for')
        return
    else:
        print(f'Generating completions for {len(test_cases)} test cases')

    # ==== Generate Completions ====
    print('Generating completions...')
    generations = generation_function(test_cases=[t['test_case'] for t in test_cases])
    print('Done')

    # ==== Reorganize results into dictionary format {behavior_id: [{test_case: t, generation: g}, ..], ...}
    for t, g in zip(test_cases, generations):
        returned_data.setdefault(t['behavior_id'], [])
        returned_data[t['behavior_id']].append(dict(test_case=t['test_case'], generation=g))

    # Create directory for save path
    print(f'Saving completions to {args.save_path}...')
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True) if os.path.dirname(args.save_path) else None
    with open(args.save_path, 'w') as file:
        json.dump(returned_data, file, indent=4)
    print('Saved')


def _vllm_generate(model, test_cases, template, **generation_kwargs):
    """
    Generate text completions using vLLM model
    
    Args:
        model: vLLM model instance
        test_cases: List of test cases
        template: Template dictionary
        **generation_kwargs: Generation parameters
        
    Returns:
        list: List of generated texts
    """
    # Format input according to template
    inputs = [template['prompt'].format(instruction=s) for s in test_cases]
    # Use model to generate output
    outputs = model.generate(inputs, **generation_kwargs)
    # Extract generated text and remove leading/trailing whitespace
    generations = [o.outputs[0].text.strip() for o in outputs]
    return generations

def _hf_generate_with_batching(model, tokenizer, test_cases, template, **generation_kwargs):
    """
    Generate text completions using Hugging Face model with batching
    
    Args:
        model: Hugging Face model instance
        tokenizer: Tokenizer
        test_cases: List of test cases
        template: Template dictionary
        **generation_kwargs: Generation parameters
        
    Returns:
        list: List of generated texts
    """
    # Use find_executable_batch_size decorator to automatically find the executable batch size
    @find_executable_batch_size(starting_batch_size=len(test_cases))
    def inner_generation_loop(batch_size):
        # Declare variables from outer function
        nonlocal model, tokenizer, test_cases, template, generation_kwargs
        generations = []
        # Use tqdm to display progress bar, iterate through test cases by batch size
        for i in tqdm(range(0, len(test_cases), batch_size)):
            # Get current batch of test cases
            batched_test_cases = test_cases[i:i+batch_size]
            # Format input according to template
            inputs = [template['prompt'].format(instruction=s) for s in batched_test_cases]
            # Process input using tokenizer
            inputs = tokenizer(inputs, return_tensors='pt', padding=True)
            # Move input to the device where the model is located
            inputs = inputs.to(model.device)
            # Do not compute gradients to save memory
            with torch.no_grad():
                # Generate output
                outputs = model.generate(inputs=inputs['input_ids'], attention_mask=inputs['attention_mask'], **generation_kwargs).cpu()
            # Extract generated tokens (excluding input part)
            generated_tokens = outputs[:, inputs['input_ids'].shape[1]:]
            # Decode generated tokens to text
            batch_generations = [tokenizer.decode(o, skip_special_tokens=True).strip() for o in generated_tokens]
            # Add current batch's generation results to total results
            generations.extend(batch_generations)
        return generations
    return inner_generation_loop()

def _api_model_generate(model, test_cases, **generation_kwargs):
    """
    Generate text completions using API model
    
    Args:
        model: API model instance
        test_cases: List of test cases
        **generation_kwargs: Generation parameters
        
    Returns:
        list: List of generated texts
    """
    # Multimodal test cases are saved in the format [[img, text], ...]
    if isinstance(test_cases[0], (tuple, list)):
        # Unpack images and prompt texts
        images, prompts = zip(*test_cases)
        # Generate completions for multimodal models
        return model.generate(prompts=list(prompts), images=list(images), **generation_kwargs)   
    # Generate completions for text models
    return model.generate(prompts=test_cases, **generation_kwargs)

def load_generation_function(model_config, max_new_tokens, test_cases_path, generate_with_vllm=False):
    """
    Load corresponding generation function based on model configuration
    
    Args:
        model_config: Model configuration dictionary
        max_new_tokens: Maximum number of new tokens to generate
        test_cases_path: Path to test cases
        generate_with_vllm: Whether to use vLLM for generation
        
    Returns:
        partial: Partial function for generation
    """
    # Get model name or path
    model_name_or_path = model_config['model_name_or_path']

    # Check if it is a multimodal model
    if model_name_or_path in dir(multimodalmodels):
        # Load multimodal model
        model = getattr(multimodalmodels, model_name_or_path)()
        # Set generation parameters
        generation_kwargs = dict(max_new_tokens=max_new_tokens, do_sample=False, num_beams=1)
        # Construct image directory path
        image_dir = os.path.join(os.path.dirname(test_cases_path), 'images')
        # Return partially applied generation function
        return partial(model.generate, image_dir=image_dir, **generation_kwargs)
    # Check if it is an API model
    if (model := api_models_map(**model_config)):
        # Set API model generation parameters
        generation_kwargs = dict(max_new_tokens=max_new_tokens, temperature=0.0, use_tqdm=True)
        # Return partially applied API generation function
        return partial(_api_model_generate, model=model, **generation_kwargs)
    # Check if vLLM generation is used
    elif generate_with_vllm:
        print('Using vLLM generation')
        # Create sampling parameters
        sp = SamplingParams(temperature=0, max_tokens=max_new_tokens)
        # Load vLLM model
        model = load_vllm_model(**model_config)
        # Set generation parameters
        generation_kwargs = dict(sampling_params=sp, use_tqdm=True)
        # Get template
        TEMPLATE = get_template(model_name_or_path, chat_template=model_config.get('chat_template', None))
        # Return partially applied vLLM generation function
        return partial(_vllm_generate, model=model, template=TEMPLATE, **generation_kwargs)
    # Default to Hugging Face generation
    else:
        print('Using HF generation')
        # Load Hugging Face model and tokenizer
        model, tokenizer = load_model_and_tokenizer(**model_config)
        # Set generation parameters
        generation_kwargs = dict(max_new_tokens=max_new_tokens, do_sample=False)
        # Get template
        TEMPLATE = get_template(model_name_or_path, chat_template=model_config.get('chat_template', None))
        # Return partially applied Hugging Face generation function
        return partial(_hf_generate_with_batching, model=model, tokenizer=tokenizer, template=TEMPLATE, **generation_kwargs)



def bubble_sort(arr):
    """
    冒泡排序算法实现
    
    Args:
        arr: 待排序的数字列表
        
    Returns:
        List: 按升序排列的列表
    """
    # Handle edge cases
    if not isinstance(arr, list):
        raise TypeError("Input must be a list")
    
    if len(arr) <= 1:
        return arr.copy()  # Return a copy to avoid modifying the original
    
    # Work on a copy to avoid modifying the original array
    arr_copy = arr.copy()
    n = len(arr_copy)
    
    # Traverse through all array elements
    for i in range(n):
        # Flag to optimize - if no swapping occurs, array is sorted
        swapped = False
        # Last i elements are already in place
        for j in range(0, n - i - 1):
            # Traverse the array from 0 to n-i-1
            # Swap if the element found is greater than the next element
            if arr_copy[j] > arr_copy[j + 1]:
                arr_copy[j], arr_copy[j + 1] = arr_copy[j + 1], arr_copy[j]
                swapped = True
        # If no swapping occurred, array is sorted
        if not swapped:
            break
    return arr_copy


if __name__ == "__main__":
    main()
