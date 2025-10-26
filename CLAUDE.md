# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Setup
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### Running the Evaluation Pipeline
The evaluation pipeline has four main steps:

1. **Generate test cases**:
```bash
# Using run_pipeline.py (recommended)
python ./scripts/run_pipeline.py --methods GCG --models all --step 1 --mode slurm

# Using individual scripts
./scripts/generate_test_cases.sh $method_name $experiment_name $behaviors_path $save_dir $start_idx $end_idx
```

2. **Merge test cases** (if needed):
```bash
# Using run_pipeline.py
python ./scripts/run_pipeline.py --methods GCG --models all --step 1.5 --mode slurm

# Using individual script
./scripts/merge_test_cases.sh $method_name $save_dir
```

3. **Generate completions**:
```bash
# Using run_pipeline.py
python ./scripts/run_pipeline.py --methods GCG --models all --step 2 --mode slurm

# Using individual script
./scripts/generate_completions.sh $model_name $behaviors_path $test_cases_path $save_path $max_new_tokens
```

4. **Evaluate completions**:
```bash
# Using run_pipeline.py
python ./scripts/run_pipeline.py --methods GCG --models all --step 3 --mode slurm

# Using individual script
./scripts/evaluate_completions.sh $cls_path $behaviors_path $completions_path $save_path
```

### Running Individual Steps with run_pipeline.py
```bash
# Run complete pipeline for GCG against all models
python ./scripts/run_pipeline.py --methods GCG --models all --step all --mode slurm

# Run pipeline for all methods against Llama 2 7B
python ./scripts/run_pipeline.py --methods all --models llama2_7b --step all --mode slurm

# Run specific steps
python ./scripts/run_pipeline.py --methods GCG --models all --step 2_and_3 --mode slurm
```

## Code Architecture

### Key Components
1. **Evaluation Pipeline**: Consists of four main scripts:
   - `generate_test_cases.py`: Generates attack prompts using red teaming methods
   - `merge_test_cases.py`: Combines test cases from individual behaviors
   - `generate_completions.py`: Runs test cases on target models
   - `evaluate_completions.py`: Evaluates if completions are successful attacks

2. **Baselines**: Contains implementations of red teaming methods in the `baselines/` directory. Each method is a subclass of `RedTeamingMethod` in `baselines/baseline.py`.

3. **Configs**: Configuration files in `configs/` directory:
   - `method_configs/`: YAML files defining hyperparameters for each red teaming method
   - `model_configs/models.yaml`: Defines target models and their parameters
   - `pipeline_configs/run_pipeline.yaml`: Configuration for the run_pipeline.py utility

4. **Data**: Contains:
   - `behavior_datasets/`: CSV files with harmful behaviors to test
   - `optimizer_targets/`: Target outputs for optimization-based attacks
   - `multimodal_behavior_images/`: Images used in multimodal attack evaluation

### Key Workflow
1. Red teaming methods in `baselines/` generate test cases based on configurations in `configs/method_configs/`
2. Test cases are saved to `results/{method}/{experiment}/test_cases/`
3. Target models defined in `configs/model_configs/models.yaml` generate completions
4. Completions are saved to `results/{method}/{experiment}/completions/`
5. Classifier evaluates completions, results saved to `results/{method}/{experiment}/results/`

### Dynamic Config System
The framework supports dynamic experiment configuration through template variables in config files:
- `<model_name>`: References model configurations in `models.yaml`
- Allows sharing model parameters across multiple method configurations
- Used in both method config files and `run_pipeline.yaml`

## Adding New Components

### Adding New Red Teaming Methods
1. Create a new subdirectory in `baselines/` for your method
2. Implement a class that inherits from `RedTeamingMethod` in `baselines/baseline.py`

   - Must implement the `generate_test_cases` method
   - Can override methods like `get_init_prompt` if needed
   - Should be added to `baselines/__init__.py`'s `_method_mapping`

3. Define your method's hyperparameters in `configs/method_configs/{method_name}_config.yaml`

   - Must include `default_method_hyperparameters` section
   - Can use dynamic configuration with `<model_name>` templates
   - Can define specific experiment configurations

4. Add your method to the `_method_mapping` dictionary in `baselines/__init__.py`

### Adding New Target Models
1. Add an entry for your model in `configs/model_configs/models.yaml`

   - Use consistent naming convention (e.g., `modelname_size_variant`)
   - Specify model parameters under the `model` key
   - Set appropriate `num_gpus` based on model size
   - Specify `model_type` (open_source, closed_source, open_source_multimodal)

2. For non-transformers models, additional implementation may be needed in `api_models.py` or `multimodalmodels/`

### Running on Different Environments
- `slurm` mode: For SLURM cluster environments with job dependencies
- `local` mode: For running sequentially on a single machine
- `local_parallel` mode: For parallelizing across GPUs on a single machine using Ray

### Key Files and Directories
- `scripts/run_pipeline.py`: Main entry point for running the evaluation pipeline
- `baselines/`: Core red teaming method implementations
- `configs/model_configs/models.yaml`: Central model configuration
- `data/behavior_datasets/`: Harmful behavior datasets for testing
- `results/`: Output directory for all pipeline results
- `notebooks/`: Jupyter notebooks for analysis and method development
- `docs/`: Documentation for the framework

### Pipeline Execution Patterns
- Use `step all` to run the complete pipeline
- Use `step 2_and_3` to generate completions and evaluate in one command
- Use specific step numbers to run individual pipeline stages
- Use `mode local` for testing on a single machine
- Use `mode slurm` for production runs on clusters

### Configuration Best Practices
- Use dynamic configuration (`<model_name>`) to avoid duplicating model paths
- Reference existing models in `models.yaml` when defining new experiments
- Use `run_pipeline.yaml` to define proper method names that map to experiments
- Keep experiment names descriptive and consistent
- Use the `base_num_gpus` field in `run_pipeline.yaml` to control SLURM job resources