# AI Notice
* Generated README Using Co-Pilot Agent
* AI models (i.e., Claude Opus 4.5 and Gemini 3 used to debug and format code for readability)
* No AI agent or model used in running or conducting any experiments

# Do LLMs Respond to Instructional Quality as Students Should? A Likelihood-Based Probe of Pedagogical Sensitivity?

This repository studies whether higher-quality tutor responses receive higher student-model likelihood under a fixed student prompt.

## Comparison Models

Student models used to compute log-probabilities:

- Qwen/Qwen3-30B-A3B-Instruct-2507-FP8
- microsoft/Phi-4-reasoning-plus
- openai/gpt-oss-20b
- google/gemma-3-12b-it
- meta-llama/Llama-3.2-3B-Instruct

Design summary:

- Mixture of thinking and non-thinking models
- Mixture of scales (3B to 30B)
- Controls include ablations with null/no-last-response and mismatch settings

## Repository Layout

- `asset/data/MRBench_V2.json`: original benchmark data
- `asset/data/ReadyForLogProb.json`: processed input for scoring
- `asset/data/logprob_results/<model_name>/`: per-student-model experiment outputs
- `asset/scripts/run_main_experiment.py`: main log-prob scoring script
- `asset/scripts/result_analysis/result_analysis_v3.py`: statistical analysis utilities
- `visualisation/figures.py`: final paper-style figure generation

## Reproduce Final Visualisations (Fast Path)

Use this if you want the same end results in this repository without re-running expensive inference.

1. Create and activate an environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

2. Install dependencies:

```bash
pip install torch transformers numpy scipy matplotlib seaborn
```

3. Generate figures from bundled results:

```bash
python visualisation/figures.py \
	--results-dir asset/data/logprob_results \
	--output-dir visualisation/figures
```

Expected outputs:

- `visualisation/figures/figure1_violins.pdf`
- `visualisation/figures/figure2_ablation.pdf`

## Full Pipeline (Including Experiment Runs)

Run this only if you want to regenerate scoring outputs.

### 1) (Optional) Rebuild `ReadyForLogProb.json`

```bash
cd asset/scripts/data_processing
python data_processing.py
cp current_data_v2.json ../../data/ReadyForLogProb.json
cd ../../..
```

### 2) Run main scoring for each student model

From repository root:

```bash
python asset/scripts/run_main_experiment.py -m "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8"
python asset/scripts/run_main_experiment.py -m "microsoft/Phi-4-reasoning-plus"
python asset/scripts/run_main_experiment.py -m "openai/gpt-oss-20b"
python asset/scripts/run_main_experiment.py -m "google/gemma-3-12b-it"
python asset/scripts/run_main_experiment.py -m "meta-llama/Llama-3.2-3B-Instruct"
```

For ablation mode:

```bash
python asset/scripts/run_main_experiment.py -m "<model_name>" --ablation
```

Notes:

- `run_main_experiment.py` currently writes `regular.json` and `ablation.json`.
- This repository already includes `mismatch.json` and `no_last_response.json` under `asset/data/logprob_results/` for figure and analysis reproduction.

### 3) Generate visualisations

```bash
python visualisation/figures.py \
	--results-dir asset/data/logprob_results \
	--output-dir visualisation/figures
```

## Optional Analysis Script

`asset/scripts/result_analysis/result_analysis_v3.py` is set up as a utility module and requires paths to be provided in code (the `PATH` list) or via a small wrapper.

## Practical Notes

- Inference is GPU-intensive; use CUDA where possible.
- Ensure model access/auth is configured for gated Hugging Face models before full reruns.
- The bundled `asset/data/logprob_results/` files are the authoritative source for reproducing current figures.