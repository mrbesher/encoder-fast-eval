# encoder-fast-eval

Fast evaluation of encoder models on NLP tasks by freezing the base model and training only the classification head.

## Installation

```bash
git clone https://github.com/yourusername/encoder-fast-eval.git
cd encoder-fast-eval
uv sync
```

## Quick Start

```bash
# Generate configuration
uv run encoder_eval.py --create-config > config.yaml

# Run evaluation
uv run encoder_eval.py config.yaml

# Override runs
uv run encoder_eval.py config.yaml --runs 10
```

## Configuration

Edit `config.yaml` to specify your datasets and hyperparameters:

```yaml
model: prajjwal1/bert-tiny
freeze_base: true
runs: 3
seed: 42
output_dir: results

datasets:
  - name: iastate/onestop_english  # Classification
    type: classification
    train_split: train
    eval_split: test
    text_column: text
    label_column: label
    epochs: 3
    learning_rate: 2e-5
    batch_size: 16

  - name: conll2003  # Token classification (NER)
    type: token_classification
    train_split: train
    eval_split: validation
    tokens_column: tokens
    tags_column: ner_tags
    epochs: 5
    learning_rate: 5e-5
    batch_size: 32

  - name: nyu-mll/glue  # Pair classification (NLI)
    type: pair_classification
    subset: mnli
    train_split: train
    eval_split: validation_matched
    text1_column: premise
    text2_column: hypothesis
    label_column: label
    epochs: 3
    learning_rate: 2e-5
    batch_size: 16

  - name: nhull/125-tripadvisor-reviews  # Regression
    type: regression
    train_split: train
    eval_split: train  # Only train split available
    text_column: text
    label_column: label
    epochs: 5
    learning_rate: 5e-5
    batch_size: 16
```

## Task Types

- **classification**: Single text classification
- **token_classification**: Token-level classification (NER, POS tagging)
- **pair_classification**: Two-sentence classification (NLI, paraphrase detection)
- **regression**: Predict continuous values (scores, ratings)

## Output

Results display in the CLI and save to `results/results.jsonl`:

```
┏━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ Dataset                 ┃ Type           ┃ Metric          ┃ Mean ± Std      ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ iastate/onestop_english │ classification │ eval_accuracy   │ 0.3386 ± 0.0123 │
│ nhull/125-tripadvisor   │ regression     │ eval_rmse       │ 0.4215 ± 0.0321 │
└─────────────────────────┴────────────────┴─────────────────┴─────────────────┘
```

```json
{"dataset": "iastate/onestop_english", "type": "classification", "num_labels": 3, "runs": 2, "metrics": {"eval_accuracy_mean": 0.3386, "eval_accuracy_std": 0.0123}}
{"dataset": "nhull/125-tripadvisor-reviews", "type": "regression", "num_labels": 1, "runs": 3, "metrics": {"eval_rmse_mean": 0.4215, "eval_r2_mean": 0.6782}}
```

## Analyzing Results

The `analyze_results.py` script analyzes evaluation results across multiple models and datasets. It provides task averages and identifies best performers.

### Usage

```bash
# Show task averages in console (default)
uv run scripts/analyze_results.py

# Show full table with all datasets
uv run scripts/analyze_results.py console --full-table

# Export to Excel (with formulas for automatic averages)
uv run scripts/analyze_results.py excel -o results.xlsx

# Export to other formats
uv run scripts/analyze_results.py markdown > results.md
uv run scripts/analyze_results.py csv > results.csv
uv run scripts/analyze_results.py json > results.json
```

### Output Features

- **Task Averages**: Automatically calculates average scores per task type (classification, regression, etc.)
- **Best Performers**: Highlights the best-scoring model for each dataset and task type
- **Excel Formulas**: The Excel output uses formulas that update automatically when values change
- **Multiple Formats**: Supports console, Excel, Markdown, CSV, and JSON outputs

### Options

- `--results-dir`: Directory containing results (default: `results`)
- `--full-table`: Show all datasets in console mode
- `--no-bold`: Don't highlight best performers
- `-o/--output`: Output file name (Excel format only)

## Running on Modal

For cloud GPU evaluation using [Modal](https://modal.com):

### Setup

```bash
uv venv
source .venv/bin/activate
uv pip install modal huggingface-hub
```

### Usage

```bash
# Single model
modal run scripts/run_modal.py --models prajjwal1/bert-tiny --config ./examples/minimal.yaml

# Mix Hugging Face and local models
modal run scripts/run_modal.py --models prajjwal1/bert-tiny ~/my_local_model --config ./examples/minimal.yaml
```

Multiple models evaluate in parallel. After completion, automatically generates analysis in all formats (Excel, Markdown, CSV, JSON) saved to the results volume.