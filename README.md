# encoder-fast-eval

Fast evaluation of encoder models on various NLP tasks by freezing the base model and training only the classification head.

## Features

- **Minimal & Clean**: Single Python script implementation
- **Multi-Task Support**: Classification, token classification (NER), and pair classification
- **Multiple Runs**: Configurable number of runs with mean ± std statistics
- **Per-Dataset Configs**: Each dataset can have its own training hyperparameters
- **Incremental Results**: JSONL output that saves results after each dataset
- **Rich Output**: Beautiful tables in the CLI

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/encoder-fast-eval.git
cd encoder-fast-eval

# Install dependencies (Python 3.12+ required)
uv sync
```

## Usage

### 1. Generate example configuration
```bash
uv run encoder_eval.py --create-config > config.yaml
```

### 2. Run evaluation
```bash
uv run encoder_eval.py config.yaml
```

### 3. Override global runs
```bash
uv run encoder_eval.py config.yaml --runs 10
```

## Configuration

The configuration file uses YAML format:

```yaml
model: prajjwal1/bert-tiny          # Model to evaluate
freeze_base: true                   # Freeze base model parameters
runs: 3                            # Default number of runs per dataset
seed: 42                           # Random seed
output_dir: results                # Output directory

datasets:
  # Text Classification
  - name: iastate/onestop_english
    type: classification
    train_split: train
    eval_split: test
    text_column: text
    label_column: label
    max_length: 128
    epochs: 3
    learning_rate: 2e-5
    batch_size: 16
    runs: 5                         # Override global runs

  # Token Classification (NER)
  - name: conll2003
    type: token_classification
    train_split: train
    eval_split: validation
    tokens_column: tokens
    tags_column: ner_tags
    max_length: 128
    epochs: 5
    learning_rate: 5e-5
    batch_size: 32

  # Pair Classification (NLI, sentence pairs)
  - name: nyu-mll/glue
    type: pair_classification
    subset: mnli
    train_split: train
    eval_split: validation_matched
    text1_column: premise
    text2_column: hypothesis
    label_column: label
    max_length: 128
    epochs: 3
    learning_rate: 2e-5
    batch_size: 16
```

## Task Types

### `classification`
Single text classification (sentiment analysis, topic classification, etc.)

### `token_classification`
Token-level classification (NER, POS tagging, etc.)

### `pair_classification`
Two-sentence classification (NLI, paraphrase detection, next sentence prediction)

## Output

### CLI Table
The script displays a clean table with mean ± standard deviation for each metric:

```
                               Evaluation Results
┏━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ Dataset                 ┃ Type           ┃ Metric          ┃ Mean ± Std      ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ iastate/onestop_english │ classification │ eval_accuracy   │ 0.3386 ± 0.0123 │
│ iastate/onestop_english │ classification │ eval_loss       │ 1.0976 ± 0.0031 │
└─────────────────────────┴────────────────┴─────────────────┴─────────────────┘
```

### JSONL Results
Results are saved incrementally to `results/results.jsonl`:

```json
{"dataset": "iastate/onestop_english", "type": "classification", "num_labels": 3, "runs": 2, "metrics": {"eval_accuracy_mean": 0.3386, "eval_accuracy_std": 0.0123, "eval_loss_mean": 1.0976, "eval_loss_std": 0.0031}}
```

## Example

```bash
# Test with a tiny model
cat > test.yaml << EOF
model: prajjwal1/bert-tiny
freeze_base: true
runs: 2
datasets:
  - name: iastate/onestop_english
    type: classification
    train_split: train
    eval_split: train
    text_column: text
    label_column: label
    epochs: 2
    learning_rate: 2e-5
    batch_size: 16
EOF

uv run encoder_eval.py test.yaml
```

## Dependencies

- `transformers` - Model loading and training
- `datasets` - Dataset loading
- `evaluate` - Metrics computation
- `pyyaml` - YAML configuration parsing
- `rich` - Beautiful CLI output
- `accelerate` - Efficient training
- `scikit-learn` - Metrics (accuracy, seqeval)