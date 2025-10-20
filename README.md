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
```

## Task Types

- **classification**: Single text classification
- **token_classification**: Token-level classification (NER, POS tagging)
- **pair_classification**: Two-sentence classification (NLI, paraphrase detection)

## Output

Results display in the CLI and save to `results/results.jsonl`:

```
┏━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ Dataset                 ┃ Type           ┃ Metric          ┃ Mean ± Std      ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ iastate/onestop_english │ classification │ eval_accuracy   │ 0.3386 ± 0.0123 │
└─────────────────────────┴────────────────┴─────────────────┴─────────────────┘
```

```json
{"dataset": "iastate/onestop_english", "type": "classification", "num_labels": 3, "runs": 2, "metrics": {"eval_accuracy_mean": 0.3386, "eval_accuracy_std": 0.0123}}
```