#!/usr/bin/env python3
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import yaml
from datasets import Dataset, DatasetDict, load_dataset
from evaluate import load as load_metric
from rich.console import Console
from rich.table import Table
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)
from transformers import set_seed


console = Console()
logger = logging.getLogger(__name__)


class DatasetConfig:
    name: str
    type: str
    subset: Optional[str] = None
    train_split: str = "train"
    eval_split: str = "validation"
    text_column: str = "text"
    label_column: str = "label"
    text1_column: Optional[str] = None
    text2_column: Optional[str] = None
    tokens_column: Optional[str] = None
    tags_column: Optional[str] = None
    max_length: int = 128
    epochs: int = 3
    learning_rate: float = 2e-5
    batch_size: int = 16
    runs: Optional[int] = None
    metric_name: Optional[str] = None

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if k in ["learning_rate", "batch_size", "max_length", "epochs", "runs", "seed"]:
                setattr(self, k, float(v) if k == "learning_rate" else int(v))
            else:
                setattr(self, k, v)


class EvalConfig:
    model: str
    freeze_base: bool = True
    runs: int = 3
    seed: int = 42
    output_dir: str = "results"
    datasets: List[DatasetConfig] = []

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if k == "datasets":
                self.datasets = [DatasetConfig(**d) for d in v]
            elif k in ["runs", "seed"]:
                setattr(self, k, int(v))
            else:
                setattr(self, k, v)


class TaskRunner:
    def __init__(self, config: EvalConfig):
        self.config = config
        self.tokenizer = None
        self.model = None

        set_seed(config.seed)

        os.makedirs(config.output_dir, exist_ok=True)

        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            level=logging.INFO,
        )

    def load_model_and_tokenizer(self, num_labels: int, label_names: Optional[List[str]] = None):
        logger.info(f"Loading model and tokenizer: {self.config.model}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model)

        if label_names:
            model_config = AutoConfig.from_pretrained(
                self.config.model,
                id2label={str(i): name for i, name in enumerate(label_names)},
                label2id={name: i for i, name in enumerate(label_names)},
            )
        else:
            model_config = AutoConfig.from_pretrained(
                self.config.model,
                num_labels=num_labels,
            )

        if self.config.freeze_base:
            self.base_model = AutoModelForSequenceClassification.from_pretrained(
                self.config.model,
                config=model_config,
            )
            for param in self.base_model.base_model.parameters():
                param.requires_grad = False
            logger.info("Frozen base model parameters")
            self.model = self.base_model
        else:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.config.model,
                config=model_config,
            )

    def load_token_classification_model(self, num_labels: int, label_names: Optional[List[str]] = None):
        logger.info(f"Loading token classification model: {self.config.model}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model)

        if label_names:
            model_config = AutoConfig.from_pretrained(
                self.config.model,
                id2label={str(i): name for i, name in enumerate(label_names)},
                label2id={name: i for i, name in enumerate(label_names)},
            )
        else:
            model_config = AutoConfig.from_pretrained(
                self.config.model,
                num_labels=num_labels,
            )

        self.model = AutoModelForTokenClassification.from_pretrained(
            self.config.model,
            config=model_config,
        )

        if self.config.freeze_base:
            for param in self.model.base_model.parameters():
                param.requires_grad = False
            logger.info("Frozen base model parameters")

    def preprocess_classification(self, dataset: Dataset, config: DatasetConfig) -> Dataset:
        def tokenize(examples):
            return self.tokenizer(
                examples[config.text_column],
                truncation=True,
                max_length=config.max_length,
            )

        return dataset.map(tokenize, batched=True)

    def preprocess_pair_classification(self, dataset: Dataset, config: DatasetConfig) -> Dataset:
        def tokenize(examples):
            return self.tokenizer(
                examples[config.text1_column],
                examples[config.text2_column],
                truncation=True,
                max_length=config.max_length,
            )

        return dataset.map(tokenize, batched=True)

    def preprocess_token_classification(self, dataset: Dataset, config: DatasetConfig) -> Dataset:
        def tokenize_and_align_labels(examples):
            tokenized_inputs = self.tokenizer(
                examples[config.tokens_column],
                truncation=True,
                max_length=config.max_length,
                is_split_into_words=True,
            )

            labels = []
            for i, label in enumerate(examples[config.tags_column]):
                word_ids = tokenized_inputs.word_ids(batch_index=i)
                previous_word_idx = None
                label_ids = []
                for word_idx in word_ids:
                    if word_idx is None:
                        label_ids.append(-100)
                    elif word_idx != previous_word_idx:
                        label_ids.append(label[word_idx])
                    else:
                        label_ids.append(-100)
                    previous_word_idx = word_idx
                labels.append(label_ids)

            tokenized_inputs["labels"] = labels
            return tokenized_inputs

        return dataset.map(tokenize_and_align_labels, batched=True)

    def compute_metrics_classification(self, eval_pred):
        metric = load_metric("accuracy")
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return metric.compute(predictions=predictions, references=labels)

  
    def compute_metrics_token_classification(self, eval_pred):
        metric = load_metric("seqeval")
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=2)

        true_predictions = [
            [self.model.config.id2label[str(p)] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [self.model.config.id2label[str(l)] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        results = metric.compute(predictions=true_predictions, references=true_labels)
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }

    def run_dataset(self, config: DatasetConfig) -> Dict[str, Any]:
        logger.info(f"Running dataset: {config.name}")

        # Load dataset once to get label information
        dataset_dict = load_dataset(config.name, name=config.subset) if config.subset else load_dataset(config.name)
        if isinstance(dataset_dict, Dataset):
            dataset_dict = DatasetDict({"train": dataset_dict, "validation": dataset_dict})

        train_dataset = dataset_dict[config.train_split]
        eval_dataset = dataset_dict[config.eval_split]

        # Determine the correct label column based on task type
        label_column = config.tags_column if config.type == "token_classification" else config.label_column

        label_names = None
        # Handle sequence features (like token classification)
        if hasattr(train_dataset.features[label_column], "feature"):
            # This is a Sequence feature
            if hasattr(train_dataset.features[label_column].feature, "names"):
                label_names = train_dataset.features[label_column].feature.names
                num_labels = len(label_names)
            else:
                # Flatten all label sequences to get unique labels
                all_labels = []
                for label_seq in train_dataset[label_column]:
                    all_labels.extend(label_seq)
                num_labels = len(set(all_labels))
        elif hasattr(train_dataset.features[label_column], "names"):
            # Regular ClassLabel feature
            label_names = train_dataset.features[label_column].names
            num_labels = len(label_names)
        else:
            # No names, infer from data
            if config.type == "token_classification":
                # Flatten all label sequences to get unique labels
                all_labels = []
                for label_seq in train_dataset[label_column]:
                    all_labels.extend(label_seq)
                num_labels = len(set(all_labels))
            else:
                num_labels = len(set(train_dataset[label_column]))

        runs = config.runs or self.config.runs
        all_metrics = []

        for run in range(runs):
            logger.info(f"Run {run + 1}/{runs} for {config.name}")

            set_seed(self.config.seed + run)

            # Reload dataset for each run to avoid modification issues
            dataset_dict = load_dataset(config.name, name=config.subset) if config.subset else load_dataset(config.name)
            if isinstance(dataset_dict, Dataset):
                dataset_dict = DatasetDict({"train": dataset_dict, "validation": dataset_dict})

            train_dataset = dataset_dict[config.train_split]
            eval_dataset = dataset_dict[config.eval_split]

            if config.type == "token_classification":
                self.load_token_classification_model(num_labels, label_names)
                train_dataset = self.preprocess_token_classification(train_dataset, config)
                eval_dataset = self.preprocess_token_classification(eval_dataset, config)
                compute_metrics = self.compute_metrics_token_classification
                data_collator = DataCollatorForTokenClassification(self.tokenizer, pad_to_multiple_of=8)
            elif config.type == "pair_classification":
                self.load_model_and_tokenizer(num_labels, label_names)
                train_dataset = self.preprocess_pair_classification(train_dataset, config)
                eval_dataset = self.preprocess_pair_classification(eval_dataset, config)
                compute_metrics = self.compute_metrics_classification
                data_collator = DataCollatorWithPadding(self.tokenizer)
            else:
                self.load_model_and_tokenizer(num_labels, label_names)
                train_dataset = self.preprocess_classification(train_dataset, config)
                eval_dataset = self.preprocess_classification(eval_dataset, config)
                compute_metrics = self.compute_metrics_classification
                data_collator = DataCollatorWithPadding(self.tokenizer)

            # Remove only the text columns that were actually used
            text_columns = []
            if config.text_column and config.text_column in train_dataset.column_names:
                text_columns.append(config.text_column)
            if config.text1_column and config.text1_column in train_dataset.column_names:
                text_columns.append(config.text1_column)
            if config.text2_column and config.text2_column in train_dataset.column_names:
                text_columns.append(config.text2_column)
            if config.tokens_column and config.tokens_column in train_dataset.column_names:
                text_columns.append(config.tokens_column)
            if config.tags_column and config.tags_column in train_dataset.column_names:
                text_columns.append(config.tags_column)

            for col in text_columns:
                if col in train_dataset.column_names:
                    train_dataset = train_dataset.remove_columns(col)
                if col in eval_dataset.column_names:
                    eval_dataset = eval_dataset.remove_columns(col)

            # Remove token_type_ids if model doesn't support it
            if "token_type_ids" in train_dataset.column_names:
                train_dataset = train_dataset.remove_columns(["token_type_ids"])
            if "token_type_ids" in eval_dataset.column_names:
                eval_dataset = eval_dataset.remove_columns(["token_type_ids"])

            training_args = TrainingArguments(
                output_dir=f"{self.config.output_dir}/{config.name}_run_{run}",
                learning_rate=config.learning_rate,
                per_device_train_batch_size=config.batch_size,
                per_device_eval_batch_size=config.batch_size,
                num_train_epochs=config.epochs,
                eval_strategy="no",
                save_strategy="no",
                logging_strategy="no",
                report_to="none",
                disable_tqdm=True,
            )

            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                compute_metrics=compute_metrics,
                data_collator=data_collator,
                processing_class=self.tokenizer,
            )

            trainer.train()
            metrics = trainer.evaluate()
            all_metrics.append(metrics)

        avg_metrics = {}
        for key in all_metrics[0].keys():
            values = [m[key] for m in all_metrics]
            avg_metrics[f"{key}_mean"] = np.mean(values)
            avg_metrics[f"{key}_std"] = np.std(values)

        result = {
            "dataset": config.name,
            "type": config.type,
            "num_labels": num_labels,
            "runs": runs,
            "metrics": avg_metrics,
        }

        self.save_result(result)
        return result

    def save_result(self, result: Dict[str, Any]):
        output_file = Path(self.config.output_dir) / "results.jsonl"
        with open(output_file, "a") as f:
            f.write(json.dumps(result) + "\n")

    def run_all(self):
        results = []

        for dataset_config in self.config.datasets:
            result = self.run_dataset(dataset_config)
            results.append(result)

        self.display_results(results)
        return results

    def display_results(self, results: List[Dict[str, Any]]):
        table = Table(title="Evaluation Results")
        table.add_column("Dataset", style="cyan", no_wrap=True)
        table.add_column("Type", style="magenta")
        table.add_column("Metric", style="green")
        table.add_column("Mean ± Std", style="yellow")

        for result in results:
            metrics = result["metrics"]
            for key, value in metrics.items():
                if key.endswith("_mean"):
                    metric_name = key[:-5]
                    std_value = metrics.get(f"{metric_name}_std", 0)
                    table.add_row(
                        result["dataset"],
                        result["type"],
                        metric_name,
                        f"{value:.4f} ± {std_value:.4f}",
                    )

        console.print(table)


def load_config(config_path: str) -> EvalConfig:
    with open(config_path, "r") as f:
        data = yaml.safe_load(f)
    return EvalConfig(**data)


def create_example_config():
    config = {
        "model": "prajjwal1/bert-tiny",
        "freeze_base": True,
        "runs": 3,
        "seed": 42,
        "output_dir": "results",
        "datasets": [
            {
                "name": "iastate/onestop_english",
                "type": "classification",
                "train_split": "train",
                "eval_split": "test",
                "text_column": "text",
                "label_column": "label",
                "max_length": 128,
                "epochs": 3,
                "learning_rate": 2e-5,
                "batch_size": 16,
                "runs": 5,
            },
            {
                "name": "conll2003",
                "type": "token_classification",
                "train_split": "train",
                "eval_split": "validation",
                "tokens_column": "tokens",
                "tags_column": "ner_tags",
                "max_length": 128,
                "epochs": 5,
                "learning_rate": 5e-5,
                "batch_size": 32,
            },
            {
                "name": "nyu-mll/glue",
                "type": "pair_classification",
                "subset": "mnli",
                "train_split": "train",
                "eval_split": "validation_matched",
                "text1_column": "premise",
                "text2_column": "hypothesis",
                "label_column": "label",
                "max_length": 128,
                "epochs": 3,
                "learning_rate": 2e-5,
                "batch_size": 16,
            },
        ],
    }
    return yaml.dump(config, default_flow_style=False)


def main():
    if len(sys.argv) < 2:
        console.print("[red]Usage: encoder_eval.py <config.yaml> [--runs N][/red]")
        console.print("[cyan]Create example config: encoder_eval.py --create-config > config.yaml[/cyan]")
        sys.exit(1)

    if sys.argv[1] == "--create-config":
        console.print(create_example_config())
        sys.exit(0)

    config_path = sys.argv[1]
    config = load_config(config_path)

    if "--runs" in sys.argv:
        idx = sys.argv.index("--runs")
        if idx + 1 < len(sys.argv):
            config.runs = int(sys.argv[idx + 1])

    runner = TaskRunner(config)
    runner.run_all()


if __name__ == "__main__":
    main()