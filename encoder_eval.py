#!/usr/bin/env python3
import argparse
import hashlib
import json
import logging
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import yaml
from datasets import Dataset, DatasetDict, load_dataset, concatenate_datasets
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
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


console = Console()
logger = logging.getLogger(__name__)


class DatasetConfig:
    name: str
    type: str
    subset: Optional[str | List[str]] = None
    train_split: str = "train"
    eval_split: str = "validation"
    text_column: str = "text"
    label_column: str = "label"
    text1_column: Optional[str] = None
    text2_column: Optional[str] = None
    tokens_column: Optional[str] = None
    tags_column: Optional[str] = None
    epochs: int = 3
    learning_rate: float = 2e-5
    batch_size: int = 16
    runs: Optional[int] = None
    metric_name: Optional[str] = None

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if k in ["learning_rate", "batch_size", "epochs", "runs", "seed"]:
                setattr(self, k, float(v) if k == "learning_rate" else int(v))
            else:
                setattr(self, k, v)


class EvalConfig:
    model: str
    freeze_base: bool = True
    runs: int = 3
    seed: int = 42
    output_dir: str = "results"
    max_length: Optional[int] = None
    datasets: List[DatasetConfig] = []

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if k == "datasets":
                self.datasets = [DatasetConfig(**d) for d in v]
            elif k in ["runs", "seed", "max_length"]:
                setattr(self, k, int(v) if v is not None else None)
            else:
                setattr(self, k, v)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for hashing."""
        return {
            "model": self.model,
            "freeze_base": self.freeze_base,
            "runs": self.runs,
            "seed": self.seed,
            "datasets": [
                {k: v for k, v in d.__dict__.items() if not k.startswith('_')}
                for d in self.datasets
            ],
        }

    def get_hash(self) -> str:
        """Generate SHA256 hash of the config."""
        config_str = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]

    def get_model_short_name(self) -> str:
        """Extract short model name from path."""
        model_path = self.model.rstrip('/')

        if Path(model_path).exists():
            folder_name = Path(model_path).name
            return f"localhost__{folder_name}"
        else:
            return model_path.replace('/', '__')


class TaskRunner:
    def __init__(self, config: EvalConfig):
        self.config = config
        self.tokenizer = None
        self.model = None
        self.config_hash = config.get_hash()
        self.model_short_name = config.get_model_short_name()

        self.unique_output_dir = Path(config.output_dir) / self.model_short_name / self.config_hash
        self.results_file = self.unique_output_dir / "results.jsonl"
        self.config_file = self.unique_output_dir / "config.yaml"

        set_seed(config.seed)
        self.unique_output_dir.mkdir(parents=True, exist_ok=True)

        if not self.config_file.exists():
            self.config_file.write_text(yaml.dump(config.to_dict(), default_flow_style=False))

        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            level=logging.INFO,
        )

        logger.info(f"Output directory: {self.unique_output_dir}")
        logger.info(f"Config hash: {self.config_hash}")

    def ensure_float_labels(self, dataset: Dataset, label_column: str) -> Dataset:
        """Convert label column to float32 for regression tasks."""
        from datasets import Value
        new_features = dataset.features.copy()
        new_features[label_column] = Value('float32')
        return dataset.cast(new_features)

    def load_model_and_tokenizer(self, num_labels: int, label_names: Optional[List[str]] = None, problem_type: Optional[str] = None):
        logger.info(f"Loading model and tokenizer: {self.config.model}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model)

        config_kwargs = {}
        if label_names:
            config_kwargs.update({
                "id2label": {str(i): name for i, name in enumerate(label_names)},
                "label2id": {name: i for i, name in enumerate(label_names)},
            })
        else:
            config_kwargs["num_labels"] = num_labels

        if problem_type:
            config_kwargs["problem_type"] = problem_type

        model_config = AutoConfig.from_pretrained(
            self.config.model,
            **config_kwargs
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

        self.set_effective_max_length()

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

        self.set_effective_max_length()

    def set_effective_max_length(self):
        if self.config.max_length is None:
            if hasattr(self.model.config, 'max_position_embeddings'):
                self.config.max_length = self.model.config.max_position_embeddings
            elif hasattr(self.tokenizer, 'model_max_length') and self.tokenizer.model_max_length > 0:
                self.config.max_length = self.tokenizer.model_max_length
            else:
                self.config.max_length = 512

    def preprocess_classification(self, dataset: Dataset, config: DatasetConfig) -> Dataset:
        if not hasattr(dataset.features[config.label_column], "names"):
            unique_labels = sorted(set(dataset[config.label_column]))
            if all(isinstance(label, str) for label in unique_labels):
                label_to_id = {label: i for i, label in enumerate(unique_labels)}

                def convert_labels(examples):
                    return {"labels": [label_to_id[label] for label in examples[config.label_column]]}

                dataset = dataset.map(convert_labels, batched=True)

        def tokenize(examples):
            return self.tokenizer(
                examples[config.text_column],
                truncation=True,
                max_length=self.config.max_length,
            )

        return dataset.map(tokenize, batched=True)

    def preprocess_pair_classification(self, dataset: Dataset, config: DatasetConfig) -> Dataset:
        def tokenize(examples):
            return self.tokenizer(
                examples[config.text1_column],
                examples[config.text2_column],
                truncation=True,
                max_length=self.config.max_length,
            )

        return dataset.map(tokenize, batched=True)

    def preprocess_token_classification(self, dataset: Dataset, config: DatasetConfig) -> Dataset:
        if not hasattr(dataset.features[config.tags_column], "feature"):
            all_labels = set()
            for label_seq in dataset[config.tags_column]:
                all_labels.update(label_seq)

            if all(isinstance(label, str) for label in all_labels):
                label_to_id = {label: i for i, label in enumerate(sorted(all_labels))}
            else:
                label_to_id = None
        else:
            label_to_id = None

        def tokenize_and_align_labels(examples):
            tokenized_inputs = self.tokenizer(
                examples[config.tokens_column],
                truncation=True,
                max_length=self.config.max_length,
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
                        if label_to_id is not None:
                            label_ids.append(label_to_id[label[word_idx]])
                        else:
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

    def compute_metrics_regression(self, eval_pred):
        predictions, labels = eval_pred
        predictions = predictions.squeeze()
        labels = labels.squeeze()

        mse = mean_squared_error(labels, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(labels, predictions)
        r2 = r2_score(labels, predictions)

        return {
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
        }


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
        subset_str = f" ({config.subset})" if config.subset else ""
        logger.info(f"Running dataset: {config.name}{subset_str}")

        # Load dataset(s) once to get label information
        if config.subset:
            if isinstance(config.subset, list):
                train_datasets = []
                eval_datasets = []
                for subset in config.subset:
                    dataset_dict = load_dataset(config.name, name=subset)
                    if isinstance(dataset_dict, Dataset):
                        dataset_dict = DatasetDict({"train": dataset_dict, "validation": dataset_dict})
                    train_datasets.append(dataset_dict[config.train_split])
                    eval_datasets.append(dataset_dict[config.eval_split])
                train_dataset = concatenate_datasets(train_datasets)
                eval_dataset = concatenate_datasets(eval_datasets)
            else:
                dataset_dict = load_dataset(config.name, name=config.subset)
                if isinstance(dataset_dict, Dataset):
                    dataset_dict = DatasetDict({"train": dataset_dict, "validation": dataset_dict})
                train_dataset = dataset_dict[config.train_split]
                eval_dataset = dataset_dict[config.eval_split]
        else:
            dataset_dict = load_dataset(config.name)
            if isinstance(dataset_dict, Dataset):
                dataset_dict = DatasetDict({"train": dataset_dict, "validation": dataset_dict})
            train_dataset = dataset_dict[config.train_split]
            eval_dataset = dataset_dict[config.eval_split]

        label_column = config.tags_column if config.type == "token_classification" else config.label_column

        if config.type == "regression":
            num_labels = 1
            label_names = None
        else:
            label_names = None
            if hasattr(train_dataset.features[label_column], "feature"):
                if hasattr(train_dataset.features[label_column].feature, "names"):
                    label_names = train_dataset.features[label_column].feature.names
                    num_labels = len(label_names)
                else:
                    all_labels = []
                    for label_seq in train_dataset[label_column]:
                        all_labels.extend(label_seq)
                    num_labels = len(set(all_labels))
            elif hasattr(train_dataset.features[label_column], "names"):
                label_names = train_dataset.features[label_column].names
                num_labels = len(label_names)
            else:
                if config.type == "token_classification":
                    all_labels = []
                    for label_seq in train_dataset[label_column]:
                        all_labels.extend(label_seq)
                    num_labels = len(set(all_labels))
                else:
                    num_labels = len(set(train_dataset[label_column]))

        runs = config.runs or self.config.runs
        completed_runs = self.check_completed_runs(config.name, config.type)

        if completed_runs > 0:
            if completed_runs >= runs:
                logger.info(f"All {runs} runs already completed for {config.name}")
                return self.load_final_result(config.name, config.type)
            logger.info(f"Resuming from run {completed_runs + 1}/{runs} for {config.name}")

        all_metrics = self.load_existing_metrics(config.name, config.type) if completed_runs > 0 else []

        for run in range(completed_runs, runs):
            logger.info(f"Run {run + 1}/{runs} for {config.name}")

            set_seed(self.config.seed + run)

            if config.type == "token_classification":
                self.load_token_classification_model(num_labels, label_names)
                train_dataset_run = self.preprocess_token_classification(train_dataset, config)
                eval_dataset_run = self.preprocess_token_classification(eval_dataset, config)
                compute_metrics = self.compute_metrics_token_classification
                data_collator = DataCollatorForTokenClassification(self.tokenizer, pad_to_multiple_of=8)
            elif config.type == "pair_classification":
                self.load_model_and_tokenizer(num_labels, label_names)
                train_dataset_run = self.preprocess_pair_classification(train_dataset, config)
                eval_dataset_run = self.preprocess_pair_classification(eval_dataset, config)
                compute_metrics = self.compute_metrics_classification
                data_collator = DataCollatorWithPadding(self.tokenizer)
            elif config.type == "regression":
                self.load_model_and_tokenizer(num_labels, label_names, problem_type="regression")
                train_dataset_run = self.preprocess_classification(train_dataset, config)
                eval_dataset_run = self.preprocess_classification(eval_dataset, config)
                compute_metrics = self.compute_metrics_regression
                data_collator = DataCollatorWithPadding(self.tokenizer)
            else:
                self.load_model_and_tokenizer(num_labels, label_names)
                train_dataset_run = self.preprocess_classification(train_dataset, config)
                eval_dataset_run = self.preprocess_classification(eval_dataset, config)
                compute_metrics = self.compute_metrics_classification
                data_collator = DataCollatorWithPadding(self.tokenizer)

            text_columns = []
            if config.text_column and config.text_column in train_dataset_run.column_names:
                text_columns.append(config.text_column)
            if config.text1_column and config.text1_column in train_dataset_run.column_names:
                text_columns.append(config.text1_column)
            if config.text2_column and config.text2_column in train_dataset_run.column_names:
                text_columns.append(config.text2_column)
            if config.tokens_column and config.tokens_column in train_dataset_run.column_names:
                text_columns.append(config.tokens_column)
            if config.tags_column and config.tags_column in train_dataset_run.column_names:
                text_columns.append(config.tags_column)

            for col in text_columns:
                if col in train_dataset_run.column_names:
                    train_dataset_run = train_dataset_run.remove_columns(col)
                if col in eval_dataset_run.column_names:
                    eval_dataset_run = eval_dataset_run.remove_columns(col)

            training_args = TrainingArguments(
                output_dir=f"{self.unique_output_dir}/{config.name}_run_{run}",
                learning_rate=config.learning_rate,
                per_device_train_batch_size=config.batch_size,
                per_device_eval_batch_size=config.batch_size,
                num_train_epochs=config.epochs,
                eval_strategy="no",
                save_strategy="no",
                logging_strategy="no",
                report_to="none",
                torch_compile=True,
            )

            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset_run,
                eval_dataset=eval_dataset_run,
                compute_metrics=compute_metrics,
                data_collator=data_collator,
                processing_class=self.tokenizer,
            )

            trainer.train()
            metrics = trainer.evaluate()
            all_metrics.append(metrics)

            run_result = {
                "dataset": config.name,
                "type": config.type,
                "num_labels": num_labels,
                "run": run + 1,
                "total_runs": runs,
                "metrics": metrics,
                "model": self.config.model,
                "config_hash": self.config_hash,
                "timestamp": time.time(),
            }
            self.save_run_result(run_result)

        avg_metrics = {}
        for key in all_metrics[0].keys():
            values = [m[key] for m in all_metrics]
            avg_metrics[f"{key}_mean"] = np.mean(values)
            avg_metrics[f"{key}_std"] = np.std(values)

        main_score = self.get_main_score(config.type, avg_metrics)
        avg_metrics["main_score"] = main_score["name"]
        avg_metrics["main_score_mean"] = main_score["mean"]
        avg_metrics["main_score_std"] = main_score["std"]

        result = {
            "dataset": config.name,
            "type": config.type,
            "num_labels": num_labels,
            "runs": runs,
            "metrics": avg_metrics,
            "model": self.config.model,
            "config_hash": self.config_hash,
            "timestamp": time.time(),
        }

        self.save_final_result(result)
        return result

    def check_completed_runs(self, dataset_name: str, task_type: str) -> int:
        if not self.results_file.exists():
            return 0

        completed_runs = 0
        with open(self.results_file) as f:
            for line in f:
                result = json.loads(line.strip())
                if (result.get("dataset") == dataset_name and
                    result.get("type") == task_type and
                    "run" in result):
                    completed_runs = max(completed_runs, result["run"])

        return completed_runs

    def load_existing_metrics(self, dataset_name: str, task_type: str) -> List[Dict[str, Any]]:
        if not self.results_file.exists():
            return []

        metrics = []
        with open(self.results_file) as f:
            for line in f:
                result = json.loads(line.strip())
                if (result.get("dataset") == dataset_name and
                    result.get("type") == task_type and
                    "run" in result):
                    metrics.append(result["metrics"])

        return metrics

    def load_final_result(self, dataset_name: str, task_type: str) -> Dict[str, Any]:
        if not self.results_file.exists():
            return {}

        with open(self.results_file) as f:
            lines = f.readlines()

        for line in reversed(lines):
            result = json.loads(line.strip())
            if (result.get("dataset") == dataset_name and
                result.get("type") == task_type and
                "run" not in result):
                return result

        return {}

    def save_run_result(self, result: Dict[str, Any]):
        with open(self.results_file, "a") as f:
            f.write(json.dumps(result) + "\n")

    def save_final_result(self, result: Dict[str, Any]):
        with open(self.results_file, "a") as f:
            f.write(json.dumps(result) + "\n")

    def get_main_score(self, task_type: str, metrics: Dict[str, Any]) -> Dict[str, Any]:
        if task_type == "classification" or task_type == "pair_classification" or task_type == "token_classification":
            score_name = "accuracy"
        elif task_type == "regression":
            score_name = "r2"
        else:
            score_name = list(metrics.keys())[0].replace("_mean", "")

        mean_key = f"eval_{score_name}_mean"
        std_key = f"eval_{score_name}_std"

        if mean_key not in metrics:
            mean_key = f"{score_name}_mean"
            std_key = f"{score_name}_std"

        return {
            "name": score_name,
            "mean": float(metrics.get(mean_key, 0)),
            "std": float(metrics.get(std_key, 0)),
        }

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
        table.add_column("Main Score", style="green")
        table.add_column("Mean ± Std", style="yellow")

        for result in results:
            metrics = result["metrics"]
            if "main_score_mean" in metrics:
                score_name = metrics.get("main_score", "score")
                mean_val = metrics["main_score_mean"]
                std_val = metrics["main_score_std"]
                table.add_row(
                    result["dataset"],
                    result["type"],
                    score_name,
                    f"{mean_val:.4f} ± {std_val:.4f}",
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
            {
                "name": "nhull/125-tripadvisor-reviews",
                "type": "regression",
                "train_split": "train",
                "eval_split": "train",
                "text_column": "text",
                "label_column": "label",
                "max_length": 128,
                "epochs": 5,
                "learning_rate": 5e-5,
                "batch_size": 16,
            },
        ],
    }
    return yaml.dump(config, default_flow_style=False)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate encoder models on various tasks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Examples:\n"
               "  encoder_eval.py config.yaml\n"
               "  encoder_eval.py config.yaml --model bert-base-uncased --runs 5\n"
               "  encoder_eval.py --create-config > config.yaml"
    )

    parser.add_argument("config", nargs="?", help="Path to config YAML file")
    parser.add_argument("--model", help="Override model name")
    parser.add_argument("--max-length", type=int, help="Override maximum sequence length")
    parser.add_argument("--runs", type=int, help="Override number of runs")
    parser.add_argument("--seed", type=int, help="Override random seed")
    parser.add_argument("--output-dir", help="Override output directory")
    parser.add_argument("--freeze-base", action="store_true", help="Freeze base model parameters")
    parser.add_argument("--no-freeze-base", action="store_true", help="Don't freeze base model parameters")
    parser.add_argument("--no-resume", action="store_true", help="Don't resume from existing results")
    parser.add_argument("--create-config", action="store_true", help="Create example config")

    args = parser.parse_args()

    if args.create_config:
        console.print(create_example_config())
        sys.exit(0)

    if not args.config:
        parser.error("config file is required unless --create-config is specified")

    config = load_config(args.config)

    if args.model:
        config.model = args.model
    if args.max_length:
        config.max_length = args.max_length
    if args.runs:
        config.runs = args.runs
    if args.seed:
        config.seed = args.seed
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.freeze_base:
        config.freeze_base = True
    elif args.no_freeze_base:
        config.freeze_base = False

    runner = TaskRunner(config)

    if args.no_resume and runner.results_file.exists():
        backup_file = runner.results_file.with_suffix('.jsonl.bak')
        runner.results_file.rename(backup_file)
        logger.info(f"Existing results backed up to {backup_file}")

    runner.run_all()


if __name__ == "__main__":
    main()