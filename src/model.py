from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
DATA_DIR = ROOT / "data"
LABELS = ["negative", "neutral", "positive"]


def load_splits() -> dict[str, tuple[list[str], np.ndarray]]:
    dataset = load_dataset("cardiffnlp/tweet_eval", "sentiment")
    required_splits = {"train", "validation", "test"}
    if not required_splits.issubset(dataset):
        raise ValueError(f"TweetEval is missing required splits: {required_splits - set(dataset)}")

    splits: dict[str, tuple[list[str], np.ndarray]] = {}
    for name in sorted(required_splits):
        split = dataset[name]
        if "text" not in split.column_names or "label" not in split.column_names:
            raise ValueError(f"Unexpected TweetEval columns in {name}: {split.column_names}")
        texts = [str(text) for text in split["text"]]
        labels = np.asarray(split["label"], dtype=int)
        if not texts or len(texts) != len(labels):
            raise ValueError(f"TweetEval {name} split is empty or malformed")
        if not set(np.unique(labels)).issubset({0, 1, 2}):
            raise ValueError(f"Unexpected class labels in {name}: {np.unique(labels)}")
        splits[name] = texts, labels
    return splits


def export_dataset_csvs(
    splits: dict[str, tuple[list[str], np.ndarray]], output_dir: Path = DATA_DIR
) -> dict[str, int]:
    required_splits = {"train", "validation", "test"}
    if not required_splits.issubset(splits):
        raise ValueError(f"Missing dataset splits: {required_splits - set(splits)}")
    output_dir.mkdir(parents=True, exist_ok=True)
    frames: dict[str, pd.DataFrame] = {}
    for split_name in ("train", "validation", "test"):
        texts, labels = splits[split_name]
        if len(texts) != len(labels):
            raise ValueError(f"Mismatched text and label counts in {split_name}")
        frames[split_name] = pd.DataFrame(
            {
                "text": texts,
                "label": labels.astype(int),
                "sentiment": [LABELS[int(label)] for label in labels],
                "split": split_name,
            }
        )
        frames[split_name].to_csv(output_dir / f"{split_name}.csv", index=False)

    combined = pd.concat(frames.values(), ignore_index=True)
    combined.to_csv(output_dir / "data.csv", index=False)
    return {name: len(frame) for name, frame in frames.items()}


def make_pipeline(classifier: Any) -> Pipeline:
    return Pipeline(
        [
            (
                "tfidf",
                FeatureUnion(
                    [
                        (
                            "word",
                            TfidfVectorizer(
                                lowercase=True,
                                strip_accents="unicode",
                                ngram_range=(1, 2),
                                min_df=2,
                                max_features=100_000,
                                sublinear_tf=True,
                            ),
                        ),
                        (
                            "character",
                            TfidfVectorizer(
                                analyzer="char_wb",
                                ngram_range=(3, 5),
                                min_df=3,
                                max_features=100_000,
                                sublinear_tf=True,
                            ),
                        ),
                    ]
                ),
            ),
            ("classifier", classifier),
        ]
    )


def build_model_searches(cv_folds: int = 3) -> dict[str, GridSearchCV]:
    if cv_folds < 2:
        raise ValueError("Cross-validation requires at least two folds")
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    return {
        "majority_baseline": GridSearchCV(
            Pipeline([("classifier", DummyClassifier(strategy="most_frequent"))]),
            param_grid={},
            scoring="f1_macro",
            cv=cv,
            n_jobs=1,
            refit=True,
        ),
        "multinomial_nb": GridSearchCV(
            make_pipeline(MultinomialNB()),
            param_grid={"classifier__alpha": [0.1, 0.5, 1.0]},
            scoring="f1_macro",
            cv=cv,
            n_jobs=1,
            refit=True,
        ),
        "logistic_regression": GridSearchCV(
            make_pipeline(
                LogisticRegression(
                    max_iter=1000, class_weight="balanced", random_state=42
                )
            ),
            param_grid={"classifier__C": [0.5, 1.0, 2.0]},
            scoring="f1_macro",
            cv=cv,
            n_jobs=1,
            refit=True,
        ),
        "linear_svm": GridSearchCV(
            make_pipeline(LinearSVC(class_weight="balanced", random_state=42)),
            param_grid={"classifier__C": [0.25, 0.5, 1.0]},
            scoring="f1_macro",
            cv=cv,
            n_jobs=1,
            refit=True,
        ),
    }


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    precision, recall, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(macro_f1),
    }


def bootstrap_macro_f1_differences(
    y_true: np.ndarray,
    selected_predictions: np.ndarray,
    comparison_predictions: dict[str, np.ndarray],
    repetitions: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    sample_count = len(y_true)
    differences: dict[str, list[float]] = {name: [] for name in comparison_predictions}
    for _ in range(repetitions):
        indices = rng.integers(0, sample_count, size=sample_count)
        selected_score = f1_score(
            y_true[indices], selected_predictions[indices], labels=[0, 1, 2],
            average="macro", zero_division=0,
        )
        for name, predictions in comparison_predictions.items():
            comparison_score = f1_score(
                y_true[indices], predictions[indices], labels=[0, 1, 2],
                average="macro", zero_division=0,
            )
            differences[name].append(float(selected_score - comparison_score))

    rows = []
    for name, values in differences.items():
        lower, upper = np.quantile(values, [0.025, 0.975])
        rows.append(
            {
                "comparison_model": name,
                "selected_minus_comparison_macro_f1": float(
                    f1_score(
                        y_true, selected_predictions, labels=[0, 1, 2],
                        average="macro", zero_division=0,
                    )
                    - f1_score(
                        y_true, comparison_predictions[name], labels=[0, 1, 2],
                        average="macro", zero_division=0,
                    )
                ),
                "paired_bootstrap_ci_2_5": float(lower),
                "paired_bootstrap_ci_97_5": float(upper),
                "repetitions": repetitions,
                "ci_excludes_zero": bool(lower > 0 or upper < 0),
            }
        )
    return pd.DataFrame(rows)


def _write_explanations(model: Pipeline, output_dir: Path) -> None:
    classifier = model.named_steps["classifier"]
    if not hasattr(classifier, "coef_"):
        raise ValueError("The selected classifier does not expose linear feature coefficients")
    terms = model.named_steps["tfidf"].get_feature_names_out()
    rows: list[dict[str, Any]] = []
    for class_index, class_name in enumerate(classifier.classes_):
        weights = classifier.coef_[class_index]
        for term_index in np.argsort(weights)[-20:][::-1]:
            rows.append(
                {
                    "class": LABELS[int(class_name)],
                    "term": terms[term_index],
                    "coefficient": float(weights[term_index]),
                }
            )
    pd.DataFrame(rows).to_csv(output_dir / "logistic_regression_top_terms.csv", index=False)


def _write_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, output_dir: Path) -> None:
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set(
        xticks=range(3),
        yticks=range(3),
        xticklabels=LABELS,
        yticklabels=LABELS,
        xlabel="Predicted label",
        ylabel="True label",
        title="TweetEval sentiment test confusion matrix",
    )
    for (row, col), value in np.ndenumerate(matrix):
        ax.text(col, row, str(value), ha="center", va="center")
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=160)
    plt.close(fig)


def _write_eda(splits: dict[str, tuple[list[str], np.ndarray]], output_dir: Path) -> None:
    class_rows = []
    for split_name, (_, labels) in splits.items():
        counts = np.bincount(labels, minlength=len(LABELS))
        for label_id, count in enumerate(counts):
            class_rows.append(
                {
                    "split": split_name,
                    "label": LABELS[label_id],
                    "count": int(count),
                    "proportion": float(count / len(labels)),
                }
            )
    distribution = pd.DataFrame(class_rows)
    distribution.to_csv(output_dir / "class_distribution.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    for split_name in ("train", "validation", "test"):
        values = distribution[distribution["split"] == split_name]
        ax.plot(values["label"], values["proportion"], marker="o", label=split_name)
    ax.set(
        xlabel="Sentiment class",
        ylabel="Proportion of split",
        title="TweetEval sentiment class proportions",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "class_distribution.png", dpi=160)
    plt.close(fig)

    lengths = np.fromiter((len(text) for text in splits["train"][0]), dtype=int)
    pd.DataFrame({"character_length": lengths}).describe().to_csv(
        output_dir / "training_text_length_summary.csv"
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(lengths, bins=40, color="steelblue", edgecolor="white")
    ax.set(
        xlabel="Characters per tweet",
        ylabel="Number of tweets",
        title="TweetEval training text length distribution",
    )
    fig.tight_layout()
    fig.savefig(output_dir / "training_text_lengths.png", dpi=160)
    plt.close(fig)


def train(output_dir: Path = ARTIFACTS) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    splits = load_splits()
    export_dataset_csvs(splits)
    _write_eda(splits, output_dir)
    train_texts, y_train = splits["train"]
    validation_texts, y_validation = splits["validation"]
    test_texts, y_test = splits["test"]

    results: dict[str, Any] = {}
    test_predictions_by_model: dict[str, np.ndarray] = {}
    cv_folds = 3
    searches = build_model_searches(cv_folds)
    cv_rows: list[dict[str, Any]] = []
    for name, search in searches.items():
        search.fit(train_texts, y_train)
        model = search.best_estimator_
        validation_predictions = model.predict(validation_texts)
        test_predictions_by_model[name] = model.predict(test_texts)
        results[name] = {
            "cross_validation": {
                "folds": cv_folds,
                "best_macro_f1": float(search.best_score_),
                "best_macro_f1_std": float(
                    search.cv_results_["std_test_score"][search.best_index_]
                ),
                "best_params": search.best_params_,
            },
            "validation": _metrics(y_validation, validation_predictions),
            "test": _metrics(y_test, test_predictions_by_model[name]),
        }
        for candidate_index, candidate_params in enumerate(search.cv_results_["params"]):
            cv_rows.append(
                {
                    "model": name,
                    "params": json.dumps(candidate_params, sort_keys=True),
                    "mean_macro_f1": float(
                        search.cv_results_["mean_test_score"][candidate_index]
                    ),
                    "std_macro_f1": float(
                        search.cv_results_["std_test_score"][candidate_index]
                    ),
                    **{
                        f"fold_{fold_index}_macro_f1": float(
                            search.cv_results_[f"split{fold_index}_test_score"][candidate_index]
                        )
                        for fold_index in range(cv_folds)
                    },
                }
            )

    selected_name = max(
        results, key=lambda name: (results[name]["validation"]["macro_f1"], name)
    )
    selected_model = searches[selected_name].best_estimator_
    test_predictions = test_predictions_by_model[selected_name]
    comparison_predictions = {
        name: predictions
        for name, predictions in test_predictions_by_model.items()
        if name != selected_name
    }
    results["selected_model"] = selected_name
    results["split_sizes"] = {
        name: len(values[1]) for name, values in splits.items()
    }
    results["labels"] = LABELS
    results["selection_metric"] = (
        "mean cross-validation macro-F1 tunes each model; validation macro-F1 "
        "selects the model family; test split is held out for final evaluation"
    )
    pd.DataFrame(cv_rows).to_csv(output_dir / "cross_validation_results.csv", index=False)

    pd.DataFrame(
        [
            {"model": name, **scores["validation"], **{f"test_{k}": v for k, v in scores["test"].items()}}
            for name, scores in results.items()
            if (
                isinstance(scores, dict)
                and isinstance(scores.get("validation"), dict)
                and isinstance(scores.get("test"), dict)
            )
        ]
    ).to_csv(output_dir / "metrics.csv", index=False)
    (output_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (output_dir / "selected_model_test_report.json").write_text(
        json.dumps(
            classification_report(
                y_test, test_predictions, target_names=LABELS, output_dict=True, zero_division=0
            ),
            indent=2,
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        {
            "text": test_texts,
            "true_label": [LABELS[label] for label in y_test],
            "predicted_label": [LABELS[label] for label in test_predictions],
        }
    ).to_csv(output_dir / "test_predictions.csv", index=False)
    bootstrap_macro_f1_differences(
        y_test, test_predictions, comparison_predictions
    ).to_csv(output_dir / "paired_bootstrap_macro_f1.csv", index=False)
    joblib.dump(selected_model, output_dir / "sentiment_model.joblib")
    _write_confusion_matrix(y_test, test_predictions, output_dir)
    _write_explanations(searches["logistic_regression"].best_estimator_, output_dir)
    return results


def predict(text: str, model_path: Path = ARTIFACTS / "sentiment_model.joblib") -> str:
    if not text.strip():
        raise ValueError("Input text must not be empty")
    model = joblib.load(model_path)
    label = int(model.predict([text])[0])
    return LABELS[label]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train or run the TweetEval sentiment prototype.")
    parser.add_argument("--predict", help="Classify one input sentence instead of training")
    parser.add_argument("--model", type=Path, default=ARTIFACTS / "sentiment_model.joblib")
    args = parser.parse_args()
    if args.predict is not None:
        print(f"Sentiment: {predict(args.predict, args.model)}")
    else:
        print(json.dumps(train(), indent=2))


if __name__ == "__main__":
    main()
