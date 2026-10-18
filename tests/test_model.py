import unittest

from sklearn.dummy import DummyClassifier

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from src.model import (
    LABELS,
    bootstrap_macro_f1_differences,
    build_model_searches,
    export_dataset_csvs,
    make_pipeline,
)
from app import dashboard_summary, dataset_examples


class SentimentPipelineTests(unittest.TestCase):
    def test_searches_use_stratified_cross_validation(self):
        searches = build_model_searches(cv_folds=4)
        self.assertEqual(set(searches), {
            "majority_baseline",
            "multinomial_nb",
            "logistic_regression",
            "linear_svm",
        })
        for search in searches.values():
            self.assertEqual(search.cv.n_splits, 4)
            self.assertEqual(search.scoring, "f1_macro")

    def test_cross_validation_rejects_one_fold(self):
        with self.assertRaises(ValueError):
            build_model_searches(cv_folds=1)

    def test_dataset_exports_combined_and_split_csvs(self):
        splits = {
            "train": (["great"], np.array([2])),
            "validation": (["okay"], np.array([1])),
            "test": (["awful"], np.array([0])),
        }
        with TemporaryDirectory() as directory:
            output = Path(directory)
            sizes = export_dataset_csvs(splits, output)
            self.assertEqual(sizes, {"train": 1, "validation": 1, "test": 1})
            combined = pd.read_csv(output / "data.csv")
            self.assertEqual(len(combined), 3)
            self.assertEqual(set(combined["split"]), {"train", "validation", "test"})
            self.assertEqual(pd.read_csv(output / "test.csv").loc[0, "sentiment"], "negative")

    def test_dashboard_reads_saved_metrics_and_confusion_matrix(self):
        summary = dashboard_summary()
        self.assertEqual(summary["split_sizes"]["train"], 45615)
        self.assertEqual(summary["split_sizes"]["test"], 12284)
        self.assertEqual(len(summary["confusion_matrix"]), 3)
        self.assertEqual(sum(map(sum, summary["confusion_matrix"])), 12284)
        self.assertEqual(summary["selected_model"], "linear_svm")

    def test_dataset_explorer_filters_and_pages(self):
        first_page = dataset_examples("test", "positive", "", page=1)
        self.assertEqual(len(first_page["examples"]), 12)
        self.assertGreater(first_page["total"], 12)
        filtered = dataset_examples("test", "all", "placeholder-query-not-in-dataset", page=1)
        self.assertEqual(filtered["total"], 0)

    def test_pipeline_fits_three_classes(self):
        texts = [
            "I love this great product",
            "This is an awful terrible product",
            "This product is okay",
            "great and wonderful",
            "terrible and awful",
            "it is an ordinary item",
        ]
        labels = [2, 0, 1, 2, 0, 1]
        model = make_pipeline(DummyClassifier(strategy="most_frequent"))
        model.fit(texts, labels)
        self.assertEqual(model.predict(["ordinary item"])[0], 0)
        self.assertEqual(len(LABELS), 3)
        feature_names = model.named_steps["tfidf"].get_feature_names_out()
        self.assertTrue(any(name.startswith("character__") for name in feature_names))

    def test_bootstrap_comparison_is_paired_and_reproducible(self):
        truth = np.array([0, 1, 2, 0, 1, 2])
        selected = np.array([0, 1, 2, 0, 1, 2])
        weaker = np.array([1, 1, 2, 0, 0, 2])
        first = bootstrap_macro_f1_differences(
            truth, selected, {"weaker": weaker}, repetitions=50, seed=7
        )
        second = bootstrap_macro_f1_differences(
            truth, selected, {"weaker": weaker}, repetitions=50, seed=7
        )
        self.assertAlmostEqual(
            first.loc[0, "selected_minus_comparison_macro_f1"], 1 / 3
        )
        self.assertEqual(
            first.loc[0, "paired_bootstrap_ci_2_5"],
            second.loc[0, "paired_bootstrap_ci_2_5"],
        )


if __name__ == "__main__":
    unittest.main()
