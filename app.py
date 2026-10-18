from __future__ import annotations

import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import joblib
import pandas as pd

from src.model import ARTIFACTS, DATA_DIR, LABELS

ROOT = Path(__file__).resolve().parent
WEB_DIR = ROOT / "web"
HOST = os.environ.get("HOST", "127.0.0.1")
PORT = int(os.environ.get("PORT", "8000"))


def _read_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Required project result file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Required project data file not found: {path}")
    return pd.read_csv(path)


def dashboard_summary() -> dict:
    results = _read_json(ARTIFACTS / "results.json")
    report = _read_json(ARTIFACTS / "selected_model_test_report.json")
    cv_results = _read_csv(ARTIFACTS / "cross_validation_results.csv")
    metrics = _read_csv(ARTIFACTS / "metrics.csv")
    split_counts: dict[str, dict[str, int]] = {}
    split_sizes: dict[str, int] = {}
    for split_name in ("train", "validation", "test"):
        dataset = _read_csv(DATA_DIR / f"{split_name}.csv")
        split_sizes[split_name] = len(dataset)
        split_counts[split_name] = {
            label: int((dataset["sentiment"] == label).sum()) for label in LABELS
        }

    predictions = _read_csv(ARTIFACTS / "test_predictions.csv")
    confusion = []
    for truth in LABELS:
        confusion.append(
            [
                int(
                    (
                        (predictions["true_label"] == truth)
                        & (predictions["predicted_label"] == predicted)
                    ).sum()
                )
                for predicted in LABELS
            ]
        )

    cv = []
    for _, row in cv_results.iterrows():
        cv.append(
            {
                "model": str(row["model"]),
                "params": json.loads(str(row["params"])),
                "mean_macro_f1": float(row["mean_macro_f1"]),
                "std_macro_f1": float(row["std_macro_f1"]),
                "fold_scores": [
                    float(row[f"fold_{fold}_macro_f1"]) for fold in range(3)
                ],
            }
        )

    model_scores = []
    for _, row in metrics.iterrows():
        model_scores.append(
            {
                "model": str(row["model"]),
                "validation_accuracy": float(row["accuracy"]),
                "validation_macro_f1": float(row["macro_f1"]),
                "test_accuracy": float(row["test_accuracy"]),
                "test_macro_f1": float(row["test_macro_f1"]),
            }
        )

    return {
        "labels": LABELS,
        "split_sizes": split_sizes,
        "split_counts": split_counts,
        "selected_model": results["selected_model"],
        "selection_metric": results["selection_metric"],
        "model_scores": model_scores,
        "cross_validation": cv,
        "test_report": report,
        "confusion_matrix": confusion,
    }


def dataset_examples(
    split: str, sentiment: str, query: str, page: int, page_size: int = 12
) -> dict:
    if split not in {"train", "validation", "test"}:
        raise ValueError("split must be train, validation, or test")
    if sentiment not in {"all", *LABELS}:
        raise ValueError("sentiment must be all, negative, neutral, or positive")
    if page < 1:
        raise ValueError("page must be a positive integer")

    frame = _read_csv(DATA_DIR / f"{split}.csv")
    if sentiment != "all":
        frame = frame[frame["sentiment"] == sentiment]
    if query:
        frame = frame[frame["text"].str.contains(query, case=False, regex=False)]
    total = len(frame)
    start = (page - 1) * page_size
    examples = frame.iloc[start : start + page_size]
    return {
        "split": split,
        "sentiment": sentiment,
        "query": query,
        "page": page,
        "page_size": page_size,
        "total": total,
        "examples": examples[["text", "sentiment"]].to_dict(orient="records"),
    }


class DashboardHandler(BaseHTTPRequestHandler):
    model = None

    def _send(self, status: int, content_type: str, content: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(content)

    def _send_json(self, status: int, payload: dict) -> None:
        content = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self._send(status, "application/json; charset=utf-8", content)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/health":
            status = 200 if self.model is not None else 503
            self._send_json(
                status,
                {"status": "ok" if status == 200 else "model_unavailable"},
            )
            return
        if parsed.path == "/api/summary":
            self._send_json(200, dashboard_summary())
            return
        if parsed.path == "/api/examples":
            params = parse_qs(parsed.query)
            try:
                payload = dataset_examples(
                    split=params.get("split", ["test"])[0],
                    sentiment=params.get("sentiment", ["all"])[0],
                    query=params.get("q", [""])[0].strip(),
                    page=int(params.get("page", ["1"])[0]),
                )
            except (ValueError, FileNotFoundError) as error:
                self._send_json(400, {"error": str(error)})
                return
            self._send_json(200, payload)
            return

        relative_path = "index.html" if parsed.path == "/" else parsed.path.lstrip("/")
        file_path = (WEB_DIR / relative_path).resolve()
        if WEB_DIR.resolve() not in file_path.parents or not file_path.is_file():
            self._send_json(404, {"error": "Page or asset not found"})
            return
        content_type = {
            ".html": "text/html; charset=utf-8",
            ".css": "text/css; charset=utf-8",
            ".js": "text/javascript; charset=utf-8",
        }.get(file_path.suffix, "application/octet-stream")
        self._send(200, content_type, file_path.read_bytes())

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/api/predict":
            self._send_json(404, {"error": "Endpoint not found"})
            return
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._send_json(400, {"error": "Invalid request length"})
            return
        if content_length < 1 or content_length > 16_384:
            self._send_json(413, {"error": "Text must be between 1 and 16,384 bytes"})
            return
        try:
            payload = json.loads(self.rfile.read(content_length))
        except (json.JSONDecodeError, UnicodeDecodeError):
            self._send_json(400, {"error": "Request body must be valid JSON"})
            return
        text = payload.get("text") if isinstance(payload, dict) else None
        if not isinstance(text, str) or not text.strip():
            self._send_json(400, {"error": "Enter non-empty text to classify"})
            return
        if self.model is None:
            self._send_json(503, {"error": "Saved model is not available. Run python -m src.model first."})
            return

        label_id = int(self.model.predict([text.strip()])[0])
        if label_id not in range(len(LABELS)):
            raise ValueError(f"Model returned an invalid class index: {label_id}")
        self._send_json(
            200,
            {"text": text.strip(), "sentiment": LABELS[label_id]},
        )

    def log_message(self, format: str, *args: object) -> None:
        print(f"{self.log_date_time_string()} {self.address_string()} {format % args}")


def main() -> None:
    model_path = ARTIFACTS / "sentiment_model.joblib"
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Trained model not found at {model_path}; run `python -m src.model` first."
        )
    DashboardHandler.model = joblib.load(model_path)
    server = ThreadingHTTPServer((HOST, PORT), DashboardHandler)
    print(f"Tweet sentiment dashboard: http://{HOST}:{PORT}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping dashboard.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
