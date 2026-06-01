"""
Utility functions for the Hybrid Search System
"""
import json
import logging
import re
import time
import yaml
from functools import wraps
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

import numpy as np


def setup_logger(name: str, level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    if not logger.handlers:
        h = logging.StreamHandler()
        h.setFormatter(
            logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        logger.addHandler(h)
    return logger


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def save_json(data: Any, filepath: str, indent: int = 2) -> None:
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False, default=str)


def load_json(filepath: str) -> Any:
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def clean_text(text: str, lowercase: bool = True, remove_extra_spaces: bool = True) -> str:
    if not text:
        return ""
    text = str(text)
    if lowercase:
        text = text.lower()
    if remove_extra_spaces:
        text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize_text(text: str, method: str = "simple") -> List[str]:
    if method == "simple":
        return re.findall(r"\b\w+\b", text.lower())
    if method == "whitespace":
        return text.lower().split()
    raise ValueError(f"Unknown tokenization method: {method}")


def timing_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        t0 = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - t0
        logging.getLogger(func.__module__).debug(
            f"{func.__name__} took {elapsed:.4f}s"
        )
        return result
    return wrapper


def batch_iterator(items: List[Any], batch_size: int) -> Generator:
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def normalize_scores(scores: np.ndarray) -> np.ndarray:
    if len(scores) == 0:
        return scores
    lo, hi = scores.min(), scores.max()
    return np.ones_like(scores) if hi == lo else (scores - lo) / (hi - lo)


def format_search_results(
    results: List[Dict[str, Any]], max_body_length: int = 200
) -> str:
    lines = []
    for i, r in enumerate(results, 1):
        lines += [
            f"\n{'='*70}",
            f"Rank {i} | Score: {r.get('score', 0):.4f}",
            f"{'='*70}",
            f"Title   : {r.get('title', 'N/A')}",
            f"Category: {r.get('category', 'N/A')} | Difficulty: {r.get('difficulty', 'N/A')}",
            f"Body    : {str(r.get('body', ''))[:max_body_length]}{'…' if len(str(r.get('body','')))>max_body_length else ''}",
        ]
        if "tags" in r:
            lines.append(f"Tags    : {r['tags']}")
    return "\n".join(lines)


def validate_search_input(query: str, top_k: int) -> None:
    if not query or not query.strip():
        raise ValueError("Query cannot be empty")
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if top_k > 1000:
        raise ValueError("top_k cannot exceed 1000")


def create_directory(path: str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


class ProgressTracker:
    def __init__(self, total: int, description: str = "Processing"):
        self.total = total
        self.current = 0
        self.description = description
        self.start_time = time.time()

    def update(self, n: int = 1) -> None:
        self.current += n
        elapsed = time.time() - self.start_time
        pct = self.current / self.total * 100
        rate = self.current / elapsed if elapsed else 0
        print(
            f"\r{self.description}: {self.current}/{self.total} ({pct:.1f}%) | {rate:.1f} it/s",
            end="",
            flush=True,
        )
        if self.current >= self.total:
            print()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        if self.current < self.total:
            print()


def compute_metrics(
    predictions: List[List[str]],
    ground_truth: List[List[str]],
    k: int = 5,
) -> Dict[str, float]:
    """Legacy metric helper – kept for backwards-compat."""
    if not predictions or not ground_truth:
        return {"precision@k": 0.0, "recall@k": 0.0, "f1@k": 0.0, "mrr": 0.0}

    precs, recs, rrs = [], [], []
    for pred, truth in zip(predictions, ground_truth):
        pred_k = pred[:k]
        truth_set = set(truth)
        pred_set = set(pred_k)
        if pred_k:
            precs.append(len(pred_set & truth_set) / len(pred_k))
        if truth_set:
            recs.append(len(pred_set & truth_set) / len(truth_set))
        for i, d in enumerate(pred_k, 1):
            if d in truth_set:
                rrs.append(1.0 / i)
                break
        else:
            rrs.append(0.0)

    p = float(np.mean(precs)) if precs else 0.0
    r = float(np.mean(recs)) if recs else 0.0
    return {
        "precision@k": p,
        "recall@k": r,
        "f1@k": 2 * p * r / (p + r) if (p + r) else 0.0,
        "mrr": float(np.mean(rrs)) if rrs else 0.0,
    }
