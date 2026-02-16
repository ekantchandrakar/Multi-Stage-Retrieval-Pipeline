"""
Utility functions for the Hybrid Search System
"""
import re
import time
import yaml
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from functools import wraps

import numpy as np


def setup_logger(name: str, level: str = "INFO") -> logging.Logger:
    """
    Set up a logger with consistent formatting.
    
    Args:
        name: Logger name
        level: Logging level
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load YAML configuration file.
    
    Args:
        config_path: Path to YAML config file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_json(data: Any, filepath: str, indent: int = 2) -> None:
    """
    Save data to JSON file.
    
    Args:
        data: Data to save
        filepath: Output file path
        indent: JSON indentation
    """
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def load_json(filepath: str) -> Any:
    """
    Load data from JSON file.
    
    Args:
        filepath: Input file path
        
    Returns:
        Loaded data
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def clean_text(text: str, lowercase: bool = True, 
               remove_extra_spaces: bool = True) -> str:
    """
    Clean and normalize text.
    
    Args:
        text: Input text
        lowercase: Convert to lowercase
        remove_extra_spaces: Remove extra whitespace
        
    Returns:
        Cleaned text
    """
    if not text:
        return ""
    
    # Convert to string if not already
    text = str(text)
    
    # Lowercase
    if lowercase:
        text = text.lower()
    
    # Remove extra spaces
    if remove_extra_spaces:
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
    
    return text


def tokenize_text(text: str, method: str = "simple") -> List[str]:
    """
    Tokenize text into words.
    
    Args:
        text: Input text
        method: Tokenization method ("simple" or "whitespace")
        
    Returns:
        List of tokens
    """
    if method == "simple":
        # Simple word tokenization
        tokens = re.findall(r'\b\w+\b', text.lower())
    elif method == "whitespace":
        # Whitespace tokenization
        tokens = text.lower().split()
    else:
        raise ValueError(f"Unknown tokenization method: {method}")
    
    return tokens


def compute_metrics(predictions: List[List[str]], 
                   ground_truth: List[List[str]], 
                   k: int = 5) -> Dict[str, float]:
    """
    Compute evaluation metrics for search results.
    
    Args:
        predictions: List of predicted document IDs for each query
        ground_truth: List of relevant document IDs for each query
        k: Top-k to evaluate
        
    Returns:
        Dictionary of metrics
    """
    metrics = {
        'precision@k': 0.0,
        'recall@k': 0.0,
        'f1@k': 0.0,
        'mrr': 0.0,  # Mean Reciprocal Rank
    }
    
    if not predictions or not ground_truth:
        return metrics
    
    precisions = []
    recalls = []
    reciprocal_ranks = []
    
    for pred, truth in zip(predictions, ground_truth):
        pred_k = pred[:k]
        truth_set = set(truth)
        pred_set = set(pred_k)
        
        # Precision@k
        if pred_k:
            precision = len(pred_set & truth_set) / len(pred_k)
            precisions.append(precision)
        
        # Recall@k
        if truth_set:
            recall = len(pred_set & truth_set) / len(truth_set)
            recalls.append(recall)
        
        # MRR
        for i, doc_id in enumerate(pred_k, 1):
            if doc_id in truth_set:
                reciprocal_ranks.append(1.0 / i)
                break
        else:
            reciprocal_ranks.append(0.0)
    
    metrics['precision@k'] = np.mean(precisions) if precisions else 0.0
    metrics['recall@k'] = np.mean(recalls) if recalls else 0.0
    
    # F1 score
    p, r = metrics['precision@k'], metrics['recall@k']
    metrics['f1@k'] = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    
    metrics['mrr'] = np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0
    
    return metrics


def timing_decorator(func):
    """
    Decorator to measure function execution time.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed_time = time.time() - start_time
        
        logger = logging.getLogger(func.__module__)
        logger.debug(f"{func.__name__} took {elapsed_time:.4f} seconds")
        
        return result
    return wrapper


def batch_iterator(items: List[Any], batch_size: int):
    """
    Create batches from a list of items.
    
    Args:
        items: List of items
        batch_size: Size of each batch
        
    Yields:
        Batches of items
    """
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


def normalize_scores(scores: np.ndarray) -> np.ndarray:
    """
    Normalize scores to [0, 1] range.
    
    Args:
        scores: Array of scores
        
    Returns:
        Normalized scores
    """
    if len(scores) == 0:
        return scores
    
    min_score = scores.min()
    max_score = scores.max()
    
    if max_score == min_score:
        return np.ones_like(scores)
    
    return (scores - min_score) / (max_score - min_score)


def format_search_results(results: List[Dict[str, Any]], 
                         max_body_length: int = 200) -> str:
    """
    Format search results for display.
    
    Args:
        results: List of search result dictionaries
        max_body_length: Maximum length of body text to display
        
    Returns:
        Formatted string
    """
    output = []
    
    for i, result in enumerate(results, 1):
        output.append(f"\n{'='*80}")
        output.append(f"Rank {i} | Score: {result.get('score', 0):.4f}")
        output.append(f"{'='*80}")
        output.append(f"Title: {result.get('title', 'N/A')}")
        output.append(f"Category: {result.get('category', 'N/A')} | "
                     f"Difficulty: {result.get('difficulty', 'N/A')}")
        
        body = result.get('body', '')
        if len(body) > max_body_length:
            body = body[:max_body_length] + "..."
        output.append(f"Body: {body}")
        
        if 'tags' in result:
            output.append(f"Tags: {result['tags']}")
        
    return '\n'.join(output)


def create_directory(path: str) -> Path:
    """
    Create directory if it doesn't exist.
    
    Args:
        path: Directory path
        
    Returns:
        Path object
    """
    dir_path = Path(path)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def validate_search_input(query: str, top_k: int) -> None:
    """
    Validate search input parameters.
    
    Args:
        query: Search query
        top_k: Number of results to return
        
    Raises:
        ValueError: If inputs are invalid
    """
    if not query or not query.strip():
        raise ValueError("Query cannot be empty")
    
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    
    if top_k > 1000:
        raise ValueError("top_k cannot exceed 1000")


class ProgressTracker:
    """Simple progress tracker for long operations."""
    
    def __init__(self, total: int, description: str = "Processing"):
        self.total = total
        self.current = 0
        self.description = description
        self.start_time = time.time()
    
    def update(self, n: int = 1):
        """Update progress by n steps."""
        self.current += n
        self._display()
    
    def _display(self):
        """Display current progress."""
        elapsed = time.time() - self.start_time
        percent = (self.current / self.total) * 100
        rate = self.current / elapsed if elapsed > 0 else 0
        
        print(f"\r{self.description}: {self.current}/{self.total} "
              f"({percent:.1f}%) | {rate:.1f} it/s", end='')
        
        if self.current >= self.total:
            print()  # New line when complete
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        if self.current < self.total:
            print()  # Ensure newline on early exit
