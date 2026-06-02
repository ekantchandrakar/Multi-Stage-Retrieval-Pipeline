"""
Dataset Loader for StaQC (HuggingFace Datasets Server API)
===========================================================
Uses GET https://datasets-server.huggingface.co/rows
No datasets library, no Parquet dependency for download.
Cache is written as CSV (always works on Windows).
"""

import csv
import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests

logger = logging.getLogger(__name__)

_ROWS_URL  = "https://datasets-server.huggingface.co/rows"
_PAGE_SIZE = 100

STAQC_CONFIGS = {
    "man_python":  ["man_python"],
    "man_sql":     ["man_sql"],
    "mca_python":  ["mca_python"],
    "all_python":  ["man_python", "mca_python"],
    "all":         ["man_python", "mca_python", "man_sql"],
    "python":      ["mca_python"],
    "sql":         ["man_sql"],
}

DATASET = "koutch/staqc"


def _fetch_rows(config: str, limit: Optional[int]) -> List[Dict[str, Any]]:
    """Page through /rows for one StaQC config."""
    rows: List[Dict[str, Any]] = []
    offset = 0
    total: Optional[int] = None

    while True:
        want = _PAGE_SIZE
        if limit is not None:
            want = min(_PAGE_SIZE, limit - len(rows))
            if want <= 0:
                break

        params = {
            "dataset": DATASET,
            "config":  config,
            "split":   "train",
            "offset":  offset,
            "length":  want,
        }

        try:
            resp = requests.get(_ROWS_URL, params=params, timeout=30)
            resp.raise_for_status()
        except requests.RequestException as exc:
            logger.error(f"Request failed (config={config}, offset={offset}): {exc}")
            break

        data = resp.json()

        if total is None:
            total = data.get("num_rows_total") or data.get("total") or 0
            cap   = min(limit, total) if limit else total
            logger.info(f"    {config}: {total:,} rows available, fetching up to {cap:,}")

        page = data.get("rows", [])
        if not page:
            break

        for item in page:
            rows.append(item.get("row", item))

        offset += len(page)
        logger.info(f"    Fetched {len(rows):,} / {min(limit, total) if limit else total:,}")

        if total and offset >= total:
            break
        if limit and len(rows) >= limit:
            break

        time.sleep(0.05)

    return rows


def _parse_tags(raw) -> str:
    if isinstance(raw, list):
        return ", ".join(str(t) for t in raw)
    if isinstance(raw, str):
        raw = raw.strip()
        if raw.startswith("["):
            try:
                return ", ".join(str(t) for t in json.loads(raw.replace("'", '"')))
            except Exception:
                pass
        return raw
    return ""


def _infer_category(tags: str, config: str) -> str:
    t = tags.lower()
    if any(k in t for k in ("list","dict","array","tree","graph","heap","stack","queue")):
        return "Data Structures"
    if any(k in t for k in ("sort","search","algorithm","recursion","dynamic")):
        return "Algorithms"
    if any(k in t for k in ("sql","database","query","join","index","transaction")):
        return "Databases"
    if any(k in t for k in ("django","flask","api","rest","http","request")):
        return "Backend Systems"
    if any(k in t for k in ("docker","kubernetes","aws","cloud","deploy")):
        return "DevOps & Cloud"
    return "Python" if "python" in config else ("SQL" if "sql" in config else "General")


def _infer_difficulty(body: str) -> str:
    n = len(str(body))
    if n < 200: return "Beginner"
    if n < 600: return "Intermediate"
    return "Advanced"


def _normalise_rows(rows: List[Dict], config: str) -> List[Dict]:
    """
    Convert raw API row dicts → canonical dicts.
    Pure Python — no pandas, no pyarrow, no chance of hanging.
    """
    out = []
    for i, row in enumerate(rows):
        # id
        doc_id = str(row.get("question_id") or row.get("id") or row.get("qid") or f"{config}_{i}")

        # title
        title = str(
            row.get("question") or row.get("title") or row.get("Question") or row.get("text") or ""
        ).strip()

        # body
        body = str(
            row.get("code_snippet") or row.get("body") or row.get("answer") or
            row.get("code") or row.get("snippet") or ""
        ).strip()

        # tags
        tags = _parse_tags(row.get("tags", config))

        combined = (title + " " + body).strip()
        if len(combined) <= 10:
            continue

        out.append({
            "id":           doc_id,
            "title":        title,
            "body":         body,
            "tags":         tags,
            "category":     _infer_category(tags, config),
            "difficulty":   _infer_difficulty(body),
            "combined_text": combined,
            "source_config": config,
        })

    return out


class StaQCDatasetLoader:
    """
    Downloads koutch/staqc via Datasets Server REST API.
    Caches locally as CSV (always works on Windows, no pyarrow needed).
    """

    def __init__(
        self,
        data_dir:    str  = "data",
        language:    str  = "mca_python",
        max_records: int  = 100_000,
        cache:       bool = True,
    ):
        if language not in STAQC_CONFIGS:
            raise ValueError(f"language must be one of {sorted(STAQC_CONFIGS)}, got '{language}'")

        self.language    = language
        self.configs     = STAQC_CONFIGS[language]
        self.data_dir    = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.max_records = max_records if max_records > 0 else None
        self.cache       = cache

    def load(self) -> List[Dict[str, Any]]:
        """
        Returns a list of canonical dicts (not a DataFrame).
        Uses CSV cache so pyarrow is never involved.
        """
        cap        = self.max_records or "full"
        cache_path = self.data_dir / f"staqc_{self.language}_{cap}.csv"

        if self.cache and cache_path.exists():
            logger.info(f"Loading from cache: {cache_path}")
            records = []
            with open(cache_path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    records.append(row)
            logger.info(f"Loaded {len(records):,} records from CSV cache")
            return records

        logger.info(
            f"Downloading {DATASET} configs={self.configs} "
            f"(target: {self.max_records or 'all'} records) …"
        )

        all_records: List[Dict] = []
        remaining = self.max_records

        for cfg in self.configs:
            if remaining is not None and remaining <= 0:
                break
            logger.info(f"  → Fetching config: {cfg}")
            raw_rows = _fetch_rows(cfg, limit=remaining)
            if not raw_rows:
                logger.warning(f"  No rows returned for config: {cfg}")
                continue
            records = _normalise_rows(raw_rows, cfg)
            logger.info(f"  Normalised {cfg}: {len(records):,} docs")
            all_records.extend(records)
            if remaining is not None:
                remaining -= len(records)

        if not all_records:
            raise RuntimeError(
                f"No data downloaded for {DATASET} / {self.configs}.\n"
                "Check your internet connection."
            )

        if self.max_records:
            all_records = all_records[: self.max_records]

        logger.info(f"Total records: {len(all_records):,}")

        if self.cache and all_records:
            logger.info(f"Writing CSV cache → {cache_path}")
            fieldnames = list(all_records[0].keys())
            with open(cache_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(all_records)
            logger.info(f"Cache written: {cache_path}")

        return all_records


class DataProcessor:
    """
    Unified DataProcessor.
      data_path ending in '.csv'  → legacy CSV
      anything else               → StaQC Datasets Server API
    """

    def __init__(
        self,
        data_path:   str,
        language:    str = "mca_python",
        max_records: int = 100_000,
        cache_dir:   str = "data",
    ):
        self.data_path   = data_path
        self.language    = language
        self.max_records = max_records
        self.cache_dir   = cache_dir
        self._use_staqc  = not str(data_path).endswith(".csv")

    def preprocess_data(
        self,
        combine_fields: Optional[List[str]] = None,
        clean: bool = True,
    ) -> Tuple[List[str], List[Dict[str, Any]]]:
        """
        Load data, return (corpus, metadata).
        Called by search_engine.build_indices() — must always complete.
        """
        if self._use_staqc:
            loader = StaQCDatasetLoader(
                data_dir=self.cache_dir,
                language=self.language,
                max_records=self.max_records,
            )
            records = loader.load()   # returns List[Dict] — pure Python, no pandas hang

            corpus   = [r["combined_text"] for r in records]
            metadata = []
            for idx, r in enumerate(records):
                metadata.append({
                    "id":         str(r.get("id", idx)),
                    "category":   str(r.get("category", "")),
                    "difficulty": str(r.get("difficulty", "")),
                    "title":      str(r.get("title", "")),
                    "body":       str(r.get("body", "")),
                    "tags":       str(r.get("tags", "")),
                    "index":      idx,
                })

        else:
            logger.info(f"Loading legacy CSV: {self.data_path}")
            df = pd.read_csv(self.data_path)

            combine_fields = combine_fields or ["title", "body"]
            corpus, metadata = [], []
            for idx, row in df.iterrows():
                parts = [str(row[f]) for f in combine_fields if f in row and pd.notna(row[f])]
                text = " ".join(parts)
                if clean:
                    text = re.sub(r"\s+", " ", text).strip().lower()
                corpus.append(text)
                metadata.append({
                    "id":         str(row.get("id", idx)),
                    "category":   str(row.get("category", "")),
                    "difficulty": str(row.get("difficulty", "")),
                    "title":      str(row.get("title", "")),
                    "body":       str(row.get("body", "")),
                    "tags":       str(row.get("tags", "")),
                    "index":      int(idx),
                })

        logger.info(f"Preprocessed {len(corpus):,} documents")
        return corpus, metadata

    def load_data(self) -> pd.DataFrame:
        corpus, metadata = self.preprocess_data()
        return pd.DataFrame(metadata)

    def get_corpus(self) -> List[str]:
        corpus, _ = self.preprocess_data()
        return corpus

    def get_metadata(self) -> List[Dict[str, Any]]:
        _, metadata = self.preprocess_data()
        return metadata

    def get_document_by_index(self, index: int) -> Dict[str, Any]:
        _, metadata = self.preprocess_data()
        return metadata[index]

    def get_statistics(self) -> Dict[str, Any]:
        _, metadata = self.preprocess_data()
        cats   = {}
        diffs  = {}
        for m in metadata:
            cats[m["category"]]   = cats.get(m["category"], 0) + 1
            diffs[m["difficulty"]] = diffs.get(m["difficulty"], 0) + 1
        return {
            "total_documents":  len(metadata),
            "categories":       cats,
            "difficulty_levels": diffs,
        }

    def __len__(self) -> int:
        corpus, _ = self.preprocess_data()
        return len(corpus)

    def __getitem__(self, i: int) -> Dict[str, Any]:
        return self.get_document_by_index(i)