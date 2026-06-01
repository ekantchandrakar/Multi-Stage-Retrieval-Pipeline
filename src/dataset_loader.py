"""
Dataset Loader for StaQC (HuggingFace Datasets Server API)
===========================================================
The koutch/staqc dataset only has a legacy staqc.py script — no Parquet
files, so `datasets.load_dataset` and HfFileSystem both fail on modern
library versions.

Fix: use the HuggingFace **Datasets Server REST API** which streams rows
directly as JSON, with zero dependency on the datasets library scripts.

API used:
  GET https://datasets-server.huggingface.co/rows
      ?dataset=koutch%2Fstaqc
      &config=mca_python
      &split=train
      &offset=0
      &length=100          (max 100 per request)

Docs: https://huggingface.co/docs/datasets-server
"""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests

logger = logging.getLogger(__name__)

# HuggingFace Datasets Server endpoint
_ROWS_URL   = "https://datasets-server.huggingface.co/rows"
_SPLITS_URL = "https://datasets-server.huggingface.co/splits"

# Max rows the API returns per request (hard limit = 100)
_PAGE_SIZE  = 100

# Valid StaQC config names
STAQC_CONFIGS = {
    "man_python": "man_python",
    "man_sql":    "man_sql",
    "mca_python": "mca_python",
    # convenience aliases
    "python":     "man_python",
    "sql":        "man_sql",
}


class StaQCDatasetLoader:
    """
    Downloads koutch/staqc via the HuggingFace Datasets Server REST API.

    No `datasets` library, no Parquet files, no legacy scripts.
    Data is fetched page-by-page (100 rows/request) and cached locally
    as Parquet after the first download.
    """

    DATASET = "koutch/staqc"

    def __init__(
        self,
        data_dir: str   = "data",
        language: str   = "python",
        max_records: int = 100_000,
        cache: bool      = True,
        request_delay: float = 0.1,   # seconds between pages (be polite)
    ):
        if language not in STAQC_CONFIGS:
            raise ValueError(
                f"language must be one of {list(STAQC_CONFIGS)}, got '{language}'"
            )
        self.config      = STAQC_CONFIGS[language]
        self.language    = language
        self.data_dir    = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.max_records = max_records if max_records > 0 else None
        self.cache       = cache
        self.delay       = request_delay
        self._df: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self) -> pd.DataFrame:
        cap = self.max_records or "full"
        cache_path = self.data_dir / f"staqc_{self.config}_{cap}.parquet"

        if self.cache and cache_path.exists():
            logger.info(f"Loading from cache: {cache_path}")
            self._df = pd.read_parquet(cache_path)
            logger.info(f"Loaded {len(self._df):,} records from cache")
            return self._df

        logger.info(
            f"Downloading {self.DATASET}/{self.config} via Datasets Server API …"
        )
        rows = self._fetch_all_rows()

        if not rows:
            raise RuntimeError(
                f"No rows returned from Datasets Server for "
                f"{self.DATASET} / {self.config}. "
                f"Check https://datasets-server.huggingface.co/splits?dataset=koutch%2Fstaqc"
            )

        raw_df   = pd.DataFrame(rows)
        self._df = self._normalise(raw_df)

        if self.cache:
            self._df.to_parquet(cache_path, index=False)
            logger.info(f"Cached {len(self._df):,} records → {cache_path}")

        return self._df

    # ------------------------------------------------------------------
    # Core fetcher
    # ------------------------------------------------------------------

    def _fetch_all_rows(self) -> List[Dict[str, Any]]:
        """
        Page through the Datasets Server /rows endpoint until we have
        max_records rows (or the dataset is exhausted).
        """
        all_rows: List[Dict[str, Any]] = []
        offset = 0
        total_in_dataset: Optional[int] = None

        while True:
            want = _PAGE_SIZE
            if self.max_records:
                want = min(_PAGE_SIZE, self.max_records - len(all_rows))
                if want <= 0:
                    break

            params = {
                "dataset": self.DATASET,
                "config":  self.config,
                "split":   "train",
                "offset":  offset,
                "length":  want,
            }

            try:
                resp = requests.get(_ROWS_URL, params=params, timeout=30)
                resp.raise_for_status()
            except requests.RequestException as exc:
                logger.error(f"Request failed at offset={offset}: {exc}")
                break

            data = resp.json()

            # First page: grab total row count
            if total_in_dataset is None:
                total_in_dataset = (
                    data.get("num_rows_total")
                    or data.get("total")
                    or 0
                )
                cap = self.max_records or total_in_dataset
                logger.info(
                    f"Dataset has {total_in_dataset:,} rows total; "
                    f"fetching up to {cap:,}"
                )

            page_rows = data.get("rows", [])
            if not page_rows:
                break   # end of dataset

            # The API wraps each row: {"row_idx": N, "row": {...}, ...}
            for item in page_rows:
                all_rows.append(item.get("row", item))

            fetched = len(all_rows)
            logger.info(
                f"  Fetched {fetched:,} rows "
                f"(offset={offset}, page={len(page_rows)})"
            )

            offset += len(page_rows)

            # Stop if we've reached the end or the cap
            if total_in_dataset and offset >= total_in_dataset:
                break
            if self.max_records and fetched >= self.max_records:
                break

            time.sleep(self.delay)

        logger.info(f"Download complete: {len(all_rows):,} rows fetched")
        return all_rows

    # ------------------------------------------------------------------
    # Normalisation
    # ------------------------------------------------------------------

    def _normalise(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Map raw StaQC columns → canonical schema used by the search engine.

        Raw columns vary by config but typically include:
          question_id, question, code_snippet, tags
        """
        logger.info(f"Raw columns: {list(df.columns)}")
        out = pd.DataFrame()

        # id
        id_col = next(
            (c for c in ("question_id", "id", "qid") if c in df.columns), None
        )
        out["id"] = (
            df[id_col].astype(str) if id_col
            else pd.Series([str(i) for i in range(len(df))])
        )

        # title  (the question text)
        title_col = next(
            (c for c in ("question", "title", "Question", "text") if c in df.columns),
            None,
        )
        out["title"] = (
            df[title_col].fillna("").astype(str) if title_col else ""
        )

        # body  (the code answer)
        body_col = next(
            (c for c in ("code_snippet", "body", "answer", "code", "snippet")
             if c in df.columns),
            None,
        )
        out["body"] = (
            df[body_col].fillna("").astype(str) if body_col else ""
        )

        # tags
        out["tags"] = (
            df["tags"].apply(self._parse_tags)
            if "tags" in df.columns
            else self.config
        )

        # derived fields
        out["category"]   = out["tags"].apply(
            lambda t: self._infer_category(t, self.config)
        )
        out["difficulty"] = out["body"].apply(self._infer_difficulty)

        # combined searchable text
        out["combined_text"] = (out["title"] + " " + out["body"]).str.strip()

        # drop empty rows
        out = out[out["combined_text"].str.len() > 10].reset_index(drop=True)

        logger.info(
            f"Normalised: {len(out):,} docs, "
            f"categories={out['category'].nunique()}"
        )
        return out

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_tags(raw) -> str:
        if isinstance(raw, list):
            return ", ".join(str(t) for t in raw)
        if isinstance(raw, str):
            raw = raw.strip()
            if raw.startswith("["):
                try:
                    lst = json.loads(raw.replace("'", '"'))
                    return ", ".join(str(t) for t in lst)
                except Exception:
                    pass
            return raw
        return ""

    @staticmethod
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
        if "python" in config:
            return "Python"
        if "sql" in config:
            return "SQL"
        return "General"

    @staticmethod
    def _infer_difficulty(body: str) -> str:
        n = len(body)
        if n < 200:  return "Beginner"
        if n < 600:  return "Intermediate"
        return "Advanced"

    def get_statistics(self) -> Dict[str, Any]:
        if self._df is None:
            self.load()
        return {
            "total_documents": len(self._df),
            "config": self.config,
            "categories": self._df["category"].value_counts().to_dict(),
            "difficulty_levels": self._df["difficulty"].value_counts().to_dict(),
        }


# ---------------------------------------------------------------------------
# DataProcessor  (drop-in replacement for existing code)
# ---------------------------------------------------------------------------

class DataProcessor:
    """
    Unified DataProcessor.
      - data_path ending in '.csv'  → legacy CSV loader
      - anything else               → StaQC Datasets Server API
    """

    def __init__(
        self,
        data_path: str,
        language: str    = "python",
        max_records: int = 100_000,
        cache_dir: str   = "data",
    ):
        self.data_path   = data_path
        self.language    = language
        self.max_records = max_records
        self.cache_dir   = cache_dir

        self._corpus:   List[str]            = []
        self._metadata: List[Dict[str, Any]] = []
        self._df:       Optional[pd.DataFrame] = None
        self._use_staqc = not str(data_path).endswith(".csv")

    # ------------------------------------------------------------------

    def load_data(self) -> pd.DataFrame:
        if self._use_staqc:
            loader = StaQCDatasetLoader(
                data_dir=self.cache_dir,
                language=self.language,
                max_records=self.max_records,
            )
            self._df = loader.load()
        else:
            logger.info(f"Loading legacy CSV: {self.data_path}")
            self._df = pd.read_csv(self.data_path)
            self._df = self._normalise_legacy_csv(self._df)
        return self._df

    def preprocess_data(
        self,
        combine_fields: Optional[List[str]] = None,
        clean: bool = True,
    ) -> Tuple[List[str], List[Dict[str, Any]]]:
        if self._df is None:
            self.load_data()

        if self._use_staqc:
            self._corpus = self._df["combined_text"].tolist()
        else:
            combine_fields = combine_fields or ["title", "body"]
            self._corpus = []
            for _, row in self._df.iterrows():
                parts = [
                    str(row[f]) for f in combine_fields
                    if f in row and pd.notna(row[f])
                ]
                text = " ".join(parts)
                if clean:
                    text = self._clean(text)
                self._corpus.append(text)

        self._metadata = []
        for idx, row in self._df.iterrows():
            self._metadata.append({
                "id":         str(row.get("id", idx)),
                "category":   str(row.get("category", "")),
                "difficulty": str(row.get("difficulty", "")),
                "title":      str(row.get("title", "")),
                "body":       str(row.get("body", "")),
                "tags":       str(row.get("tags", "")),
                "index":      int(idx),
            })

        logger.info(f"Preprocessed {len(self._corpus):,} documents")
        return self._corpus, self._metadata

    def get_corpus(self) -> List[str]:
        if not self._corpus:
            self.preprocess_data()
        return self._corpus

    def get_metadata(self) -> List[Dict[str, Any]]:
        if not self._metadata:
            self.preprocess_data()
        return self._metadata

    def get_document_by_index(self, index: int) -> Dict[str, Any]:
        if not self._metadata:
            self.preprocess_data()
        return self._metadata[index]

    def get_statistics(self) -> Dict[str, Any]:
        if self._df is None:
            self.load_data()
        return {
            "total_documents": len(self._df),
            "categories": (
                self._df["category"].value_counts().to_dict()
                if "category" in self._df.columns else {}
            ),
            "difficulty_levels": (
                self._df["difficulty"].value_counts().to_dict()
                if "difficulty" in self._df.columns else {}
            ),
        }

    @staticmethod
    def _clean(text: str) -> str:
        import re
        return re.sub(r"\s+", " ", str(text)).strip().lower()

    @staticmethod
    def _normalise_legacy_csv(df: pd.DataFrame) -> pd.DataFrame:
        if "combined_text" not in df.columns:
            title = df.get("title", pd.Series([""] * len(df))).fillna("")
            body  = df.get("body",  pd.Series([""] * len(df))).fillna("")
            df = df.copy()
            df["combined_text"] = (title + " " + body).str.strip()
        for col in ("id", "category", "difficulty", "title", "body", "tags"):
            if col not in df.columns:
                df[col] = ""
        return df

    def __len__(self) -> int:
        if self._df is None:
            self.load_data()
        return len(self._df)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return self.get_document_by_index(index)