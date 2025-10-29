#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, json, argparse, pickle
from typing import Dict, Any, Iterable, List, Tuple
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

# ===== Pfade / Defaults =====
DEFAULT_DATASET_PATH = "../WikiExtraction/data/derivatives/joined_pages_full.jsonl"
DEFAULT_CACHE_DIR    = "cag_cache"
DEFAULT_MAX_DOCS     = 2_000_000   # nur als Sicherheitsdeckel
DEFAULT_MIN_DF       = 2
DEFAULT_MAX_FEATURES = 250_000     # je nach RAM anpassen

def iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

def select_text(row: Dict[str, Any]) -> str:
    txt = (
        row.get("text_plain_plus_attachments")
        or row.get("text_plain")
        or row.get("body_view")
        or ""
    )
    # Whitespace glätten
    return " ".join(str(txt).split())

def main():
    ap = argparse.ArgumentParser(description="Build TF-IDF Cache (CAG) aus JSONL")
    ap.add_argument("--dataset", default=DEFAULT_DATASET_PATH)
    ap.add_argument("--outdir",  default=DEFAULT_CACHE_DIR)
    ap.add_argument("--min-df",  type=int, default=DEFAULT_MIN_DF)
    ap.add_argument("--max-features", type=int, default=DEFAULT_MAX_FEATURES)
    ap.add_argument("--max-docs", type=int, default=DEFAULT_MAX_DOCS)
    args = ap.parse_args()

    dataset   = args.dataset
    outdir    = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    docs: List[str] = []
    metas: List[Dict[str, Any]] = []

    print(f"[CAG] Lade JSONL: {dataset}")
    for i, row in enumerate(iter_jsonl(dataset), start=1):
        content = select_text(row)
        if not content:
            continue
        docs.append(content)
        metas.append({
            "page_id": str(row.get("id", "")),
            "title":   row.get("title", ""),
            "url":     row.get("url", "") or row.get("source", ""),
            "space":   row.get("space_key", ""),
            "updated": row.get("updated", "")
        })
        if i % 1000 == 0:
            print(f"[CAG] Gelesen: {i}")
        if i >= args.max_docs:
            break

    print(f"[CAG] Baue TF-IDF (n_docs={len(docs)}) …")
    vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1,2),       # Unigram + Bigram ist stabil für Tech-Doku
        min_df=args.min_df,
        max_features=args.max_features,
        lowercase=True
    )
    X = vectorizer.fit_transform(docs)  # CSR-Matrix

    print(f"[CAG] Speichere Cache nach: {outdir}")
    sparse.save_npz(outdir / "tfidf_matrix.npz", X)
    with open(outdir / "vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)
    with open(outdir / "meta.jsonl", "w", encoding="utf-8") as f:
        for m in metas:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")

    print("[CAG] Fertig.")

if __name__ == "__main__":
    main()
