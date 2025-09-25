#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
to_rag.py
---------
Baut einen FAISS-Index (Dense Embeddings) über Confluence-Chunks für RAG.

Beispiel (Index bauen):
  python to_rag.py build \
    --in data/raw/confluence.jsonl \
    --out-dir data/rag \
    --chunk-chars 1200 \
    --overlap 200 \
    --model sentence-transformers/all-MiniLM-L6-v2

Beispiel (Query testen):
  python to_rag.py query \
    --out-dir data/rag \
    --model sentence-transformers/all-MiniLM-L6-v2 \
    --q "Wie installiere ich QT6?" \
    --k 5
"""

from __future__ import annotations
import os, re, json, argparse, time, sys
from typing import Dict, Any, List, Tuple

import numpy as np

# Embeddings
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    print("Bitte installieren: pip install sentence-transformers")
    raise

# FAISS
try:
    import faiss
except Exception:
    print("Bitte installieren: pip install faiss-cpu")
    raise

def log(msg: str) -> None:
    print(f"[rag] {msg}")

def chunk_text(txt: str, max_chars: int, overlap: int) -> List[Tuple[int, str]]:
    """
    Char-basiertes Chunking mit Overlap.
    return: Liste (start_index, text)
    """
    txt = (txt or "").strip()
    if not txt:
        return []
    n = len(txt)
    if n <= max_chars:
        return [(0, txt)]
    chunks: List[Tuple[int, str]] = []
    start = 0
    while start < n:
        end = min(start + max_chars, n)
        chunk = txt[start:end]
        chunks.append((start, chunk.strip()))
        if end == n:
            break
        start = end - overlap if (end - overlap) > start else end
    return chunks

def build_index(inp: str, out_dir: str, model_name: str, chunk_chars: int, overlap: int) -> None:
    os.makedirs(out_dir, exist_ok=True)
    model = SentenceTransformer(model_name)

    texts: List[str] = []
    metas: List[Dict[str, Any]] = []

    pages = 0
    chunks_total = 0

    with open(inp, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                page = json.loads(line)
            except Exception:
                continue
            pages += 1
            base = (page.get("text_plain") or "").strip()
            if not base:
                continue

            for ci, (start_idx, ch) in enumerate(chunk_text(base, chunk_chars, overlap)):
                if not ch:
                    continue
                texts.append(ch)
                metas.append({
                    "doc_id": f"{page.get('id')}#{ci}",
                    "page_id": page.get("id"),
                    "page_title": page.get("title"),
                    "url": page.get("url"),
                    "chunk_index": ci,
                    "start_char": start_idx,
                })
                chunks_total += 1

    log(f"Seiten: {pages}, Chunks: {chunks_total} → Embeddings…")
    if not texts:
        log("Keine Texte gefunden. Abbruch.")
        return

    # Embeddings
    embs = model.encode(texts, batch_size=64, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)
    dim = embs.shape[1]
    index = faiss.IndexFlatIP(dim)  # Cosine (mit normalisierten Vektoren = Inner Product)
    index.add(embs)

    # Speichern
    faiss.write_index(index, os.path.join(out_dir, "index.faiss"))
    with open(os.path.join(out_dir, "chunks.jsonl"), "w", encoding="utf-8") as fout:
        for m, t in zip(metas, texts):
            rec = {"meta": m, "text": t}
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # Metainfo
    info = {
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": model_name,
        "dim": int(dim),
        "chunks": int(chunks_total),
        "source": os.path.abspath(inp),
        "chunk_chars": chunk_chars,
        "overlap": overlap,
    }
    with open(os.path.join(out_dir, "index_meta.json"), "w", encoding="utf-8") as fmeta:
        json.dump(info, fmeta, ensure_ascii=False, indent=2)

    log(f"Index gespeichert unter {out_dir}/index.faiss, Chunks unter {out_dir}/chunks.jsonl")

def load_index(out_dir: str, model_name: str):
    index = faiss.read_index(os.path.join(out_dir, "index.faiss"))
    model = SentenceTransformer(model_name)
    metas, texts = [], []
    with open(os.path.join(out_dir, "chunks.jsonl"), "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            rec = json.loads(line)
            metas.append(rec["meta"])
            texts.append(rec["text"])
    return index, model, metas, texts

def query_index(out_dir: str, model_name: str, q: str, k: int) -> None:
    index, model, metas, texts = load_index(out_dir, model_name)
    qv = model.encode([q], convert_to_numpy=True, normalize_embeddings=True)
    D, I = index.search(qv, k)
    print("\nTop-K Ergebnisse:")
    for score, idx in zip(D[0].tolist(), I[0].tolist()):
        if idx < 0 or idx >= len(metas): continue
        m = metas[idx]
        t = texts[idx][:300].replace("\n", " ")
        print(f"- score={score:.3f} | {m.get('page_title')} (#{m.get('chunk_index')})")
        print(f"  URL: {m.get('url')}")
        print(f"  Snippet: {t}")
    print("")

def main():
    ap = argparse.ArgumentParser(description="FAISS-RAG Index Builder/Query über Confluence-Chunks.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="Index bauen")
    b.add_argument("--in", dest="inp", required=True, help="canonical JSONL")
    b.add_argument("--out-dir", required=True, help="Zielordner (Index + Chunks)")
    b.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2", help="Embedding-Modell (lokal oder HF-Name)")
    b.add_argument("--chunk-chars", type=int, default=1200)
    b.add_argument("--overlap", type=int, default=200)

    q = sub.add_parser("query", help="Index abfragen")
    q.add_argument("--out-dir", required=True)
    q.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    q.add_argument("--q", required=True, help="Frage/Textquery")
    q.add_argument("--k", type=int, default=5)

    args = ap.parse_args()

    if args.cmd == "build":
        build_index(args.inp, args.out_dir, args.model, args.chunk_chars, args.overlap)
    elif args.cmd == "query":
        query_index(args.out_dir, args.model, args.q, args.k)

if __name__ == "__main__":
    main()
