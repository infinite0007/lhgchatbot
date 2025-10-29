#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, json, argparse, uuid
from typing import Dict, Any, Iterable, List, Tuple
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

# ===== Defaults =====
DEFAULT_DATASET_PATH   = "../WikiExtraction/data/derivatives/joined_pages_full.jsonl"
DEFAULT_CACHE_DIR      = "cag_cache"
DEFAULT_MAX_DOCS       = 2_000_000
DEFAULT_EMBED_MODEL    = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_CHUNK_SIZE     = 1000
DEFAULT_CHUNK_OVERLAP  = 200
DEFAULT_BATCH_SIZE     = 256
DEFAULT_TOP_K          = 5

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
    return " ".join(str(txt).split())

def chunk_text(text: str, size: int, overlap: int) -> List[Tuple[int, int, str]]:
    """Returns list of (start_idx, end_idx, chunk_text)."""
    if not text:
        return []
    if size <= 0:
        return [(0, len(text), text)]
    chunks = []
    step = max(1, size - overlap)
    i = 0
    n = len(text)
    while i < n:
        end = min(n, i + size)
        chunks.append((i, end, text[i:end]))
        if end == n:
            break
        i += step
    return chunks

def l2_normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12
    return mat / norms

def cosine_topk(query_vec: np.ndarray, emb: np.ndarray, k: int) -> List[int]:
    # emb und query_vec sind L2-normalized -> Cosine = Dot
    scores = emb @ query_vec  # (N,)
    if k >= len(scores):
        return list(np.argsort(-scores))
    return list(np.argpartition(-scores, k)[:k][np.argsort(-scores[np.argpartition(-scores, k)[:k]])])

def main():
    ap = argparse.ArgumentParser(description="Build CAG-Cache mit Sentence-Transformer-Embeddings (einfach).")
    ap.add_argument("--dataset",       default=DEFAULT_DATASET_PATH)
    ap.add_argument("--outdir",        default=DEFAULT_CACHE_DIR)
    ap.add_argument("--embed-model",   default=DEFAULT_EMBED_MODEL)
    ap.add_argument("--chunk-size",    type=int, default=DEFAULT_CHUNK_SIZE)
    ap.add_argument("--chunk-overlap", type=int, default=DEFAULT_CHUNK_OVERLAP)
    ap.add_argument("--batch-size",    type=int, default=DEFAULT_BATCH_SIZE)
    ap.add_argument("--max-docs",      type=int, default=DEFAULT_MAX_DOCS)
    ap.add_argument("--no-normalize",  action="store_true", help="Deaktiviert L2-Normalisierung der Embeddings.")
    ap.add_argument("--test-query",    type=str, default="", help="Optional: ad-hoc Query gegen frisch gebauten Cache.")
    ap.add_argument("--top-k",         type=int, default=DEFAULT_TOP_K)
    args = ap.parse_args()

    dataset_path   = args.dataset
    outdir         = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # 1) Daten laden & chunken
    texts: List[str] = []
    metas: List[Dict[str, Any]] = []

    print(f"[CAG] Lade JSONL: {dataset_path}")
    doc_count = 0
    for row in iter_jsonl(dataset_path):
        content = select_text(row)
        if not content:
            continue

        chunks = chunk_text(content, args.chunk_size, args.chunk_overlap)
        base_meta = {
            "page_id": str(row.get("id", "")),
            "title":   row.get("title", ""),
            "url":     row.get("url", "") or row.get("source", ""),
            "space":   row.get("space_key", ""),
            "updated": row.get("updated", ""),
        }

        for (start, end, ctext) in chunks:
            cid = str(uuid.uuid4())
            texts.append(ctext)
            meta = dict(base_meta)
            meta.update({
                "chunk_id": cid,
                "start": start,
                "end": end,
                "length": end - start
            })
            metas.append(meta)

        doc_count += 1
        if doc_count % 100 == 0:
            print(f"[CAG] Seiten: {doc_count}, Chunks gesamt: {len(texts)}")
        if doc_count >= args.max_docs:
            break

    if not texts:
        print("[CAG] Keine Texte gefunden. Abbruch.")
        return

    # 2) Embeddings berechnen
    print(f"[CAG] Lade Embedding-Modell: {args.embed_model}")
    model = SentenceTransformer(args.embed_model)

    print(f"[CAG] Encode {len(texts)} Chunks (batch_size={args.batch_size}) …")
    emb = model.encode(
        texts,
        batch_size=args.batch_size,
        convert_to_numpy=True,
        normalize_embeddings=not args.no_normalize,  # normalisiere standardmäßig
        show_progress_bar=True
    ).astype(np.float32)

    # Fallback-Normalisierung, falls obige Option deaktiviert wurde
    if args.no_normalize:
        emb = l2_normalize(emb).astype(np.float32)

    # 3) Speichern
    np.save(outdir / "embeddings.npy", emb)
    with open(outdir / "meta.jsonl", "w", encoding="utf-8") as f:
        for m in metas:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
    with open(outdir / "texts.jsonl", "w", encoding="utf-8") as f:
        for cid, t in zip([m["chunk_id"] for m in metas], texts):
            f.write(json.dumps({"chunk_id": cid, "text": t}, ensure_ascii=False) + "\n")
    config = {
        "dataset": dataset_path,
        "outdir": str(outdir),
        "embed_model": args.embed_model,
        "chunk_size": args.chunk_size,
        "chunk_overlap": args.chunk_overlap,
        "batch_size": args.batch_size,
        "normalized": True,
        "num_chunks": len(texts),
        "embedding_dim": int(emb.shape[1]),
    }
    with open(outdir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    print(f"[CAG] Fertig. Chunks: {len(texts)}, Embedding-Dim: {emb.shape[1]}")
    print(f"[CAG] Output: {outdir}/embeddings.npy, meta.jsonl, texts.jsonl, config.json")

    # 4) Optional: Ad-hoc Top-k-Test
    if args.test_query:
        print(f"[CAG] Test-Query: {args.test_query!r}")
        q = model.encode(
            [args.test_query],
            convert_to_numpy=True,
            normalize_embeddings=True
        ).astype(np.float32)[0]
        idxs = cosine_topk(q, emb, k=args.top_k)
        print(f"[CAG] Top-{args.top_k} Chunk-IDs:")
        for rank, i in enumerate(idxs, 1):
            m = metas[i]
            print(f"- {rank:>2}. {m['chunk_id']} | {m.get('title','')} | {m.get('url','')}")
            # Optional: Kurz-Snippet ausgeben
            snippet = texts[i][:200].replace("\n", " ")
            print(f"      {snippet}...")
        print("[CAG] Test-Query fertig.")

if __name__ == "__main__":
    main()
