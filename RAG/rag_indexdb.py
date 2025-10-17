#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, json, shutil, uuid, argparse
from typing import Dict, Any, Iterable, List

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# ========= Defaults =========
DEFAULT_DATASET_PATH      = "../WikiExtraction/data/derivatives/joined_pages_full.jsonl"
DEFAULT_DATABASE_LOCATION = "chroma_db"
DEFAULT_COLLECTION_NAME   = "rag_data"
DEFAULT_EMBEDDING_MODEL   = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_CHUNK_SIZE        = 1000
DEFAULT_CHUNK_OVERLAP     = 200
DEFAULT_RESET_DB          = True

# Chroma kann pro Upsert nur ~5.4k Items; bleib deutlich darunter:
MAX_CHROMA_BATCH          = 1000   # sicher < 5461

def select_text(row: Dict[str, Any]) -> str:
    txt = (
        row.get("text_plain_plus_attachments")
        or row.get("text_plain")
        or row.get("body_view")
        or ""
    )
    return " ".join(str(txt).split())

def iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

def batched(lst: List[Any], n: int) -> Iterable[List[Any]]:
    for i in range(0, len(lst), n):
        yield lst[i : i + n]

def main():
    ap = argparse.ArgumentParser(description="Ingest JSONL → Chroma (HF Embeddings)")
    ap.add_argument("--dataset", default=DEFAULT_DATASET_PATH)
    ap.add_argument("--db", default=DEFAULT_DATABASE_LOCATION)
    ap.add_argument("--collection", default=DEFAULT_COLLECTION_NAME)
    ap.add_argument("--embed-model", default=DEFAULT_EMBEDDING_MODEL)
    ap.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    ap.add_argument("--chunk-overlap", type=int, default=DEFAULT_CHUNK_OVERLAP)
    ap.add_argument("--reset", action="store_true", default=DEFAULT_RESET_DB)
    args = ap.parse_args()

    dataset_path      = args.dataset
    database_location = args.db
    collection_name   = args.collection
    embedding_model   = args.embed_model
    chunk_size        = args.chunk_size
    chunk_overlap     = args.chunk_overlap
    reset_db          = args.reset

    if reset_db and os.path.exists(database_location):
        print(f"[INGEST] RESET_DB=True → remove '{database_location}'")
        shutil.rmtree(database_location, ignore_errors=True)

    # Embeddings (lokal)
    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model,
        # encode_kwargs={"normalize_embeddings": True},  # optional
    )

    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=database_location,
    )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )

    total_pages, total_chunks = 0, 0
    for row in iter_jsonl(dataset_path):
        content = select_text(row)
        if not content:
            continue

        meta = {
            "source": row.get("source", ""),
            "page_id": str(row.get("id", "")),
            "title": row.get("title", ""),
            "url": row.get("url", ""),
            "space_key": row.get("space_key", ""),
            "updated": row.get("updated", ""),
        }

        docs = splitter.create_documents([content], metadatas=[meta])
        if not docs:
            continue

        # IDs + Batch-Upsert
        ids = [str(uuid.uuid4()) for _ in docs]
        for docs_batch, ids_batch in zip(batched(docs, MAX_CHROMA_BATCH), batched(ids, MAX_CHROMA_BATCH)):
            vector_store.add_documents(documents=docs_batch, ids=ids_batch)

        total_pages += 1
        total_chunks += len(docs)
        if total_pages % 100 == 0:
            print(f"[INGEST] {total_pages} Seiten, {total_chunks} Chunks …")

    print(f"[INGEST] DONE. Seiten: {total_pages}, Chunks: {total_chunks}")

if __name__ == "__main__":
    main()
