#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, json, argparse, uuid
from pathlib import Path
from typing import Dict, Any, Iterable, List, Tuple

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.cache_utils import DynamicCache
from sentence_transformers import SentenceTransformer

# ---------- Defaults ----------
DEFAULT_DATASET_PATH   = "../WikiExtraction/data/derivatives/joined_pages_full.jsonl"
DEFAULT_OUTDIR         = "cag_cache"
DEFAULT_EMBED_MODEL    = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_CHUNK_SIZE     = 1000
DEFAULT_CHUNK_OVERLAP  = 200
DEFAULT_BATCH_SIZE     = 256
DEFAULT_MAX_DOCS       = 2_000_000
DEFAULT_HF_MODEL       = "../Qwen2.5-0.5B-Instruct"
DEFAULT_HOTSET_N       = 1024   # wie viele Chunks beim Serverstart vorladen

SYSTEM_PROMPT = (
    "You are a precise assistant for enterprise knowledge.\n"
    "Use only the given context. If nothing relevant is found, reply exactly 'Not in context.'\n"
    "Add inline citations like [1], [2] when supported by the context."
)
TEMPLATE = {
    "system": SYSTEM_PROMPT,
    "prefix_before_ctx": "Context:\n",
    "suffix_after_ctx": "\n\nQuestion: ",
    "answer_prefix": "\n\nAnswer:"
}

def iter_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

def select_text(row: Dict[str, Any]) -> str:
    txt = (row.get("text_plain_plus_attachments") or
           row.get("text_plain") or
           row.get("body_view") or "")
    return " ".join(str(txt).split())

def chunk_text(text: str, size: int, overlap: int) -> List[Tuple[int,int,str]]:
    if not text:
        return []
    if size <= 0:
        return [(0, len(text), text)]
    out, step, i, n = [], max(1, size - overlap), 0, len(text)
    while i < n:
        end = min(n, i + size)
        out.append((i, end, text[i:end]))
        if end == n: break
        i += step
    return out

def build_texts_meta(dataset_path: str, outdir: Path, chunk_size: int, chunk_overlap: int, max_docs: int):
    outdir.mkdir(parents=True, exist_ok=True)
    texts, metas = [], []
    pages = 0
    print(f"[CAG] Lade JSONL: {dataset_path}")
    for row in iter_jsonl(dataset_path):
        content = select_text(row)
        if not content:
            continue
        pieces = chunk_text(content, chunk_size, chunk_overlap)
        base_meta = {
            "page_id": str(row.get("id", "")),
            "title":   row.get("title", ""),
            "url":     row.get("url", "") or row.get("source", ""),
            "space":   row.get("space_key", ""),
            "updated": row.get("updated", ""),
        }
        for (start, end, ctext) in pieces:
            cid = str(uuid.uuid4())
            texts.append(ctext)
            m = dict(base_meta)
            m.update({"chunk_id": cid, "start": start, "end": end, "length": end - start})
            metas.append(m)
        pages += 1
        if pages % 100 == 0:
            print(f"[CAG] Seiten: {pages}, Chunks gesamt: {len(texts)}")
        if pages >= max_docs:
            break

    if not texts:
        raise RuntimeError("[CAG] Keine Texte gefunden.")

    (outdir / "texts.jsonl").write_text(
        "".join(json.dumps({"chunk_id": m["chunk_id"], "text": t}, ensure_ascii=False) + "\n"
                for m, t in zip(metas, texts)),
        encoding="utf-8"
    )
    (outdir / "meta.jsonl").write_text(
        "".join(json.dumps(m, ensure_ascii=False) + "\n" for m in metas),
        encoding="utf-8"
    )
    cfg = {
        "dataset": dataset_path,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "num_chunks": len(texts),
        "embed_model": DEFAULT_EMBED_MODEL,
    }
    (outdir / "config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[CAG] Fertig: Chunks={len(texts)}")
    return texts, metas

def build_embeddings(texts: List[str], outdir: Path, embed_model: str, batch_size: int):
    print(f"[CAG] Lade Embedding-Modell: {embed_model}")
    model = SentenceTransformer(embed_model)
    print(f"[CAG] Encode {len(texts)} Chunks (batch_size={batch_size}) …")
    emb = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,   # L2-normiert -> Cosine == Dot
        show_progress_bar=True
    ).astype(np.float32)
    np.save(outdir / "embeddings.npy", emb)
    print(f"[CAG] Embeddings gespeichert: {emb.shape}")
    return emb

def build_kv_per_chunk(outdir: Path, texts: List[str], metas: List[Dict[str, Any]], hf_model_path: str):
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    model_path = str(Path(hf_model_path).expanduser().resolve())
    print(f"[KV] Lade Tokenizer/Model (lokal): {model_path}")

    tok = AutoTokenizer.from_pretrained(model_path, use_fast=True, local_files_only=True, trust_remote_code=True)
    mdl = AutoModelForCausalLM.from_pretrained(
        model_path, device_map="auto", torch_dtype="auto", local_files_only=True, trust_remote_code=True
    ).eval()

    kv_dir = outdir / "kv"
    kv_dir.mkdir(parents=True, exist_ok=True)

    sys_ids = tok(SYSTEM_PROMPT, return_tensors="pt", add_special_tokens=False).input_ids.to(mdl.device)

    def _prefill(text: str):
        tail = TEMPLATE["prefix_before_ctx"] + text + TEMPLATE["suffix_after_ctx"]
        tail_ids = tok(tail, return_tensors="pt", add_special_tokens=False).input_ids.to(mdl.device)
        input_ids = torch.cat([sys_ids, tail_ids], dim=1)
        kv = DynamicCache()
        with torch.no_grad():
            _ = mdl(input_ids=input_ids, past_key_values=kv, use_cache=True, return_dict=True)

        # als **LEGACY** sichern (CPU/fp16) -> sicher mit torch.load(weights_only=True)
        kv_legacy = []
        for k, v in kv.to_legacy_cache():
            kv_legacy.append((
                k.to("cpu", dtype=torch.float16, non_blocking=True),
                v.to("cpu", dtype=torch.float16, non_blocking=True),
            ))
        return kv_legacy, int(input_ids.shape[1])

    count = 0
    for t, m in zip(texts, metas):
        chunk_id = m["chunk_id"]
        kv_legacy, seq_len = _prefill(t)
        torch.save({
            "chunk_id": chunk_id,
            "seq_len": seq_len,
            "kv_legacy": kv_legacy,
            "template": TEMPLATE,
            "tokenizer_name_or_path": tok.name_or_path,
            "model_id": model_path,
        }, (outdir / "kv" / f"{chunk_id}.pt"))
        count += 1
        if count % 500 == 0:
            print(f"[KV] {count} KV-Caches geschrieben …")

    print(f"[KV] Fertig: {count} KV-Dateien")

def pick_hotset(metas: List[Dict[str, Any]], N: int) -> List[str]:
    # Proxy: längere Chunks sind oft informativer. Ersetze später durch echte Nutzungs-Logs.
    metas_sorted = sorted(metas, key=lambda m: int(m.get("length", 0)), reverse=True)
    return [m["chunk_id"] for m in metas_sorted[:N]]

def main():
    ap = argparse.ArgumentParser("Build CAG hot-set: texts/meta + embeddings + KV(legacy)")
    ap.add_argument("--dataset",       default=DEFAULT_DATASET_PATH)
    ap.add_argument("--outdir",        default=DEFAULT_OUTDIR)
    ap.add_argument("--embed-model",   default=DEFAULT_EMBED_MODEL)
    ap.add_argument("--chunk-size",    type=int, default=DEFAULT_CHUNK_SIZE)
    ap.add_argument("--chunk-overlap", type=int, default=DEFAULT_CHUNK_OVERLAP)
    ap.add_argument("--batch-size",    type=int, default=DEFAULT_BATCH_SIZE)
    ap.add_argument("--max-docs",      type=int, default=DEFAULT_MAX_DOCS)
    ap.add_argument("--hf-model",      default=DEFAULT_HF_MODEL)
    ap.add_argument("--hotset",        type=int, default=DEFAULT_HOTSET_N)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    texts, metas = build_texts_meta(args.dataset, outdir, args.chunk_size, args.chunk_overlap, args.max_docs)
    build_embeddings(texts, outdir, args.embed_model, args.batch_size)
    build_kv_per_chunk(outdir, texts, metas, args.hf_model)

    hot = pick_hotset(metas, args.hotset)
    (outdir / "kv_hotset.json").write_text(json.dumps({"hot_chunk_ids": hot}, indent=2), encoding="utf-8")
    print(f"[HOT] Hot-Set Größe: {len(hot)}")

if __name__ == "__main__":
    main()
