#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
to_finetune.py
--------------
Erzeugt Q&A-JSONL aus canonical JSONL OHNE Online-LLM.
- chunked über 'text_plain' (oder optional body_storage -> Text)
- generiert 2..3 extraktive Q&A pro Chunk (keine Halluzinationen)
- fügt Zitate/Snippets und Metadaten hinzu

Beispiel:
  python to_finetune.py \
    --in data/raw/confluence.jsonl \
    --out data/finetune/qa.jsonl \
    --chunk-chars 1800 \
    --max-items-per-chunk 3
"""

from __future__ import annotations
import os, re, json, argparse, time
from typing import Dict, Any, List, Iterable

from bs4 import BeautifulSoup

def log(msg: str) -> None:
    print(f"[finetune] {msg}")

def sentence_split(text: str) -> List[str]:
    # sehr einfache Satztrennung
    text = re.sub(r'\s+', ' ', text).strip()
    if not text:
        return []
    # split by punctuation + space
    parts = re.split(r'(?<=[\.\!\?])\s+', text)
    return [p.strip() for p in parts if p.strip()]

def extract_headings_from_html(html: str, max_n: int = 5) -> List[str]:
    heads: List[str] = []
    try:
        soup = BeautifulSoup(html or "", "html.parser")
        for tag in soup.find_all(["h1","h2","h3","h4"]):
            txt = tag.get_text(" ").strip()
            if txt and txt not in heads:
                heads.append(txt)
            if len(heads) >= max_n:
                break
    except Exception:
        pass
    return heads

def chunk_text(txt: str, max_chars: int) -> List[str]:
    txt = (txt or "").strip()
    if not txt:
        return []
    if len(txt) <= max_chars:
        return [txt]
    chunks: List[str] = []
    start = 0
    while start < len(txt):
        end = min(start + max_chars, len(txt))
        # versuche am Satzende zu schneiden
        slice_txt = txt[start:end]
        # rückwärts bis letztes Satzende
        m = re.search(r'.*[\.\!\?]\s', slice_txt)
        if m and (start + m.end()) - start > max_chars * 0.5:
            end = start + m.end()
        chunks.append(txt[start:end].strip())
        start = end
    return [c for c in chunks if c]

def make_items_for_chunk(page: Dict[str, Any], chunk: str, heads: List[str], max_items: int) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    sents = sentence_split(chunk)

    # 1) Zusammenfassung (kurz/lang) aus den ersten Sätzen
    if sents:
        short = " ".join(sents[:2]).strip()[:400]
        long = " ".join(sents[:6]).strip()[:1200]
        items.append({
            "question": f"Worum geht es im folgenden Abschnitt der Seite „{page.get('title','')}“?",
            "short_answer": short,
            "long_answer": long,
            "citations": sents[:3],  # erste Sätze als Zitat
        })

    # 2) Drei wichtige Punkte (extraktiv: nimm 3 „lange“ Sätze)
    if len(sents) >= 3:
        # nimm 3 Sätze mit größter Länge
        top = sorted(sents, key=len, reverse=True)[:3]
        bullets = ["- " + t for t in top]
        items.append({
            "question": "Nenne drei wichtige Punkte aus dem Abschnitt.",
            "short_answer": " ".join(t[:120] for t in top),
            "long_answer": "\n".join(bullets),
            "citations": top,
        })

    # 3) Überschriften (falls vorhanden)
    if heads:
        items.append({
            "question": f"Welche Hauptüberschriften kommen auf der Seite „{page.get('title','')}“ vor?",
            "short_answer": "; ".join(heads[:3]),
            "long_answer": "\n".join([f"- {h}" for h in heads]),
            "citations": heads[:3],
        })

    # Kappen auf max_items
    return items[:max_items]

def main():
    ap = argparse.ArgumentParser(description="Erzeuge extraktive Q&A-JSONL aus canonical JSONL.")
    ap.add_argument("--in", dest="inp", required=True, help="canonical JSONL")
    ap.add_argument("--out", required=True, help="Ziel-JSONL (Q&A)")
    ap.add_argument("--chunk-chars", type=int, default=1800, help="Chunkgröße in Zeichen")
    ap.add_argument("--max-items-per-chunk", type=int, default=3, help="max. Q&A-Items pro Chunk")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    total_pages = 0
    total_chunks = 0
    total_items = 0

    with open(args.inp, "r", encoding="utf-8") as fin, open(args.out, "w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            try:
                page = json.loads(line)
            except Exception:
                continue

            total_pages += 1
            text = page.get("text_plain") or ""
            chunks = chunk_text(text, args.chunk_chars)
            heads = extract_headings_from_html(page.get("body_storage") or page.get("body_view") or "")
            for ci, ch in enumerate(chunks):
                total_chunks += 1
                items = make_items_for_chunk(page, ch, heads, args.max_items_per_chunk)
                for it in items:
                    rec = {
                        "page_id": page.get("id"),
                        "page_title": page.get("title"),
                        "source_url": page.get("url"),
                        "chunk_index": ci,
                        "question": it["question"],
                        "short_answer": it["short_answer"],
                        "long_answer": it["long_answer"],
                        "citations": it.get("citations", []),
                        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    }
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    total_items += 1

    log(f"Fertig. Seiten: {total_pages}, Chunks: {total_chunks}, Q&A-Items: {total_items}, Output: {args.out}")

if __name__ == "__main__":
    main()
