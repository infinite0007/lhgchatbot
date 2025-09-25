#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
attachments_join_page.py
----------------------------
Mergt OCR-/Draw.io-Texte je Seite zurück in ein Feld 'attachments_text'.
Schreibt eine neue JSONL: pro Seite ein Record inkl. erweitertem Text.

Beispiel:
  python attachments_join_page.py \
    --pages data/raw/confluence.jsonl \
    --ocr data/derivatives/ocr.jsonl \
    --pdf data/derivatives/pdf_ocr.jsonl \
    --drawio data/derivatives/drawio_text.jsonl \
    --out data/derivatives/pages_with_attachments.jsonl
"""

from __future__ import annotations
import os, json, argparse
from collections import defaultdict

def load_rows(path):
    rows = []
    if not path or not os.path.exists(path): return rows
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try: rows.append(json.loads(line))
                except: pass
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages", required=True)
    ap.add_argument("--ocr", default=None)
    ap.add_argument("--pdf", default=None)
    ap.add_argument("--drawio", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ocr_rows   = load_rows(args.ocr)
    pdf_rows   = load_rows(args.pdf)
    draw_rows  = load_rows(args.drawio)

    by_page = defaultdict(list)
    for r in ocr_rows:
        if r.get("page_id") and r.get("ocr_text"):
            by_page[r["page_id"]].append(r["ocr_text"])
    for r in pdf_rows:
        if r.get("page_id") and r.get("ocr_text"):
            by_page[r["page_id"]].append(r["ocr_text"])
    for r in draw_rows:
        if r.get("page_id") and r.get("drawio_text"):
            by_page[r["page_id"]].append(r["drawio_text"])

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    pages = load_rows(args.pages)
    with open(args.out, "w", encoding="utf-8") as fout:
        for p in pages:
            pid = p.get("id")
            extras = "\n".join(by_page.get(pid, []))
            p["attachments_text"] = extras
            # Optional: in text_plain anhängen (für RAG direkt verwertbar)
            if extras:
                p["text_plain_plus_attachments"] = (p.get("text_plain") or "") + "\n\n" + extras
            else:
                p["text_plain_plus_attachments"] = p.get("text_plain") or ""
            fout.write(json.dumps(p, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()
