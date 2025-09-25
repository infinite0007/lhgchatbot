#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
attachments_pdf_ocr.py
----------------------
OCR für PDF-Attachments, die von canonical_extractor.py heruntergeladen wurden.
- Konvertiert jede PDF-Seite in ein Bild (pdf2image)
- Läuft EasyOCR drüber
- Speichert ein JSONL (1 Zeile pro PDF mit zusammengeführtem Text)
- benötigt poppler

Beispiel:

  python attachments_pdf_ocr.py --attachments-dir data/raw/attachments --canonical-json data/raw/confluence.jsonl --out data/derivatives/pdf_ocr.jsonl --langs de en --dpi 250 --poppler-path "data/poppler-25.07.0/Library/bin"

  python attachments_pdf_ocr.py
    --attachments-dir data/raw/attachments
    --canonical-json data/raw/confluence.jsonl
    --out data/derivatives/pdf_ocr.jsonl
    --langs de en
    --dpi 250
    --poppler-path "data/poppler-25.07.0/Library/bin"
"""

from __future__ import annotations
import os, json, time, argparse, tempfile, shutil
from typing import Dict, Any, List, Optional

from pdf2image import convert_from_path
import easyocr

def log(msg: str) -> None:
    print(f"[pdf-ocr] {msg}")

def load_canonical_map(canonical_path: Optional[str]) -> Dict[str, Dict[str, Any]]:
    by_local: Dict[str, Dict[str, Any]] = {}
    if not canonical_path or not os.path.exists(canonical_path):
        return by_local
    with open(canonical_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            page_id = row.get("id")
            page_title = row.get("title")
            for a in (row.get("attachments") or []):
                lp = a.get("local_path")
                if not lp:
                    continue
                key = os.path.normpath(lp)
                by_local[key] = {
                    "page_id": page_id,
                    "page_title": page_title,
                    "attachment_id": a.get("id"),
                    "title": a.get("title"),
                    "mediaType": a.get("mediaType"),
                    "fileSize": a.get("fileSize"),
                }
    return by_local

def find_pdfs(attachments_dir: str) -> List[str]:
    out = []
    for root, _, files in os.walk(attachments_dir):
        for fn in files:
            if os.path.splitext(fn)[1].lower() == ".pdf":
                out.append(os.path.join(root, fn))
    return out

def ocr_pdf(reader: "easyocr.Reader", pdf_path: str, dpi: int, poppler_path: Optional[str]) -> Dict[str, Any]:
    # Jede Seite rendern, OCR drüber, Texte zusammenführen
    with tempfile.TemporaryDirectory() as tmpdir:
        pages = convert_from_path(pdf_path, dpi=dpi, output_folder=tmpdir, poppler_path=poppler_path)
        page_texts: List[str] = []
        for i, pil_img in enumerate(pages):
            img_path = os.path.join(tmpdir, f"p{i:04d}.png")
            pil_img.save(img_path)
            txt = " ".join(reader.readtext(img_path, detail=0)).strip()
            page_texts.append(txt)
        return {
            "pages": len(page_texts),
            "text": "\n".join(page_texts).strip()
        }

def main():
    ap = argparse.ArgumentParser(description="PDF → Image → EasyOCR für Confluence-Attachments.")
    ap.add_argument("--attachments-dir", default="data/raw/attachments")
    ap.add_argument("--canonical-json", default=None)
    ap.add_argument("--out", default="data/derivatives/pdf_ocr.jsonl")
    ap.add_argument("--langs", nargs="+", default=["de","en"])
    ap.add_argument("--dpi", type=int, default=250)
    ap.add_argument("--poppler-path", default=None, help="Pfad zum Poppler bin-Verzeichnis (Windows)")
    ap.add_argument("--skip-existing", action="store_true", help="Bestehende Records (local_path) im Output überspringen")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    meta_by_local = load_canonical_map(args.canonical_json)

    existing: set[str] = set()
    if args.skip_existing and os.path.exists(args.out):
        with open(args.out, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    row = json.loads(line)
                    if row.get("local_path"):
                        existing.add(os.path.normpath(row["local_path"]))
                except Exception:
                    pass

    reader = easyocr.Reader(args.langs, gpu=False)
    pdfs = find_pdfs(args.attachments_dir)

    done = 0
    with open(args.out, "a", encoding="utf-8") as fout:
        for pdf in pdfs:
            npath = os.path.normpath(pdf)
            if npath in existing:
                continue
            try:
                res = ocr_pdf(reader, npath, args.dpi, args.poppler_path)
                rec = {
                    "local_path": npath,
                    "ocr_text": res["text"],
                    "pages": res["pages"],
                    "langs": args.langs,
                    "extracted_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                }
                rec.update(meta_by_local.get(npath, {}))
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                done += 1
            except Exception as e:
                log(f"WARN: OCR fehlgeschlagen für {npath}: {e}")

    log(f"Fertig. PDFs gefunden: {len(pdfs)} | erfolgreich OCR: {done} | Output: {args.out}")

if __name__ == "__main__":
    main()
