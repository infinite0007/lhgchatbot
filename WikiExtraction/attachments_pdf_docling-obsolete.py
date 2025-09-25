#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
attachments_pdf_docling.py
--------------------------
Parst PDF-Attachments mit Docling (ohne OCR).
Schreibt 1 JSONL-Zeile pro PDF (Text und optional Markdown).

OFFLINE-NUTZUNG:
- pip install docling
- Doku: https://docling-project.github.io/docling/usage/advanced_options/
- Modelle vorab ziehen (auf einem Rechner mit Internet):
    docling-tools models download
  Falls SSL/Proxy Probleme: auf einem anderen Rechner laden und den kompletten
  Modelle-Ordner z.B. nach: data/doclingdata/models kopieren.
- Danach hier per --artifacts-path oder Umgebungsvariable DOCLING_ARTIFACTS_PATH
  zu diesem Ordner zeigen.

Beispiel:
  python attachments_pdf_docling.py --attachments-dir data/raw/attachments --canonical-json data/raw/confluence.jsonl --out data/derivatives/pdf_docling.jsonl --export-md --artifacts-path data/doclingdata/models

  python attachments_pdf_docling.py \
    --attachments-dir data/raw/attachments \
    --canonical-json data/raw/confluence.jsonl \
    --out data/derivatives/pdf_docling.jsonl \
    --export-md \
    --artifacts-path data/doclingdata/models
"""

from __future__ import annotations
import os, json, time, argparse
from typing import Dict, Any, List, Optional

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, EasyOcrOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

def log(msg: str) -> None:
    print(f"[pdf-docling] {msg}")

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
                if not lp or not lp.lower().endswith(".pdf"):
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

def main():
    ap = argparse.ArgumentParser(description="PDF-Parsing via Docling (ohne OCR).")
    ap.add_argument("--attachments-dir", default="data/raw/attachments")
    ap.add_argument("--canonical-json", default=None)
    ap.add_argument("--out", default="data/derivatives/pdf_docling.jsonl")
    ap.add_argument("--export-md", action="store_true", help="zusätzlich Markdown ausgeben")
    ap.add_argument("--skip-existing", action="store_true")
    # Offline / Modelle:
    ap.add_argument("--artifacts-path", default=None,
                    help="Pfad zu vorab geladenen Docling-Modellen (offline). "
                         "Alternativ DOCLING_ARTIFACTS_PATH als Envvar setzen.")
    # Optionale Limits – NUR setzen, wenn du sie wirklich brauchst:
    ap.add_argument("--max-num-pages", type=int, default=None,
                    help="Maximale Seitenzahl pro PDF (int). Nur setzen, wenn gewünscht.")
    ap.add_argument("--max-file-size", type=int, default=None,
                    help="Maximale Dateigröße in Bytes (int). Nur setzen, wenn gewünscht.")
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

    # Pipeline-Optionen inkl. Offline-Modelle
    pipeline_options = PdfPipelineOptions(
        artifacts_path=args.artifacts_path,  # kann None sein -> Docling nutzt Default/Env
        do_ocr=True,
        ocr_options=EasyOcrOptions(
            force_full_page_ocr=True,
            lang=["de","en"]                        # deine Sprachen
        )
    )
    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
    )

    pdfs = find_pdfs(args.attachments_dir)
    done = 0
    with open(args.out, "a", encoding="utf-8") as fout:
        for pdf in pdfs:
            npath = os.path.normpath(pdf)
            if args.skip_existing and npath in existing:
                continue
            try:
                # Nur INT-Parameter setzen, wenn wirklich angegeben
                convert_kwargs: Dict[str, Any] = {}
                if args.max_num_pages is not None:
                    convert_kwargs["max_num_pages"] = args.max_num_pages
                if args.max_file_size is not None:
                    convert_kwargs["max_file_size"] = args.max_file_size

                res = converter.convert(npath, **convert_kwargs)
                doc = res.document

                txt = doc.export_to_text().strip() if hasattr(doc, "export_to_text") else ""
                rec: Dict[str, Any] = {
                    "local_path": npath,
                    "text": txt,
                    "extracted_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "engine": "docling",
                }
                if args.export_md and hasattr(doc, "export_to_markdown"):
                    try:
                        rec["markdown"] = doc.export_to_markdown()
                    except Exception:
                        pass

                rec.update(meta_by_local.get(npath, {}))
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                done += 1
            except Exception as e:
                log(f"WARN: Parsing fehlgeschlagen für {npath}: {e}")

    log(f"Fertig. PDFs gefunden: {len(pdfs)} | erfolgreich geparst: {done} | Output: {args.out}")

if __name__ == "__main__":
    main()
