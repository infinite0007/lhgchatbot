#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
attachments_ocr.py
------------------
Liest gespeicherte Attachments (aus canonical_extractor.py --with-attachments)
und führt OCR mit EasyOCR auf Bilddateien aus. Ergebnisse werden als JSONL
geschrieben (1 Zeile = 1 Attachment-OCR-Result).

Beispiel:
  python attachments_ocr.py --attachments-dir data/raw/attachments --canonical-json data/raw/conf_with_atts.jsonl --out data/derivatives/ocr.jsonl --langs de

  python attachments_ocr.py \
    --attachments-dir data/raw/attachments \
    --canonical-json data/raw/conf_with_atts.jsonl \
    --out data/derivatives/ocr.jsonl \
    --langs de en \
    --with-boxes

Unterstützte Bild-Endungen: .png .jpg .jpeg .bmp .tif .tiff .webp
PDF/DRAWIO werden geloggt, aber ausgelassen (dafür gibt es eigene Skripte).

EasyOCR: https://github.com/JaidedAI/EasyOCR
"""

from __future__ import annotations
import os
import sys
import json
import time
import argparse
from typing import Dict, Any, List, Optional

# --------------------- EasyOCR ---------------------
try:
    import easyocr
except Exception:
    print("EasyOCR ist nicht installiert. Bitte: pip install easyocr")
    raise

# --------------------- Helpers ---------------------

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

def log(msg: str) -> None:
    print(f"[ocr] {msg}")

def _to_int(x) -> int:
    """Sicher zu Python-int casten (gegen numpy.int32/64 etc.)."""
    try:
        return int(x)
    except Exception:
        return int(float(x))

def _to_float(x) -> float:
    """Sicher zu Python-float casten (gegen numpy.float32/64 etc.)."""
    try:
        return float(x)
    except Exception:
        return float(str(x))

def _normalize_bbox(bbox) -> List[List[int]]:
    """
    EasyOCR bbox ist i.d.R. Liste/Tuple von 4 Punkten:
    [(x1,y1),(x2,y2),(x3,y3),(x4,y4)] – wir casten alles auf eingebaute int.
    """
    norm: List[List[int]] = []
    try:
        for p in bbox:
            x, y = p[0], p[1]
            norm.append([_to_int(x), _to_int(y)])
    except Exception:
        # Fallback: leere Box, falls Format unerwartet
        norm = []
    return norm

def load_canonical_map(canonical_path: Optional[str]) -> Dict[str, Dict[str, Any]]:
    """
    Baut ein Lookup 'local_path' -> Attachment-Metadaten
    (inkl. page_id, page_title, page_url, attachment_id, title, mediaType, fileSize).
    """
    by_local: Dict[str, Dict[str, Any]] = {}
    if not canonical_path or not os.path.exists(canonical_path):
        return by_local

    with open(canonical_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue

            page_id = row.get("id")
            page_title = row.get("title")
            page_url = row.get("url")
            atts = row.get("attachments", []) or []

            for a in atts:
                lp = a.get("local_path")
                if not lp:
                    continue
                key = os.path.normpath(lp)
                by_local[key] = {
                    "page_id": page_id,
                    "page_title": page_title,
                    "page_url": page_url,
                    "attachment_id": a.get("id"),
                    "title": a.get("title"),
                    "mediaType": a.get("mediaType"),
                    "fileSize": a.get("fileSize"),
                }
    return by_local

def find_attachments(attachments_dir: str) -> List[str]:
    paths: List[str] = []
    for root, _, files in os.walk(attachments_dir):
        for fn in files:
            paths.append(os.path.join(root, fn))
    return paths

def do_easy_ocr(reader: "easyocr.Reader", path: str, with_boxes: bool) -> dict | str:
    """
    with_boxes=False  -> gibt reinen Text (string) zurück
    with_boxes=True   -> gibt {"items":[{"bbox":[[x,y]..], "text":str, "confidence":float}, ...]} zurück
    """
    if with_boxes:
        triples = reader.readtext(path, detail=1)  # [ [ (x,y)*4 ], text, conf ] je Treffer
        items = []
        for triple in triples:
            # defensive unpacking, falls einzelne Reader-Backends leicht andere Formen liefern
            if not isinstance(triple, (list, tuple)) or len(triple) < 3:
                continue
            bbox, text, conf = triple[0], triple[1], triple[2]
            items.append({
                "bbox": _normalize_bbox(bbox),
                "text": str(text) if text is not None else "",
                "confidence": _to_float(conf),
            })
        return {"items": items}
    else:
        texts = reader.readtext(path, detail=0)  # Liste von Strings
        return " ".join([str(t).strip() for t in texts if isinstance(t, str)]).strip()

# --------------------- Main ---------------------

def main():
    ap = argparse.ArgumentParser(description="OCR für gespeicherte Confluence-Attachments (EasyOCR).")
    ap.add_argument("--attachments-dir", default="data/raw/attachments", help="Wurzelordner der Attachments")
    ap.add_argument("--canonical-json", default=None, help="Pfad zur canonical JSONL (für Metadaten-Anreicherung)")
    ap.add_argument("--out", default="data/derivatives/ocr.jsonl", help="Ausgabe JSONL")
    ap.add_argument("--langs", nargs="+", default=["de", "en"], help="OCR-Sprachen, z.B.: de en")
    ap.add_argument("--skip-existing", action="store_true", help="Bestehende Ausgabedatei als Cache nutzen (Pfad-Matching)")
    ap.add_argument("--with-boxes", action="store_true", help="OCR mit detail=1 (Bounding-Boxes + Confidence) speichern")
    ap.add_argument("--gpu", action="store_true", help="GPU für EasyOCR verwenden (falls vorhanden)")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    meta_by_local = load_canonical_map(args.canonical_json)
    log(f"Metadaten geladen für {len(meta_by_local)} Attachments (aus canonical).")

    existing: set[str] = set()
    if args.skip_existing and os.path.exists(args.out):
        with open(args.out, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    row = json.loads(line)
                    if row.get("local_path"):
                        existing.add(os.path.normpath(row["local_path"]))
                except Exception:
                    continue
        log(f"Cache aktiv: {len(existing)} OCR-Einträge vorhanden (werden übersprungen).")

    # EasyOCR Reader
    reader = easyocr.Reader(args.langs, gpu=bool(args.gpu))

    count_total = 0
    count_ocr = 0

    with open(args.out, "a", encoding="utf-8") as out:
        for apath in find_attachments(args.attachments_dir):
            count_total += 1
            npath = os.path.normpath(apath)

            if args.skip_existing and npath in existing:
                continue

            ext = os.path.splitext(npath)[1].lower()
            if ext not in IMAGE_EXTS:
                if ext in {".pdf", ".drawio"}:
                    meta = meta_by_local.get(npath, {})
                    page_ref = meta.get("page_url") or meta.get("page_id") or "?"
                    log(f"Info: ausgelassen (kein Bild-OCR): {npath} (Seite: {page_ref})")
                continue

            try:
                ocr_result = do_easy_ocr(reader, npath, args.with_boxes)
                rec: Dict[str, Any] = {
                    "local_path": npath,
                    "extracted_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "langs": args.langs,
                }
                # Seite/Attachment-Metadaten anreichern (Link für spätere Navigation)
                rec.update(meta_by_local.get(npath, {}))

                if args.with_boxes:
                    # {"items":[{bbox:[...], text:"", confidence:0.0}, ...]}
                    rec["ocr_boxes"] = ocr_result.get("items", []) if isinstance(ocr_result, dict) else []
                else:
                    rec["ocr_text"] = ocr_result if isinstance(ocr_result, str) else ""

                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                count_ocr += 1
            except Exception as e:
                log(f"WARN: OCR fehlgeschlagen für {npath}: {e}")

    log(f"Fertig. Dateien gesamt gescannt: {count_total}, OCR durchgeführt: {count_ocr}, Output: {args.out}")

if __name__ == "__main__":
    main()
