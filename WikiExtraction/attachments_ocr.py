#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Liest gespeicherte Attachments (aus canonical_extractor.py --with-attachments)
und führt OCR mit EasyOCR auf Bilddateien aus. Ergebnisse werden als JSONL
geschrieben (1 Zeile = 1 Attachment-OCR-Result).

Beispiel:
  python attachments_ocr.py --attachments-dir data/raw/attachments --canonical-json data/raw/confluence.jsonl --out data/derivatives/ocr.jsonl --langs de en --gpu

  python attachments_ocr.py \
    --attachments-dir data/raw/attachments \
    --canonical-json data/raw/conf_with_atts.jsonl \
    --out data/derivatives/ocr.jsonl \
    --langs de en \
    --with-boxes

Unterstützte Bild-Endungen: .png .jpg .jpeg .bmp .tif .tiff .webp
PDF/DRAWIO werden geloggt, aber ausgelassen (dafür gibt es eigene Skripte).

EasyOCR: https://github.com/JaidedAI/EasyOCR

- Fallback-Ermittlung der page_id aus dem Pfad unterhalb von --attachments-dir.
- Auch bei --with-boxes wird zusätzlich ein zusammengefasster ocr_text gespeichert.
- Am Laufzeitende werden Konfidenz-Metriken ausgegeben (mean, median, p10/p90),
  sofern --with-boxes verwendet wurde.
"""

from __future__ import annotations
import os
import sys
import json
import time
import argparse
import mimetypes
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
    try:
        return int(x)
    except Exception:
        return int(float(x))

def _to_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return float(str(x))

def _normalize_bbox(bbox) -> List[List[int]]:
    norm: List[List[int]] = []
    try:
        for p in bbox:
            x, y = p[0], p[1]
            norm.append([_to_int(x), _to_int(y)])
    except Exception:
        norm = []
    return norm

def load_canonical_map(canonical_path: Optional[str]) -> Dict[str, Dict[str, Any]]:
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
                    "page_id": str(page_id) if page_id is not None else None,
                    "page_title": page_title,
                    "page_url": page_url,
                    "attachment_id": a.get("id"),
                    "title": a.get("title"),
                    "mediaType": a.get("mediaType"),
                    "fileSize": a.get("fileSize"),
                }
    return by_local

def load_page_meta_by_id(canonical_path: Optional[str]) -> Dict[str, Dict[str, Any]]:
    meta: Dict[str, Dict[str, Any]] = {}
    if not canonical_path or not os.path.exists(canonical_path):
        return meta
    with open(canonical_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            pid = row.get("id")
            if pid is None:
                continue
            meta[str(pid)] = {
                "page_title": row.get("title"),
                "page_url": row.get("url"),
            }
    return meta

def find_attachments(attachments_dir: str) -> List[str]:
    paths: List[str] = []
    for root, _, files in os.walk(attachments_dir):
        for fn in files:
            paths.append(os.path.join(root, fn))
    return paths

def infer_page_id_from_path(file_path: str, attachments_dir: str) -> Optional[str]:
    try:
        rel = os.path.relpath(file_path, attachments_dir)
    except Exception:
        return None
    parts = rel.split(os.sep)
    if not parts:
        return None
    candidate = parts[0]
    return candidate if candidate.isdigit() else None

def do_easy_ocr(reader: "easyocr.Reader", path: str, with_boxes: bool):
    """
    with_boxes=False  -> gibt reinen Text (string) zurück
    with_boxes=True   -> gibt {"items":[{"bbox":[[x,y]..], "text":str, "confidence":float}, ...]} zurück
    """
    if with_boxes:
        triples = reader.readtext(path, detail=1)
        items = []
        for triple in triples:
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
        texts = reader.readtext(path, detail=0)
        return " ".join([str(t).strip() for t in texts if isinstance(t, str)]).strip()

def percentiles(values: List[float], ps: List[float]) -> List[float]:
    if not values:
        return [0.0 for _ in ps]
    v = sorted(values)
    out = []
    for p in ps:
        if p <= 0:
            out.append(v[0]); continue
        if p >= 100:
            out.append(v[-1]); continue
        k = (len(v)-1) * (p/100.0)
        f = int(k)
        c = min(f+1, len(v)-1)
        if f == c:
            out.append(v[f])
        else:
            d0 = v[f] * (c - k)
            d1 = v[c] * (k - f)
            out.append(d0 + d1)
    return out

# --------------------- Main ---------------------

def main():
    start_time = time.time()
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
    page_meta_by_id = load_page_meta_by_id(args.canonical_json)
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

    reader = easyocr.Reader(args.langs, gpu=bool(args.gpu))

    count_total_files         = 0
    count_images              = 0
    count_ocr_images          = 0
    count_images_with_text    = 0
    count_errors              = 0

    conf_all_items: List[float] = []

    with open(args.out, "a", encoding="utf-8") as out:
        for apath in find_attachments(args.attachments_dir):
            count_total_files += 1
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

            count_images += 1

            try:
                ocr_result = do_easy_ocr(reader, npath, args.with_boxes)
                count_ocr_images += 1

                rec: Dict[str, Any] = {
                    "local_path": npath,
                    "extracted_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "langs": args.langs,
                }
                # Canonical-Metadaten (per local_path)
                rec.update(meta_by_local.get(npath, {}))

                # Fallback: page_id aus Pfad + Seiten-Meta
                if not rec.get("page_id"):
                    pid = infer_page_id_from_path(npath, args.attachments_dir)
                    if pid:
                        rec["page_id"] = pid
                        if pid in page_meta_by_id:
                            rec.setdefault("page_title", page_meta_by_id[pid].get("page_title"))
                            rec.setdefault("page_url",   page_meta_by_id[pid].get("page_url"))

                # Best-Effort-Felder
                rec.setdefault("title", os.path.basename(npath))
                mt, _ = mimetypes.guess_type(npath)
                if mt and not rec.get("mediaType"):
                    rec["mediaType"] = mt

                if args.with_boxes:
                    items = ocr_result.get("items", []) if isinstance(ocr_result, dict) else []
                    rec["ocr_boxes"] = items
                    # immer auch ocr_text mitschreiben (zusammengefasst), damit join_all_pages es anhängen kann
                    concat_text = " ".join([(it.get("text") or "").strip() for it in items if isinstance(it, dict)]).strip()
                    rec["ocr_text"] = concat_text

                    if concat_text:
                        count_images_with_text += 1
                    for it in items:
                        try:
                            conf_all_items.append(_to_float(it.get("confidence", 0.0)))
                        except Exception:
                            pass
                else:
                    text = ocr_result if isinstance(ocr_result, str) else ""
                    rec["ocr_text"] = text
                    if text.strip():
                        count_images_with_text += 1

                out.write(json.dumps(rec, ensure_ascii=False) + "\n")

            except Exception as e:
                log(f"WARN: OCR fehlgeschlagen für {npath}: {e}")
                count_errors += 1

    log(f"Fertig. Dateien gesamt gescannt: {count_total_files}, OCR auf Bildern: {count_ocr_images}, Output: {args.out}")

    elapsed = time.time() - start_time
    print("\n=== Zusammenfassung (OCR Attachments) ===")
    print(f"Bild-Attachments gefunden     : {count_images}")
    print(f"OCR durchgeführt (Bilder)     : {count_ocr_images}")
    print(f"Bilder mit erkanntem Text     : {count_images_with_text}")
    print(f"Fehler (OCR)                  : {count_errors}")
    print(f"Laufzeit                      : {elapsed:.2f}s")

    if args.with_boxes:
        if conf_all_items:
            mean = sum(conf_all_items) / len(conf_all_items)
            med = percentiles(conf_all_items, [50])[0]
            p10, p90 = percentiles(conf_all_items, [10, 90])
            print("\n— Konfidenzmetriken (nur verfügbar mit --with-boxes) —")
            print(f"Items gesamt                  : {len(conf_all_items)}")
            print(f"mean={mean:.3f}  median={med:.3f}  p10={p10:.3f}  p90={p90:.3f}")
        else:
            print("\n— Konfidenzmetriken —")
            print("Keine Konfidenzwerte erfasst (keine Items).")
    else:
        print("\n(Hinweis: Konfidenzmetriken nur verfügbar, wenn du mit --with-boxes läufst.)")

if __name__ == "__main__":
    main()
