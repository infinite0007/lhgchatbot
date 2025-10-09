#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Teil B der Pipeline:
- Liest die JSONL aus Teil A (z. B. pdf_docling_prepared.jsonl)
- Für jede dort aufgeführte Figure (PNG) wird EasyOCR ausgeführt
- Ergebnisse als JSONL (1 Zeile = 1 Figure-OCR-Result) geschrieben

Eigenschaften:
- Sequentiell pro PDF: erst alle Figures eines PDFs, dann nächstes PDF
- Optional: Bounding-Boxes + Confidence speichern (--with-boxes)
- Skip-Cache (--skip-existing) auf Basis der Figure-Pfade
- Min-Confidence-Filter und einfache Entduplizierung von Zeilen
- Robust gegen fehlende/gelöschte Figure-Dateien

Beispiel:
  python attachments_pdf_docling_prepareB_figuresocr.py --prepared-jsonl data/derivatives/pdf_docling_prepared.jsonl --out data/derivatives/prepared_figures_ocr.jsonl --langs de en --with-boxes --min-conf 0.25 --skip-existing --gpu

  python attachments_pdf_docling_prepareB_figuresocr.py \
    --prepared-jsonl data/derivatives/pdf_docling_prepared.jsonl \
    --out data/derivatives/prepared_figures_ocr.jsonl \
    --langs de en \
    --with-boxes \
    --min-conf 0.25 \
    --skip-existing
"""

from __future__ import annotations
import os
import json
import time
import argparse
from typing import Dict, Any, List, Optional, Iterable, Tuple

# --------------------- EasyOCR ---------------------
try:
    import easyocr
except Exception:
    print("EasyOCR ist nicht installiert. Bitte: pip install easyocr")
    raise

# --------------------- Helpers ---------------------

def log(msg: str) -> None:
    print(f"[fig-ocr] {msg}")

def normpath(p: str) -> str:
    return os.path.normpath(p)

def load_prepared_lines(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue

def dedupe_preserve_order(lines: List[str]) -> List[str]:
    seen = set()
    out = []
    for l in lines:
        s = l.strip()
        if not s:
            continue
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out

def do_easyocr_text(reader: "easyocr.Reader", img_path: str, with_boxes: bool,
                    min_conf: float, dedupe: bool) -> Dict[str, Any]:
    """
    Rückgabe:
      with_boxes=False -> {"text":"..."}
      with_boxes=True  -> {"items":[{"bbox":[[x,y]..], "text":"", "confidence":float}, ...],
                           "text":"... (zusammengefasst, gefiltert)"}
    """
    if with_boxes:
        triples = reader.readtext(img_path, detail=1)  # [(bbox, text, conf), ...]
        items = []
        texts_for_concat = []
        for t in triples:
            if not isinstance(t, (list, tuple)) or len(t) < 3:
                continue
            bbox, txt, conf = t[0], t[1], t[2]
            try:
                conf_f = float(conf)
            except Exception:
                conf_f = 0.0
            if conf_f < min_conf:
                continue

            # bbox zu int
            norm_bbox: List[List[int]] = []
            try:
                for p in bbox:
                    norm_bbox.append([int(p[0]), int(p[1])])
            except Exception:
                norm_bbox = []

            s_txt = str(txt).strip() if txt is not None else ""
            if s_txt:
                texts_for_concat.append(s_txt)

            items.append({
                "bbox": norm_bbox,
                "text": s_txt,
                "confidence": conf_f,
            })

        if dedupe:
            texts_for_concat = dedupe_preserve_order(texts_for_concat)

        return {
            "items": items,
            "text": " ".join(texts_for_concat).strip()
        }
    else:
        texts = reader.readtext(img_path, detail=0)  # [ "text", ... ]
        joined = " ".join([str(t).strip() for t in texts if isinstance(t, str)]).strip()
        if dedupe:
            # grobe Zeilen-Entduplizierung (split an mehrfache Spaces)
            parts = [p for p in joined.replace("\n", " ").split(" ") if p.strip()]
            joined = " ".join(dedupe_preserve_order(parts))
        return {"text": joined}

# --------------------- Main ---------------------

def main():
    ap = argparse.ArgumentParser(description="Figure-OCR (EasyOCR) für Docling-Prepared-Ausgabe (Teil B).")
    ap.add_argument("--prepared-jsonl", required=True,
                    help="JSONL aus Teil A (pdf_docling_prepared.jsonl)")
    ap.add_argument("--out", default="data/derivatives/figures_ocr.jsonl",
                    help="Ausgabe JSONL (pro Figure ein Record)")
    ap.add_argument("--langs", nargs="+", default=["de", "en"],
                    help="OCR-Sprachen (EasyOCR), z. B. de en")
    ap.add_argument("--with-boxes", action="store_true",
                    help="Detailausgabe mit Bounding-Boxes + Confidence")
    ap.add_argument("--min-conf", type=float, default=0.20,
                    help="Mindest-Confidence (nur bei --with-boxes wirksam)")
    ap.add_argument("--dedupe-lines", action="store_true",
                    help="Einfache Entduplizierung zusammengefasster Texte")
    ap.add_argument("--gpu", action="store_true",
                    help="GPU für EasyOCR verwenden (falls vorhanden)")
    ap.add_argument("--skip-existing", action="store_true",
                    help="Bereits geloggte Figure-Pfade überspringen (aus --out gelesen)")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    # Skip-Cache laden
    existing_figs: set[str] = set()
    if args.skip_existing and os.path.exists(args.out):
        with open(args.out, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    row = json.loads(line)
                    figp = row.get("figure_path")
                    if figp:
                        existing_figs.add(normpath(figp))
                except Exception:
                    continue
        log(f"Cache aktiv: {len(existing_figs)} Figure-OCR Einträge vorhanden (werden übersprungen).")

    # EasyOCR Reader
    reader = easyocr.Reader(args.langs, gpu=bool(args.gpu))

    # Wir verarbeiten sequentiell pro PDF: Wir nutzen die Reihenfolge der prepared JSONL.
    processed_pdf = 0
    processed_figs = 0
    skipped_missing = 0

    with open(args.out, "a", encoding="utf-8") as fout:
        for rec in load_prepared_lines(args.prepared_jsonl):
            pdf_path = rec.get("local_path")
            figures = rec.get("figures") or []

            # pro PDF erst alle Figures…
            if not figures:
                continue

            for fig in figures:
                fig_path = normpath(str(fig.get("path", "")))
                if not fig_path:
                    continue
                if args.skip_existing and fig_path in existing_figs:
                    continue
                if not os.path.exists(fig_path):
                    skipped_missing += 1
                    log(f"WARN: Figure fehlt auf Platte, ausgelassen: {fig_path}")
                    continue

                try:
                    ocr_res = do_easyocr_text(
                        reader=reader,
                        img_path=fig_path,
                        with_boxes=args.with_boxes,
                        min_conf=args.min_conf,
                        dedupe=args.dedupe_lines,
                    )
                    out_row: Dict[str, Any] = {
                         "pdf_local_path": normpath(pdf_path) if pdf_path else None,
                         "figure_path": fig_path,
                         "extracted_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                         "langs": args.langs,
                         "page_index": fig.get("page_index"),
                         "caption": fig.get("caption"),
                        "source_type": "pdf_figure_ocr",
                     }
                    # Wichtige Confluence-Metadaten aus dem Prepared-Record übernehmen:
                    for k in ("page_id", "page_title", "page_url",
                              "attachment_id", "title", "mediaType", "fileSize"):
                        if k in rec:
                            out_row[k] = rec.get(k)

                    if args.with_boxes:
                        out_row["ocr_items"] = ocr_res.get("items", [])
                        out_row["ocr_text"] = ocr_res.get("text", "")
                    else:
                        out_row["ocr_text"] = ocr_res.get("text", "")

                    fout.write(json.dumps(out_row, ensure_ascii=False) + "\n")
                    processed_figs += 1

                except Exception as e:
                    log(f"WARN: OCR fehlgeschlagen für {fig_path}: {e}")

            processed_pdf += 1  # dieses PDF ist komplett

    log(f"Fertig. PDFs mit Figures: {processed_pdf} | Figures OCR: {processed_figs} | Fehlende Figures: {skipped_missing} | Output: {args.out}")


if __name__ == "__main__":
    main()
