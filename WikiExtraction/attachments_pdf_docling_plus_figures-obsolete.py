#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Parst PDF-Attachments mit Docling, optional:
- Full-Page-OCR aktivieren (für gescannte PDFs)
- Markdown exportieren
- Figures/Images und Tabellen exportieren (robust: PictureItem/TableItem-Loop)
- OCR über exportierte Figure-Images (EasyOCR)
- Saubere Zuordnung (PDF -> Seite -> Figure) inkl. BBox/Caption
- Mixed-PDF-Erkennung (--check-scan) mit optionaler per-Seite-OCR-Übersteuerung

OFFLINE-NUTZUNG / Modelle:
- Doku: https://docling-project.github.io/docling/usage/advanced_options/
- Modelle vorab ziehen (auf einem Rechner mit Internet):
    docling-tools models download
  Falls SSL/Proxy/CA Probleme: auf anderem Rechner laden und den kompletten
  Modelle-Ordner z. B. nach: data/doclingdata/models kopieren.
- Dann hier per --artifacts-path oder Umgebungsvariable DOCLING_ARTIFACTS_PATH
  auf diesen Ordner zeigen.

WICHTIG für Figure-Export:
- Die Pipeline muss Bilder rendern:
  PdfPipelineOptions.generate_picture_images = True  (und oft images_scale > 1)
  Siehe: Pipeline-Optionen/Images. (Docling Reference)
- Der robuste Weg zum Export (siehe offizielles Beispiel) ist, über
  PictureItem/TableItem zu iterieren und .get_image(doc) zu speichern.
  (Docling Example „Figure export“)

Beispiele:
  python attachments_pdf_docling_plus_figures.py --attachments-dir data/raw/attachments --canonical-json data/raw/confluence.jsonl --out data/derivatives/pdf_docling_plus.jsonl --export-md --export-figures --figures-dir data/derivatives/figures --with-figure-ocr --export-tables --tables-dir data/derivatives/tables --per-page-ocr --ocr-langs de en --images-scale 2.0 --artifacts-path data/doclingdata/models

  python attachments_pdf_docling_plus_figures.py \
    --attachments-dir data/raw/attachments \
    --canonical-json data/raw/confluence.jsonl \
    --out data/derivatives/pdf_docling_plus.jsonl \
    --export-md \
    --export-figures \
    --figures-dir data/derivatives/figures \
    --with-figure-ocr \
    --export-tables \
    --tables-dir data/derivatives/tables \
    --check-scan \
    --scan-threshold-chars 20 \
    --scan-ratio-threshold 0.5 \
    --per-page-ocr \
    --ocr-langs de en \
    --images-scale 2.0 \
    --artifacts-path data/doclingdata/models
"""

from __future__ import annotations
import os, json, time, argparse, glob
from typing import Dict, Any, List, Optional, Tuple

# Docling
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, EasyOcrOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

# In neueren Releases liegen PictureItem/TableItem unter docling_core
try:
    from docling_core.types.doc import PictureItem, TableItem  # robustes Export-API
    from docling_core.types.doc import ImageRefMode           # optional für MD/HTML
except Exception:
    PictureItem = None
    TableItem = None
    ImageRefMode = None

# EasyOCR (nur falls --with-figure-ocr)
try:
    import easyocr  # noqa: F401
    _HAVE_EASYOCR = True
except Exception:
    _HAVE_EASYOCR = False

# Pandas für Tabellen (optional)
try:
    import pandas as pd
    _HAVE_PANDAS = True
except Exception:
    _HAVE_PANDAS = False


def log(msg: str) -> None:
    print(f"[pdf-docling] {msg}")

def ensure_dir(path: str) -> None:
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def list_files(dirpath: str) -> set[str]:
    return set(glob.glob(os.path.join(dirpath, "**", "*"), recursive=True))

def load_canonical_map(canonical_path: Optional[str]) -> Dict[str, Dict[str, Any]]:
    """ Baut Lookup 'local_path' -> Attachment-Metadaten (Confluence). """
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
            page_url = row.get("url")
            for a in (row.get("attachments") or []):
                lp = a.get("local_path")
                if not lp or not lp.lower().endswith(".pdf"):
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

def find_pdfs(attachments_dir: str) -> List[str]:
    out: List[str] = []
    for root, _, files in os.walk(attachments_dir):
        for fn in files:
            if os.path.splitext(fn)[1].lower() == ".pdf":
                out.append(os.path.join(root, fn))
    return out

def make_reader(lang_list: List[str]):
    if not _HAVE_EASYOCR:
        raise RuntimeError("EasyOCR ist nicht installiert: pip install easyocr")
    return easyocr.Reader(lang_list, gpu=False)

def ocr_image_path(reader, img_path: str) -> str:
    try:
        texts = reader.readtext(img_path, detail=0)
        return " ".join([t.strip() for t in texts if isinstance(t, str)]).strip()
    except Exception as e:
        log(f"WARN: Figure-OCR fehlgeschlagen für {img_path}: {e}")
        return ""


def convert_once(pdf_path: str,
                 artifacts_path: Optional[str],
                 do_fullpage_ocr: bool,
                 ocr_langs: List[str],
                 images_scale: float,
                 need_picture_images: bool,
                 need_page_images: bool,
                 max_num_pages: Optional[int],
                 max_file_size: Optional[int]):

    pipeline_options = PdfPipelineOptions(
        artifacts_path=artifacts_path,
    )

    # Bild-Rendering für Export
    if need_picture_images:
        pipeline_options.generate_picture_images = True
        pipeline_options.images_scale = images_scale
    if need_page_images:
        pipeline_options.generate_page_images = True
        # (images_scale wirkt i.d.R. global)

    # Optional: Full-Page-OCR (scans)
    if do_fullpage_ocr:
        pipeline_options.do_ocr = True
        pipeline_options.ocr_options = EasyOcrOptions(
            force_full_page_ocr=True,
            lang=ocr_langs,
        )

    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
    )

    kwargs: Dict[str, Any] = {}
    if max_num_pages is not None:
        kwargs["max_num_pages"] = max_num_pages
    if max_file_size is not None:
        kwargs["max_file_size"] = max_file_size

    return converter.convert(pdf_path, **kwargs)


def main():
    ap = argparse.ArgumentParser(description="PDF → Docling (+ Figures/Tables & optional OCR) für Confluence-Attachments.")
    ap.add_argument("--attachments-dir", default="data/raw/attachments")
    ap.add_argument("--canonical-json", default=None)
    ap.add_argument("--out", default="data/derivatives/pdf_docling_plus.jsonl")
    ap.add_argument("--export-md", action="store_true", help="zusätzlich Markdown speichern")
    ap.add_argument("--skip-existing", action="store_true")

    # Figures
    ap.add_argument("--export-figures", action="store_true", help="Figures/Images als Dateien exportieren")
    ap.add_argument("--figures-dir", default="data/derivatives/figures", help="Zielordner für Figure-Images")
    ap.add_argument("--with-figure-ocr", action="store_true", help="OCR über exportierte Figure-Images laufen lassen")
    ap.add_argument("--ocr-langs", nargs="+", default=["de", "en"], help="Sprachen für OCR (Fullpage/Figures)")
    ap.add_argument("--images-scale", type=float, default=2.0, help="Skalierung für gerenderte Bilder (Default 2.0).")

    # Tables
    ap.add_argument("--export-tables", action="store_true", help="Erkannte Tabellen als CSV/HTML exportieren")
    ap.add_argument("--tables-dir", default="data/derivatives/tables", help="Zielordner für Tabellen-Exporte")

    # Docling Offline/Modelle
    ap.add_argument("--artifacts-path", default=None,
                    help="Pfad zu vorab geladenen Docling-Modellen (offline). "
                         "Alternativ DOCLING_ARTIFACTS_PATH als Envvar setzen.")

    # Docling Optionen
    ap.add_argument("--fullpage-ocr", action="store_true",
                    help="Erzwinge Full-Page-OCR (force_full_page_ocr=True).")
    ap.add_argument("--max-num-pages", type=int, default=None, help="Max Seitenzahl (nur setzen, wenn gewünscht).")
    ap.add_argument("--max-file-size", type=int, default=None, help="Max Dateigröße Bytes (nur setzen, wenn gewünscht).")

    # Mixed-PDF Handling
    ap.add_argument("--check-scan", action="store_true",
                    help="Erkennt gescannte Seiten (sehr wenig Text).")
    ap.add_argument("--scan-threshold-chars", type=int, default=20,
                    help="Seiten mit <N Zeichen gelten als ‚scan-verdächtig‘.")
    ap.add_argument("--scan-ratio-threshold", type=float, default=0.5,
                    help="Wenn Anteil scan-verdächtiger Seiten > Ratio, rerun mit Full-Page-OCR.")
    ap.add_argument("--per-page-ocr", action="store_true",
                    help="Nur gescannte Seiten per OCR überschreiben (setzt generate_page_images).")

    args = ap.parse_args()

    ensure_dir(os.path.dirname(args.out) or ".")
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

    # Optional OCR-Reader für Figures
    reader = None
    if args.export_figures and args.with_figure_ocr:
        reader = make_reader(args.ocr_langs)

    # Für Table-Export Ordner vorbereiten (nur wenn auch gebraucht)
    if args.export_tables:
        ensure_dir(args.tables_dir)

    pdfs = find_pdfs(args.attachments_dir)
    done = 0

    with open(args.out, "a", encoding="utf-8") as fout:
        for pdf in pdfs:
            npath = os.path.normpath(pdf)
            if args.skip_existing and npath in existing:
                continue

            try:
                need_pic_imgs = bool(args.export_figures)
                need_page_imgs = bool(args.check_scan and args.per_page_ocr)

                # 1) Erster Durchlauf (ohne Full-Page-OCR, außer explizit angefordert)
                conv = convert_once(
                    pdf_path=npath,
                    artifacts_path=args.artifacts_path,
                    do_fullpage_ocr=args.fullpage_ocr,
                    ocr_langs=args.ocr_langs,
                    images_scale=args.images_scale,
                    need_picture_images=need_pic_imgs,
                    need_page_images=need_page_imgs,
                    max_num_pages=args.max_num_pages,
                    max_file_size=args.max_file_size,
                )
                doc = conv.document

                # 2) Optional: Scan-Check (pro Seite grob an Text-Länge prüfen)
                scan_page_idxs: List[int] = []
                if args.check_scan:
                    # Heuristik: Page-Text über layoutierte Items zusammenbauen (fallback simpel)
                    page_count = getattr(doc, "num_pages", None) or len(getattr(doc, "pages", {}))
                    for pno, page in doc.pages.items():
                        # naive Textsammlung aus Paragraphs/Lines (fallback auf empty)
                        page_text = ""
                        try:
                            # docling liefert meist page.lines / page.paragraphs — je nach Version.
                            if hasattr(page, "to_text"):
                                page_text = page.to_text()
                            else:
                                # Fallback: iterate items and concat texts
                                for item, _lvl in doc.iterate_items(page_filter=[page]):
                                    t = getattr(item, "text", None)
                                    if isinstance(t, str):
                                        page_text += t + "\n"
                        except Exception:
                            pass
                        if len((page_text or "").strip()) < args.scan_threshold_chars:
                            scan_page_idxs.append(pno)

                    # Wenn ‚viele‘ Seiten leer -> Full-Page-OCR Gesamtdokument neu laufen lassen
                    total_pages = max(len(doc.pages), 1)
                    if (len(scan_page_idxs) / total_pages) > args.scan_ratio_threshold and not args.fullpage_ocr:
                        log(f"Scan-Check: {len(scan_page_idxs)}/{total_pages} leer ⇒ Rerun mit Full-Page-OCR")
                        conv = convert_once(
                            pdf_path=npath,
                            artifacts_path=args.artifacts_path,
                            do_fullpage_ocr=True,
                            ocr_langs=args.ocr_langs,
                            images_scale=args.images_scale,
                            need_picture_images=need_pic_imgs,
                            need_page_images=need_page_imgs,
                            max_num_pages=args.max_num_pages,
                            max_file_size=args.max_file_size,
                        )
                        doc = conv.document
                        scan_page_idxs = []  # neu bewertet; lassen wir jetzt leer
                    elif args.per_page_ocr and scan_page_idxs:
                        # per-Seite OCR: wir brauchen Page-Images
                        if not need_page_imgs:
                            # Page-Images nachträglich verfügbar machen → erneuter Convert mit Page-Images
                            conv = convert_once(
                                pdf_path=npath,
                                artifacts_path=args.artifacts_path,
                                do_fullpage_ocr=args.fullpage_ocr,
                                ocr_langs=args.ocr_langs,
                                images_scale=args.images_scale,
                                need_picture_images=need_pic_imgs,
                                need_page_images=True,
                                max_num_pages=args.max_num_pages,
                                max_file_size=args.max_file_size,
                            )
                            doc = conv.document

                # 3) Text / Markdown
                text_out = doc.export_to_text().strip() if hasattr(doc, "export_to_text") else ""
                rec: Dict[str, Any] = {
                    "local_path": npath,
                    "engine": "docling",
                    "text": text_out,
                    "extracted_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                }
                if args.export_md and hasattr(doc, "export_to_markdown"):
                    try:
                        rec["markdown"] = doc.export_to_markdown()
                    except Exception:
                        pass

                # 4) Per-Page-OCR-Overrides (nur wenn angefordert und Seiten verdächtig)
                if args.check_scan and args.per_page_ocr and scan_page_idxs:
                    overrides: List[Dict[str, Any]] = []
                    # page.image.pil_image laut Figure-Export/Pages-API (wenn generate_page_images=True)
                    for pno in scan_page_idxs:
                        page = doc.pages.get(pno)
                        try:
                            if page and hasattr(page, "image") and hasattr(page.image, "pil_image"):
                                if reader is None:
                                    reader = make_reader(args.ocr_langs)
                                # Temporär als PNG wegspeichern ist nicht nötig; EasyOCR kann PIL nicht direkt,
                                # deshalb hier kurzer Weg über PNG-Datei:
                                tmpdir = os.path.join(os.path.dirname(args.out), "_tmp_pages")
                                ensure_dir(tmpdir)
                                tmp_path = os.path.join(tmpdir, f"{os.path.splitext(os.path.basename(npath))[0]}-p{pno}.png")
                                page.image.pil_image.save(tmp_path, format="PNG")
                                ocr_txt = ocr_image_path(reader, tmp_path)
                                overrides.append({"page_index": pno, "ocr_text": ocr_txt})
                        except Exception as e:
                            log(f"WARN: per-Page OCR Fehlgeschlagen p{pno} ({npath}): {e}")
                    if overrides:
                        rec["ocr_overrides"] = overrides

                # 5) Figures exportieren (robust)
                if args.export_figures:
                    figures_dir_for_doc = os.path.join(
                        args.figures_dir, os.path.splitext(os.path.basename(npath))[0]
                    )
                    figures_data: List[Dict[str, Any]] = []
                    exported_any = False

                    # a) bevorzugt: offizielles robustes Pattern (iterate_items + get_image)
                    if PictureItem is not None and hasattr(doc, "iterate_items"):
                        ensure_dir(figures_dir_for_doc)
                        pic_idx = 0
                        for element, _lvl in doc.iterate_items():
                            if isinstance(element, PictureItem):
                                pic_idx += 1
                                fn = os.path.join(figures_dir_for_doc, f"picture-{pic_idx}.png")
                                try:
                                    im = element.get_image(doc)
                                    im.save(fn, "PNG")
                                    exported_any = True
                                    entry: Dict[str, Any] = {
                                        "path": fn.replace("\\", "/"),
                                        "page_index": getattr(getattr(element, "page_ref", None), "page_idx", None),
                                        "bbox": getattr(element, "bbox", None),
                                        "caption": getattr(element, "caption_text", None),
                                    }
                                    if reader is not None:
                                        entry["ocr_text"] = ocr_image_path(reader, fn)
                                    figures_data.append(entry)
                                except Exception as e:
                                    log(f"WARN: Picture export failed ({npath}): {e}")

                    # b) Fallback: Falls vorhanden, zusätzlich/alternativ export_figure_images nutzen
                    if hasattr(doc, "export_figure_images"):
                        before = set()
                        if not exported_any:
                            ensure_dir(figures_dir_for_doc)
                            before = list_files(figures_dir_for_doc)
                        try:
                            doc.export_figure_images(figures_dir_for_doc)
                            after = list_files(figures_dir_for_doc)
                            created = sorted(list(after - before))
                            for idx, fig_path in enumerate([p for p in created if os.path.isfile(p)]):
                                entry = {"path": fig_path.replace("\\", "/")}
                                # keine sichere Zuordnung -> minimaler Datensatz
                                if reader is not None:
                                    entry["ocr_text"] = ocr_image_path(reader, fig_path)
                                figures_data.append(entry)
                                exported_any = True
                        except Exception as e:
                            log(f"WARN: export_figure_images failed ({npath}): {e}")

                    if exported_any and figures_data:
                        rec["figures"] = figures_data
                    else:
                        # Leeren Ordner entfernen, wenn nix exportiert
                        try:
                            if os.path.isdir(figures_dir_for_doc) and not any(os.scandir(figures_dir_for_doc)):
                                os.rmdir(figures_dir_for_doc)
                        except Exception:
                            pass

                # 6) Tabellen exportieren (CSV + HTML)
                if args.export_tables and TableItem is not None:
                    tables_dir_for_doc = os.path.join(
                        args.tables_dir, os.path.splitext(os.path.basename(npath))[0]
                    )
                    ensure_dir(tables_dir_for_doc)
                    table_idx = 0
                    tables_meta: List[Dict[str, Any]] = []
                    for element, _lvl in doc.iterate_items():
                        if isinstance(element, TableItem):
                            table_idx += 1
                            base = os.path.join(tables_dir_for_doc, f"table-{table_idx}")
                            # CSV (wenn pandas verfügbar)
                            if _HAVE_PANDAS:
                                try:
                                    df = element.export_to_dataframe()
                                    csv_path = base + ".csv"
                                    df.to_csv(csv_path, index=False)
                                    tables_meta.append({"csv": csv_path.replace("\\", "/")})
                                except Exception as e:
                                    log(f"WARN: table CSV export failed ({npath}): {e}")
                            # HTML (ohne pandas möglich)
                            try:
                                html_path = base + ".html"
                                html_str = element.export_to_html(doc=doc)
                                with open(html_path, "w", encoding="utf-8") as fp:
                                    fp.write(html_str)
                                if tables_meta and "csv" in tables_meta[-1]:
                                    tables_meta[-1]["html"] = html_path.replace("\\", "/")
                                else:
                                    tables_meta.append({"html": html_path.replace("\\", "/")})
                            except Exception as e:
                                log(f"WARN: table HTML export failed ({npath}): {e}")
                    if tables_meta:
                        rec["tables"] = tables_meta

                # 7) Confluence-Metadaten
                rec.update(meta_by_local.get(npath, {}))

                # 8) Schreiben
                with open(args.out, "a", encoding="utf-8") as fout:
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                done += 1

            except Exception as e:
                log(f"WARN: Parsing fehlgeschlagen für {npath}: {e}")

    log(f"Fertig. PDFs gefunden: {len(pdfs)} | erfolgreich verarbeitet: {done} | Output: {args.out}")
    

if __name__ == "__main__":
    main()