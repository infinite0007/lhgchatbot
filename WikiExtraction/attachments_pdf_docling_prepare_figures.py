#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import os, json, time, argparse
from pathlib import Path
from typing import Dict, Any, List, Optional

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

# python attachments_pdf_docling_prepare_figures.py --attachments-dir data/raw/attachments --canonical-json data/raw/confluence.jsonl --out data/derivatives/pdf_docling_prepared.jsonl --export-md --export-figures --figures-dir data/derivatives/figures --export-tables  --tables-dir data/derivatives/tables --images-scale 2.0 --artifacts-path data/doclingdata/models
# robuste Typ-Imports (je nach Docling-Version vorhanden)
try:
    from docling_core.types.doc import PictureItem, TableItem
except Exception:
    PictureItem = None
    TableItem = None

def log(msg: str) -> None:
    print(f"[prepare] {msg}")

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def safe_attr(obj, name: str):
    """Attribut sicher holen; Callables ggf. aufrufen; bei Fehler -> None."""
    try:
        val = getattr(obj, name, None)
        if callable(val):
            try:
                return val()
            except Exception:
                return None
        return val
    except Exception:
        return None

def _json_default(o):
    """Letzte Rettung: Nicht-JSON-Typen in String umwandeln."""
    try:
        from pathlib import Path
        if isinstance(o, Path):
            return str(o)
    except Exception:
        pass
    return str(o)

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

def find_pdfs(attachments_dir: str) -> List[Path]:
    root = Path(attachments_dir)
    return list(root.rglob("*.pdf"))

def main():
    ap = argparse.ArgumentParser(description="Phase A: Docling-Prepare (Text/MD + Figures + Tables).")
    ap.add_argument("--attachments-dir", default="data/raw/attachments")
    ap.add_argument("--canonical-json", default=None)
    ap.add_argument("--out", default="data/derivatives/pdf_docling_prepared.jsonl")
    ap.add_argument("--export-md", action="store_true")
    ap.add_argument("--skip-existing", action="store_true")

    # Exporte
    ap.add_argument("--export-figures", action="store_true")
    ap.add_argument("--figures-dir", default="data/derivatives/figures")
    ap.add_argument("--export-tables", action="store_true")
    ap.add_argument("--tables-dir", default="data/derivatives/tables")

    # Render/Qualität
    ap.add_argument("--images-scale", type=float, default=2.0)

    # Offline / Modelle
    ap.add_argument("--artifacts-path", default=None)

    args = ap.parse_args()

    out_path = Path(args.out)
    ensure_dir(out_path.parent)
    meta_by_local = load_canonical_map(args.canonical_json)

    # Docling-Pipeline
    pipeline_options = PdfPipelineOptions(
        artifacts_path=args.artifacts_path,
        images_scale=args.images_scale,
    )
    if args.export_figures:
        pipeline_options.generate_picture_images = True  # nötig für Figure-Bitmaps

    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
    )

    # Cache bereits verarbeiteter Pfade (optional)
    existing: set[str] = set()
    if args.skip_existing and out_path.exists():
        with out_path.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    row = json.loads(line)
                    if row.get("local_path"):
                        existing.add(os.path.normpath(row["local_path"]))
                except Exception:
                    pass

    pdfs = find_pdfs(args.attachments_dir)
    done = 0

    with out_path.open("a", encoding="utf-8") as fout:
        for pdf in pdfs:
            npath = os.path.normpath(str(pdf))
            if args.skip_existing and npath in existing:
                continue
            try:
                conv_res = converter.convert(pdf)
                doc = conv_res.document

                # -------- Text (+ optional Markdown)
                text_out = ""
                if hasattr(doc, "export_to_text"):
                    try:
                        text_out = doc.export_to_text().strip()
                    except Exception:
                        text_out = ""

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

                # -------- Figures (nur wenn angefordert)
                figure_entries: List[Dict[str, Any]] = []
                if args.export_figures and PictureItem is not None and hasattr(doc, "iterate_items"):
                    figs_root = Path(args.figures_dir) / pdf.stem
                    saved_any_picture = False
                    pic_idx = 0

                    for element, _level in doc.iterate_items():
                        if isinstance(element, PictureItem):
                            try:
                                im = element.get_image(doc)  # PIL.Image
                                pic_idx += 1
                                figs_root.mkdir(parents=True, exist_ok=True)
                                saved_any_picture = True
                                out_file = figs_root / f"{pdf.stem}-picture-{pic_idx}.png"
                                im.save(out_file, "PNG")

                                entry = {
                                    "path": str(out_file).replace("\\", "/"),
                                    "page_index": safe_attr(getattr(element, "page_ref", None), "page_idx") if getattr(element, "page_ref", None) else None,
                                    # WICHTIG: caption_text kann Methode sein -> safe_attr
                                    "caption": safe_attr(element, "caption_text"),
                                }
                                figure_entries.append(entry)
                            except Exception as e:
                                log(f"WARN: Figure-Save fehlgeschlagen ({pdf.name}): {e}")

                    if figure_entries:
                        rec["figures"] = figure_entries
                    if not saved_any_picture and (figs_root.exists() and not any(figs_root.iterdir())):
                        try:
                            figs_root.rmdir()
                        except Exception:
                            pass

                # -------- Tables (nur wenn angefordert)
                table_entries: List[Dict[str, Any]] = []
                if args.export_tables and TableItem is not None and hasattr(doc, "iterate_items"):
                    tabs_root = Path(args.tables_dir) / pdf.stem
                    saved_any_table = False
                    tbl_idx = 0

                    for element, _level in doc.iterate_items():
                        if isinstance(element, TableItem):
                            try:
                                df = element.export_to_dataframe(doc=doc)  # doc-Arg wichtig
                                if df is None or df.empty:
                                    continue  # leere Tabellen NICHT speichern

                                tabs_root.mkdir(parents=True, exist_ok=True)
                                saved_any_table = True
                                tbl_idx += 1

                                csv_path = tabs_root / f"{pdf.stem}-table-{tbl_idx}.csv"
                                html_path = tabs_root / f"{pdf.stem}-table-{tbl_idx}.html"

                                df.to_csv(csv_path, index=False)

                                html_str = element.export_to_html(doc=doc)
                                with html_path.open("w", encoding="utf-8") as fp:
                                    fp.write(html_str)

                                entry = {
                                    "csv": str(csv_path).replace("\\", "/"),
                                    "html": str(html_path).replace("\\", "/"),
                                    "page_index": safe_attr(getattr(element, "page_ref", None), "page_idx") if getattr(element, "page_ref", None) else None,
                                }
                                table_entries.append(entry)
                            except Exception as e:
                                log(f"WARN: Table-Export fehlgeschlagen ({pdf.name}): {e}")

                    if table_entries:
                        rec["tables"] = table_entries
                    if not saved_any_table and (tabs_root.exists() and not any(tabs_root.iterdir())):
                        try:
                            tabs_root.rmdir()
                        except Exception:
                            pass

                # Confluence-Metadaten ergänzen
                rec.update(meta_by_local.get(npath, {}))

                # Schreiben – absolut JSON-sicher
                fout.write(json.dumps(rec, ensure_ascii=False, default=_json_default) + "\n")
                fout.flush()
                done += 1

            except Exception as e:
                log(f"WARN: Verarbeitung fehlgeschlagen für {npath}: {e}")

    log(f"Fertig. PDFs: {len(pdfs)} | erfolgreich verarbeitet: {done} | Output: {out_path}")

if __name__ == "__main__":
    main()
