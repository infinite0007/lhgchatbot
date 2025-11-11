#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
join_all_pages.py
-----------------
Example:
python .\join_all_pages.py --pages .\data\raw\confluence.jsonl --ocr-images .\data\derivatives\ocr.jsonl --drawio .\data\derivatives\drawio_text.jsonl --pdf-docling .\data\derivatives\pdf_docling_prepared.jsonl --pdf-fig-ocr .\data\derivatives\prepared_figures_ocr.jsonl --out .\data\derivatives\joined_pages_full.jsonl

Mergt alle Attachment-Signale (Image-OCR, Drawio, PDF-Text, PDF-Figure-OCR, PDF-Tabellen)
zur Seite zurück. Schreibt pro Seite genau *einen* Record und erzeugt:
  - attachments_index: strukturierter Index der Anhänge/Signale
  - attachments_text:   zusammengeführter Klartext (OCR/Drawio/PDF/Table-Preview)
  - text_plain_plus_attachments: Basistext + attachments_text
  - Sanftes Entgraten von Confluence-UI/Boilerplate (scrub_base_noise)
  - Optionale Abschnitts-Labels für die KI (--label-blocks)
  - Tabellen-Preview als Markdown (--tables-as-text mit Limits)
"""

from __future__ import annotations
import os, json, argparse, csv, re, time
from collections import defaultdict
from typing import Dict, Any, Iterable, List, Optional

# --------------------- UI-/Boilerplate-Entgratung ---------------------

UI_NOISE_PATTERNS = [
    r"Zum Anfang der Metadaten",
    r"Invalid label:\s*\"?<Define your search lable>\"?",
    r"Feature\s*/\s*Requirement change Request section",
    r"Affected Objects\s*<Define your search lable>",
    r"\bGetting issue details\.\.\.\b",               # JIRA Platzhalter
    r"\bSTATUS\b",                                     # JIRA Platzhalter
    r"\btrue\s+true\s+false\s+\S+\s+false\s+\d+\s+\d+\b",  # Drawio-Makrozeilen im text_plain
]

def scrub_base_noise(s: str) -> str:
    """Entfernt typische Confluence-UI-Fetzen & glättet Whitespace.
       Lässt fachlichen Inhalt unangetastet."""
    if not isinstance(s, str):
        return ""
    # Confluence-/Makro-Codefences „auspacken“ (Inhalt behalten)
    s = re.sub(r"```+\s*([^`]+?)\s*```+", r"\1", s, flags=re.DOTALL)
    # UI-Fetzen entfernen
    for rx in UI_NOISE_PATTERNS:
        s = re.sub(rx, " ", s, flags=re.IGNORECASE)
    # Whitespace glätten
    s = re.sub(r"\s+", " ", s).strip()
    return s

# --------------------- Helpers ---------------------

def load_rows(path: Optional[str]) -> Iterable[Dict[str, Any]]:
    if not path or not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue

def normpath(p: Optional[str]) -> Optional[str]:
    return os.path.normpath(p) if isinstance(p, str) else None

def safe_append_text(parts: List[str], s: Optional[str]) -> None:
    if isinstance(s, str):
        st = s.strip()
        if st:
            parts.append(st)

def trim_cell(s: str, max_chars: int) -> str:
    return s if len(s) <= max_chars else (s[: max(0, max_chars - 1)] + "…")

def csv_to_markdown_preview(csv_path: str,
                            max_rows: int = 25,
                            max_cols: int = 10,
                            max_cell_chars: int = 80) -> Optional[str]:
    """Liest CSV und rendert eine kompakte Markdown-Vorschau mit harten Limits."""
    if not (isinstance(csv_path, str) and os.path.exists(csv_path)):
        return None
    try:
        with open(csv_path, "r", encoding="utf-8", newline="") as fp:
            reader = csv.reader(fp)
            rows: List[List[str]] = []
            for i, row in enumerate(reader):
                if i == 0:
                    header = [trim_cell((c or ""), max_cell_chars) for c in row[:max_cols]]
                    rows.append(header)
                else:
                    if i > max_rows:
                        break
                    cut = [trim_cell((c or ""), max_cell_chars) for c in row[:max_cols]]
                    rows.append(cut)
        if not rows:
            return None
        header, body = rows[0], rows[1:]
        md: List[str] = []
        if header:
            md.append("| " + " | ".join(header) + " |")
            md.append("| " + " | ".join(["---"] * len(header)) + " |")
        for r in body:
            md.append("| " + " | ".join(r) + " |")
        if not md:
            return None
        if len(body) >= max_rows:
            md.append(f"\n*… gekürzt auf {max_rows} Zeilen / {max_cols} Spalten …*")
        return "\n".join(md)
    except Exception:
        return None

# --------------------- Main ---------------------

def main():
    start_time = time.time()
    ap = argparse.ArgumentParser(description="Join Seiten + Attachments (OCR/Drawio/PDF/Figures/Tables).")
    ap.add_argument("--pages", required=True, help="confluence.jsonl (Seiten)")
    ap.add_argument("--ocr-images", default=None, help="ocr.jsonl (Image-OCR)")
    ap.add_argument("--drawio", default=None, help="drawio_text.jsonl")
    ap.add_argument("--pdf-docling", default=None, help="pdf_docling_prepared.jsonl")
    ap.add_argument("--pdf-fig-ocr", default=None, help="prepared_figures_ocr.jsonl")
    ap.add_argument("--out", required=True)

    # Optionen für Tabellen → Text
    ap.add_argument("--tables-as-text", action="store_true",
                    help="Docling-Tabellen (CSV) als Markdown-Vorschau in attachments_text aufnehmen.")
    ap.add_argument("--tables-max-rows", type=int, default=25)
    ap.add_argument("--tables-max-cols", type=int, default=10)
    ap.add_argument("--tables-max-cell-chars", type=int, default=80)

    # Optional: OCR-Bounding-Boxes zusätzlich mitschreiben
    ap.add_argument("--include-ocr-boxes", action="store_true",
                    help="OCR-Boxen (figures/image) zusätzlich in attachments_index speichern.")

    # Optional: Label-Header vor Attachment-Textblöcke setzen
    ap.add_argument("--label-blocks", action="store_true",
                    help="Fügt Abschnitts-Labels in attachments_text ein (hilft dem LLM beim Verstehen).")

    args = ap.parse_args()
    out_dir = os.path.dirname(args.out) or "."
    os.makedirs(out_dir, exist_ok=True)

    # ---------- Buckets je Seite ----------
    texts_by_page: Dict[str, List[str]] = defaultdict(list)     # für attachments_text
    index_by_page: Dict[str, List[Dict[str, Any]]] = defaultdict(list)  # attachments_index

    def maybe_label(pid: str, label: str):
        """Optionaler Abschnitts-Header, damit die KI Quelle/Typ erkennt."""
        if args.label_blocks and isinstance(label, str) and label.strip():
            safe_append_text(texts_by_page[pid], f"\n### {label}\n")

    # ---------- Image-OCR ----------
    for r in load_rows(args.ocr_images):
        pid = r.get("page_id")
        if not pid:
            continue
        entry = {
            "type": "image_ocr",
            "local_path": normpath(r.get("local_path")),
            "attachment_id": r.get("attachment_id"),
            "title": r.get("title"),
            "mediaType": r.get("mediaType"),
            "fileSize": r.get("fileSize"),
            "page_url": r.get("page_url"),
        }
        txt = r.get("ocr_text") or ""
        if args.include_ocr_boxes and "ocr_boxes" in r:
            entry["ocr_boxes"] = r.get("ocr_boxes")  # evtl. groß!
        index_by_page[pid].append(entry)
        if isinstance(txt, str) and txt.strip():
            maybe_label(pid, f"Image OCR ({os.path.basename(entry['local_path'] or '')})")
            safe_append_text(texts_by_page[pid], txt)

    # ---------- Drawio ----------
    for r in load_rows(args.drawio):
        pid = r.get("page_id")
        if not pid:
            continue
        entry = {
            "type": "drawio",
            "local_path": normpath(r.get("local_path")),
            "attachment_id": r.get("attachment_id"),
            "title": r.get("title"),
            "mediaType": r.get("mediaType"),
            "fileSize": r.get("fileSize"),
            "page_url": r.get("page_url"),
        }
        index_by_page[pid].append(entry)
        dtext = r.get("drawio_text")
        if isinstance(dtext, str) and dtext.strip():
            maybe_label(pid, f"Diagram (draw.io) {entry['title'] or ''}")
            # Drawio-„Graph“ bleibt als Klartext (z. B. Bedingungen/Edges)
            safe_append_text(texts_by_page[pid], dtext)

    # ---------- PDF (Docling-Text + Tables/ Figures-Referenzen) ----------
    #  A) PDF-Text + Tables-Index (+optional Tables als Text)
    for r in load_rows(args.pdf_docling):
        pid = r.get("page_id")
        if not pid:
            continue

        pdf_path = normpath(r.get("local_path"))
        pdf_title = r.get("title")
        pdf_text = r.get("text") or ""

        if isinstance(pdf_text, str) and pdf_text.strip():
            index_by_page[pid].append({
                "type": "pdf_text",
                "pdf_local_path": pdf_path,
                "attachment_id": r.get("attachment_id"),
                "title": pdf_title,
                "page_url": r.get("page_url"),
            })
            maybe_label(pid, f"PDF Text ({pdf_title or os.path.basename(pdf_path or '')})")
            safe_append_text(texts_by_page[pid], pdf_text)

        # Tabellen
        for i, t in enumerate(r.get("tables") or [], start=1):
            csv_path = t.get("csv")
            html_path = t.get("html")
            page_idx = t.get("page_index")
            table_entry = {
                "type": "pdf_table",
                "pdf_local_path": pdf_path,
                "attachment_id": r.get("attachment_id"),
                "title": pdf_title,
                "page_index": page_idx,
                "csv": csv_path,
                "html": html_path,
            }
            if args.tables_as_text and csv_path:
                md = csv_to_markdown_preview(
                    csv_path,
                    max_rows=args.tables_max_rows,
                    max_cols=args.tables_max_cols,
                    max_cell_chars=args.tables_max_cell_chars,
                )
                if md:
                    maybe_label(pid, f"Table from PDF '{pdf_title}' (page_index={page_idx}, file={os.path.basename(csv_path)})")
                    safe_append_text(texts_by_page[pid], md)
                    table_entry["text_preview"] = md
            index_by_page[pid].append(table_entry)

        # (Hinweis: Figures-Referenzen sind in r['figures'], das OCR dazu kommt separat unten)

    #  B) Figures-OCR (aus Fig-Detektor/Extractor) – Text + optional Boxen
    for r in load_rows(args.pdf_fig_ocr):
        pid = r.get("page_id")
        if not pid:
            continue
        entry = {
            "type": "pdf_figure_ocr",
            "pdf_local_path": normpath(r.get("pdf_local_path")),
            "figure_path": normpath(r.get("figure_path")),
            "attachment_id": r.get("attachment_id"),
            "title": r.get("title"),
            "page_index": r.get("page_index"),
            "caption": r.get("caption"),
            "page_url": r.get("page_url"),
        }
        if args.include_ocr_boxes and "ocr_items" in r:
            entry["ocr_items"] = r.get("ocr_items")
        index_by_page[pid].append(entry)
        ftxt = r.get("ocr_text")
        if isinstance(ftxt, str) and ftxt.strip():
            maybe_label(pid, f"Figure OCR ({os.path.basename(entry['figure_path'] or '')})")
            if entry.get("caption"):
                safe_append_text(texts_by_page[pid], f"Caption: {entry['caption']}")
            safe_append_text(texts_by_page[pid], ftxt)

    # ---------- Schreiben ----------
    with open(args.out, "w", encoding="utf-8") as fout:
        for p in load_rows(args.pages):
            pid = p.get("id")
            # Basistext sanft entgraten (UI-Reste weg, Inhalt behalten)
            base = scrub_base_noise((p.get("text_plain") or "").strip())
            extras_text = "\n".join(texts_by_page.get(pid, []))

            p["attachments_index"] = index_by_page.get(pid, [])
            p["attachments_text"] = extras_text

            if base and extras_text:
                p["text_plain_plus_attachments"] = base + "\n\n" + extras_text
            elif extras_text:
                p["text_plain_plus_attachments"] = extras_text
            else:
                p["text_plain_plus_attachments"] = base

            fout.write(json.dumps(p, ensure_ascii=False) + "\n")

    end_time = time.time()
    elapsed = end_time - start_time
    print(f"Gesamtdauer: {elapsed:.2f} Sekunden ({elapsed/60:.2f} Minuten)")
    print(f"[join] Fertig. Output: {args.out}")

if __name__ == "__main__":
    main()
