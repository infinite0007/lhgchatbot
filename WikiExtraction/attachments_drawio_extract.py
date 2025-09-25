#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
attachments_drawio_extract.py
-----------------------------
Extrahiert TEXT-Inhalte aus .drawio (mxfile / mxGraph) Attachments.
- Unterstützt unkomprimierte XML und compressed="true" Diagramme (base64+zlib).
- Holt value-Attribute der mxCell-Knoten (HTML wird zu Plaintext gestrippt).
- Schreibt JSONL (1 Zeile pro .drawio) mit zusammengeführtem Text.

Beispiel:
  python attachments_drawio_extract.py \
    --attachments-dir data/raw/attachments \
    --canonical-json data/raw/confluence.jsonl \
    --out data/derivatives/drawio_text.jsonl
"""

from __future__ import annotations
import os, sys, json, time, argparse, base64, zlib, xml.etree.ElementTree as ET
from typing import Dict, Any, List, Optional
from html import unescape

try:
    from bs4 import BeautifulSoup
except Exception:
    print("Bitte installieren: pip install beautifulsoup4")
    raise

def log(msg: str) -> None:
    print(f"[drawio] {msg}")

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

def find_drawio(attachments_dir: str) -> List[str]:
    out = []
    for root, _, files in os.walk(attachments_dir):
        for fn in files:
            if os.path.splitext(fn)[1].lower() in {".drawio", ".xml"}:
                out.append(os.path.join(root, fn))
    return out

def strip_html_to_text(html: str) -> str:
    # Draw.io speichert meist HTML in value-Attributen
    if not html:
        return ""
    soup = BeautifulSoup(unescape(html), "html.parser")
    txt = soup.get_text(" ", strip=True)
    return " ".join(txt.split())

def extract_text_from_mxgraph(xml_text: str) -> str:
    try:
        root = ET.fromstring(xml_text)
    except Exception:
        return ""
    texts: List[str] = []
    # Werte sind i.d.R. in mxCell@value
    for cell in root.iter():
        if cell.tag.endswith("mxCell"):
            val = cell.attrib.get("value")
            if val:
                t = strip_html_to_text(val)
                if t:
                    texts.append(t)
    return "\n".join(texts).strip()

def parse_drawio(path: str) -> str:
    """
    .drawio ist ein <mxfile> mit 1..n <diagram>-Kindern.
    - uncompressed: <diagram> enthält XML als Klartext (mxGraphModel)
    - compressed="true": Base64+zlib komprimiertes XML
    Wir extrahieren Text aus allen Diagrammen und mergen.
    """
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        data = f.read()

    try:
        root = ET.fromstring(data)
    except Exception:
        return ""  # Not a valid drawio XML; silently skip

    all_texts: List[str] = []
    for diag in root.findall(".//diagram"):
        txt = diag.text or ""
        compressed = (diag.attrib.get("compressed", "false").lower() == "true")
        xml_inner = ""
        if compressed:
            try:
                raw = base64.b64decode(txt)
                xml_inner = zlib.decompress(raw, -15).decode("utf-8", errors="ignore")
            except Exception:
                continue
        else:
            # uncompressed: Inhalt kann direkt das mxGraphModel XML sein
            xml_inner = txt

        if xml_inner:
            t = extract_text_from_mxgraph(xml_inner)
            if t:
                all_texts.append(t)

    return "\n".join(all_texts).strip()

def main():
    ap = argparse.ArgumentParser(description="Draw.io (.drawio) Text-Extraktion.")
    ap.add_argument("--attachments-dir", default="data/raw/attachments")
    ap.add_argument("--canonical-json", default=None)
    ap.add_argument("--out", default="data/derivatives/drawio_text.jsonl")
    ap.add_argument("--skip-existing", action="store_true")
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

    files = find_drawio(args.attachments_dir)
    done = 0
    with open(args.out, "a", encoding="utf-8") as fout:
        for fp in files:
            npath = os.path.normpath(fp)
            if args.skip_existing and npath in existing:
                continue
            try:
                text = parse_drawio(npath)
                if not text:
                    continue
                rec = {
                    "local_path": npath,
                    "drawio_text": text,
                    "extracted_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                }
                rec.update(meta_by_local.get(npath, {}))
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                done += 1
            except Exception as e:
                log(f"WARN: Draw.io-Parsing fehlgeschlagen für {npath}: {e}")

    log(f"Fertig. Draw.io-Dateien gefunden: {len(files)} | mit Text extrahiert: {done} | Output: {args.out}")

if __name__ == "__main__":
    main()
