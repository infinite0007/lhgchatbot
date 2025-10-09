#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
attachments_drawio_extract.py – robust(er)
- Unterstützt .drawio/.xml mit <mxfile><diagram …>, sowohl compressed als auch uncompressed.
- Bei uncompressed: liest Kindelement(e) von <diagram> (mxGraphModel) via ET.tostring().
- Bei compressed: versucht Base64 und urlsafe_b64; zlib mit/ohne Raw (-15).

Beispiel:

# Basis
python .\attachments_drawio_extract.py --attachments-dir data/raw/attachments --canonical-json data/raw/confluence.jsonl --out data/derivatives/drawio_text.jsonl

python .\attachments_drawio_extract.py `
  --attachments-dir data/raw/attachments `
  --canonical-json data/raw/confluence.jsonl `
  --out data/derivatives/drawio_text.jsonl

  
# Mit Cache (überspringt bereits verarbeitete Dateien)
python .\attachments_drawio_extract.py `
  --attachments-dir data/raw/attachments `
  --canonical-json data/raw/confluence.jsonl `
  --out data/derivatives/drawio_text.jsonl `
  --skip-existing
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
            page_url = row.get("url")
            for a in (row.get("attachments") or []):
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

def _looks_like_drawio(path: str) -> bool:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            head = f.read(4096)
        return "<mxfile" in head or "<diagram" in head
    except Exception:
        return False

def find_drawio(attachments_dir: str) -> List[str]:
    out: List[str] = []
    for root, _, files in os.walk(attachments_dir):
        for fn in files:
            ext = os.path.splitext(fn)[1].lower()
            fp = os.path.join(root, fn)
            if ext in {".drawio", ".xml"}:
                out.append(fp)
            elif ext == ".bin" and _looks_like_drawio(fp):
                out.append(fp)
    return out

def strip_html_to_text(html: str) -> str:
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
    for cell in root.iter():
        # Namespaces ignoren: nur auf Suffix prüfen
        if cell.tag.endswith("mxCell"):
            val = cell.attrib.get("value")
            if val:
                t = strip_html_to_text(val)
                if t:
                    texts.append(t)
    return "\n".join(texts).strip()

def _decompress_base64_both(b64txt: str) -> Optional[str]:
    """Versucht normal & urlsafe Base64; zlib raw & normal."""
    if not b64txt:
        return None
    candidates = []
    try:
        candidates.append(base64.b64decode(b64txt))
    except Exception:
        pass
    try:
        candidates.append(base64.urlsafe_b64decode(b64txt + "=="))
    except Exception:
        pass
    for raw in candidates:
        for wbits in (-15, zlib.MAX_WBITS):
            try:
                return zlib.decompress(raw, wbits).decode("utf-8", errors="ignore")
            except Exception:
                continue
    return None

def parse_drawio(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        data = f.read()

    try:
        root = ET.fromstring(data)
    except Exception:
        return ""  # keine gültige drawio/xml

    all_texts: List[str] = []

    # Finde alle <diagram>-Knoten (ohne Namespaces)
    diagrams = [el for el in root.iter() if el.tag.endswith("diagram")]
    if not diagrams:
        # Manche .xml haben direkt ein mxGraphModel ohne <mxfile>/<diagram>
        # => versuche direkt zu extrahieren
        txt = extract_text_from_mxgraph(data)
        return txt

    for diag in diagrams:
        compressed = (diag.attrib.get("compressed", "false").lower() == "true")

        if compressed:
            txtnode = diag.text or ""
            xml_inner = _decompress_base64_both(txtnode) or ""
        else:
            # Uncompressed: Inhalt liegt als Kindelement (z. B. <mxGraphModel/>) vor
            if len(diag):
                # kombiniere alle direkten Kinder
                parts = [ET.tostring(child, encoding="unicode") for child in list(diag)]
                xml_inner = "\n".join(parts)
            else:
                # manchmal liegt das XML (HTML-escaped) direkt im Text
                xml_inner = diag.text or ""

        if not xml_inner:
            continue

        t = extract_text_from_mxgraph(xml_inner)
        if t:
            all_texts.append(t)

    return "\n".join(all_texts).strip()

def main():
    ap = argparse.ArgumentParser(description="Draw.io (.drawio/.xml) Text-Extraktion.")
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
    done, empty = 0, 0

    with open(args.out, "a", encoding="utf-8") as fout:
        for fp in files:
            npath = os.path.normpath(fp)
            if args.skip_existing and npath in existing:
                continue
            try:
                text = parse_drawio(npath)
                if not text:
                    empty += 1
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

    log(f"Fertig. Draw.io-Dateien gefunden: {len(files)} | mit Text extrahiert: {done} | leer/ohne Text: {empty} | Output: {args.out}")

if __name__ == "__main__":
    main()
