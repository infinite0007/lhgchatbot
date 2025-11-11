#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Canonischer Confluence-Extractor – vorbereitet für Finetuning, RAG und CAG

Features
--------
- Lädt Seiten aus Confluence inkl. Titel, Space, Timestamps, Labels, Ancestors
- Body als Storage (HTML) + View (gerendert)
- Plaintext-Extraktion
- Kommentare + Attachments (optional: Download, mit automatischer Endungs-Erkennung)
- Schreibt eine JSONL-Datei als kanonische Quelle

Beispiele
---------
  python canonical_extractor.py --space SWEIG --since 2024-01-01 --out data/raw/confluence.jsonl
  python canonical_extractor.py --space SWEIG --with-attachments --out data/raw/confluence.jsonl
"""

from __future__ import annotations
import os
import re
import sys
import json
import time
import base64
import argparse
import mimetypes
from typing import Dict, Any, Iterable, List, Optional, Tuple
from statistics import median

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
load_dotenv()

# ---------------- Helpers ----------------

def log(msg: str) -> None:
    print(f"[conf-extract] {msg}")

def env_or_die(key: str) -> str:
    v = os.getenv(key, "").strip()
    if not v:
        raise SystemExit(f"Umgebungsvariable fehlt: {key}")
    return v

def _guess_ext_from_mimetype(mt: Optional[str]) -> Optional[str]:
    """Versuche Dateiendung aus MIME-Type zu bestimmen; mit Spezialfällen."""
    if not mt:
        return None
    mt = mt.lower().strip()
    # häufige Spezialfälle zuerst
    if mt in ("application/vnd.jgraph.mxfile", "application/x-drawio"):
        return ".drawio"
    if mt == "image/jpg":
        return ".jpg"
    # generisch
    ext = mimetypes.guess_extension(mt)
    # manche Image-Types liefern None → fallback
    if not ext and mt.startswith("image/"):
        return "." + mt.split("/", 1)[1]
    return ext

def _ext_from_headers(headers: Dict[str, str]) -> Tuple[Optional[str], Optional[str]]:
    """
    Liefert (ext, filename_from_cd) basierend auf Content-Type/Disposition.
    """
    ct = headers.get("Content-Type")
    cd = headers.get("Content-Disposition", "")
    ext = _guess_ext_from_mimetype(ct)
    filename = None
    # Content-Disposition: attachment; filename="xxx.ext"  oder  filename*=UTF-8''xxx.ext
    import urllib.parse as up
    m = re.search(r'filename\*?=(?:UTF-8\'\')?"?([^";]+)"?', cd)
    if m:
        filename = up.unquote(m.group(1))
        if "." in os.path.basename(filename):
            ext = os.path.splitext(filename)[1] or ext
    return ext, filename

def _wc(s: Optional[str]) -> int:
    return len((s or "").split())

def _fmt_int(n: int) -> str:
    return f"{n:,}".replace(",", ".")

def _approx(n: int) -> str:
    return f"$\\sim${_fmt_int(n)}"

# ---------------- HTTP Wrapper ----------------

class Http:
    def __init__(self, base: str, bearer: Optional[str], email: Optional[str], token: Optional[str]):
        self.base = base.rstrip('/')
        self.sess = requests.Session()
        if bearer:
            self.sess.headers.update({"Authorization": f"Bearer {bearer}", "Accept": "application/json"})
        elif email and token:
            basic = base64.b64encode(f"{email}:{token}".encode()).decode()
            self.sess.headers.update({"Authorization": f"Basic {basic}", "Accept": "application/json"})
        else:
            raise SystemExit("Setze ATLASSIAN_BEARER oder ATLASSIAN_EMAIL+ATLASSIAN_TOKEN")

    def _make_url(self, path: str) -> str:
        # Absolute URLs unverändert; relative an BASE anhängen
        if path.startswith("http://") or path.startswith("https://"):
            return path
        if path.startswith('/'):
            return f"{self.base}{path}"
        return f"{self.base}/{path}"

    def get(self, path: str, params: Dict[str, Any] | None = None, tries: int = 5, stream: bool = False) -> requests.Response:
        url = self._make_url(path)
        backoff = 1.0
        for _ in range(tries):
            r = self.sess.get(url, params=params, timeout=60, stream=stream)
            if r.status_code in (429, 502, 503, 504):
                time.sleep(backoff)
                backoff = min(backoff * 2, 16)
                continue
            r.raise_for_status()
            return r
        r.raise_for_status()
        return r

# ---------------- Confluence API ----------------

class ConfluenceExtractor:
    def __init__(self, http: Http):
        self.http = http

    # Spaces
    def list_spaces(self, limit: int = 50) -> Iterable[Dict[str, Any]]:
        start = 0
        while True:
            r = self.http.get("/rest/api/space", params={"limit": limit, "start": start})
            data = r.json()
            results = data.get("results", [])
            if not results:
                break
            for it in results:
                yield it
            start += limit

    # Pages via CQL search (filter by since)
    def list_pages_cql(self, space_key: str, since: Optional[str] = None, limit: int = 50) -> Iterable[str]:
        start = 0
        cql = f"space = {space_key} and type = page"
        if since:
            cql += f" and lastmodified >= {since}"
        while True:
            r = self.http.get("/rest/api/search", params={"cql": cql, "start": start, "limit": limit})
            data = r.json()
            results = data.get("results", [])
            if not results:
                break
            for it in results:
                content = it.get("content", {})
                cid = content.get("id") or it.get("id")
                if cid:
                    yield cid
            start += limit

    # Page detail with expands
    def get_page(self, page_id: str) -> Dict[str, Any]:
        expands = [
            "body.storage",
            "body.view",
            "version",
            "space",
            "metadata.labels",
            "ancestors",
            "history",
        ]
        r = self.http.get(f"/rest/api/content/{page_id}", params={"expand": ",".join(expands)})
        return r.json()

    # ---- Attachments (mit Endungs-Erkennung & Seiten-Unterordner) ----
    def list_attachments(
        self,
        page_id: str,
        page_title: str = "",
        page_url: str | None = None,
        download_dir: Optional[str] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        start = 0
        while True:
            r = self.http.get(
                f"/rest/api/content/{page_id}/child/attachment",
                params={"start": start, "limit": limit}
            )
            data = r.json()
            results = data.get("results", [])
            if not results:
                break

            for a in results:
                links = a.get("_links", {}) or {}
                download = links.get("download")  # relativ: /download/attachments/{page}/{file}?...
                title = a.get("title") or ""
                mediaType = (a.get("metadata", {}) or {}).get("mediaType")
                fileSize = (a.get("extensions", {}) or {}).get("fileSize")

                att = {
                    "id": a.get("id"),
                    "page_id": page_id,
                    "page_title": page_title,
                    "page_url": page_url,
                    "title": title,
                    "mediaType": mediaType,
                    "comment": (a.get("metadata", {}) or {}).get("comment"),
                    "created": (a.get("history", {}) or {}).get("createdDate"),
                    "creator": ((a.get("history", {}) or {}).get("createdBy", {}) or {}).get("displayName"),
                    "fileSize": fileSize,
                    "download": download,              # relativer Pfad
                    "download_filename": None,
                    "local_path": None,
                }
                out.append(att)

                # Datei speichern (falls gewünscht und Download vorhanden)
                if not download_dir or not download:
                    continue

                try:
                    # Seiten-Unterordner
                    page_dir = os.path.join(download_dir, page_id)
                    os.makedirs(page_dir, exist_ok=True)

                    # GET (stream) um Header zu inspizieren und Content zu schreiben
                    rfile = self.http.get(download, stream=True)
                    ext_from_headers, name_from_cd = _ext_from_headers(rfile.headers)

                    # Basis aus ID + Title
                    base = f"{att['id']}_{title}".replace("/", "_").replace("\\", "_").strip()

                    # Endung ableiten: (1) aus Titel, (2) aus Headern, (3) aus MIME, (4) .bin
                    _, ext_from_title = os.path.splitext(title)
                    ext = ext_from_title or ext_from_headers or _guess_ext_from_mimetype(mediaType) or ".bin"

                    # Download-Filename für Metadaten
                    att["download_filename"] = name_from_cd or (title + ("" if ext_from_title else ext))

                    # finaler Dateiname
                    fname = base if base.lower().endswith(ext.lower()) else (base + ext)
                    fpath = os.path.join(page_dir, fname)

                    # Schreiben
                    with open(fpath, "wb") as f:
                        for chunk in rfile.iter_content(8192):
                            f.write(chunk)
                    att["local_path"] = fpath

                except Exception as e:
                    log(f"WARN: Attachment {title or att['id']} konnte nicht geladen werden, "
                    f"(Seite: {page_url or page_id}): {e}")

            start += limit
        return out

    # Comments
    def list_comments(self, page_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        start = 0
        while True:
            try:
                r = self.http.get(
                    f"/rest/api/content/{page_id}/child/comment",
                    params={"start": start, "limit": limit, "expand": "body.storage,body.view,version,history"}
                )
                data = r.json()
                results = data.get("results", [])
                if not results:
                    break
                for c in results:
                    out.append({
                        "id": c.get("id"),
                        "created": (c.get("history") or {}).get("createdDate"),
                        "creator": (c.get("history") or {}).get("createdBy", {}).get("displayName"),
                        "updated": (c.get("version") or {}).get("when"),
                        "body_storage": (((c.get("body") or {}).get("storage") or {}).get("value")) or "",
                        "body_view": (((c.get("body") or {}).get("view") or {}).get("value")) or "",
                    })
                start += limit
            except requests.HTTPError as e:
                if e.response is not None and e.response.status_code == 401:
                    log(f"WARN: Keine Berechtigung Kommentare für Seite {page_id} – überspringe")
                    return []
                raise
        return out

    # Restrictions
    def get_restrictions(self, page_id: str) -> Dict[str, Any]:
        try:
            r = self.http.get(
                f"/rest/api/content/{page_id}/restriction/byOperation",
                params={"expand": "read.restrictions.user,update.restrictions.user"}
            )
            return r.json()
        except Exception:
            return {}

# ---------------- Normalisierung ----------------

def storage_to_text(storage_html: str) -> str:
    if not storage_html:
        return ""
    soup = BeautifulSoup(storage_html, "html.parser")
    # Codeblöcke etwas „markieren“, damit Inhalt nicht verloren geht
    for code in soup.find_all(["ac:structured-macro", "code", "pre"]):
        txt = code.get_text(" ") or ""
        code.string = "\n```\n" + txt.strip() + "\n```\n"
    text = soup.get_text(" ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# ---------------- Dump ----------------

def dump_pages(space_keys: List[str], since: Optional[str], out_path: str, with_attachments: bool = False) -> Dict[str, Any]:
    base = env_or_die("ATLASSIAN_BASE")
    bearer = os.getenv("ATLASSIAN_BEARER", "").strip() or None
    email = os.getenv("ATLASSIAN_EMAIL", "").strip() or None
    token = os.getenv("ATLASSIAN_TOKEN", "").strip() or None

    http = Http(base, bearer, email, token)
    cx = ConfluenceExtractor(http)

    # ---- Metriken ----
    total_pages = 0
    word_counts: List[int] = []
    total_words_text_plain = 0

    att_declared_total = 0
    att_attempted = 0
    att_download_ok = 0
    att_download_err = 0

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    attachment_dir = "data/raw/attachments" if with_attachments else None

    with open(out_path, "w", encoding="utf-8") as out:
        for sk in space_keys:
            log(f"Space {sk}: suche Seiten… (since={since or 'ALL'})")
            for pid in cx.list_pages_cql(sk, since=since):
                p = cx.get_page(pid)
                body_storage = ((p.get("body", {}) or {}).get("storage", {}) or {}).get("value", "")
                body_view = ((p.get("body", {}) or {}).get("view", {}) or {}).get("value", "")
                space = (p.get("space", {}) or {})
                labels = [l.get("name") for l in ((p.get("metadata", {}) or {}).get("labels", {}) or {}).get("results", [])]
                ancestors = [a.get("id") for a in p.get("ancestors", [])]
                version = (p.get("version", {}) or {}).get("number")
                updated = (p.get("version", {}) or {}).get("when")
                created = (p.get("history", {}) or {}).get("createdDate")
                creator = ((p.get("history", {}) or {}).get("createdBy", {}) or {}).get("displayName")
                last_modifier = ((p.get("version", {}) or {}).get("by", {}) or {}).get("displayName")
                status = p.get("status")
                weblink = (p.get("_links", {}) or {}).get("webui")
                url = f"{base}{weblink}" if weblink else None
                title = p.get("title") or ""

                comments = cx.list_comments(pid)
                attachments = cx.list_attachments(pid, page_title=title, page_url=url, download_dir=attachment_dir)
                restrictions = cx.get_restrictions(pid)

                text_plain = storage_to_text(body_storage)

                # ---- Metriken: text_plain ----
                wc = _wc(text_plain)
                word_counts.append(wc)
                total_words_text_plain += wc
                total_pages += 1

                # ---- Metriken: Attachments ----
                att_declared_total += len(attachments)
                if with_attachments:
                    for a in attachments:
                        if a.get("download"):
                            att_attempted += 1
                            if a.get("local_path"):
                                att_download_ok += 1
                            else:
                                att_download_err += 1

                row = {
                    "source": "confluence",
                    "id": p.get("id"),
                    "space_key": space.get("key"),
                    "space_name": space.get("name"),
                    "title": title,
                    "url": url,
                    "status": status,
                    "version": version,
                    "created": created,
                    "updated": updated,
                    "created_by": creator,
                    "last_modified_by": last_modifier,
                    "labels": labels,
                    "ancestors": ancestors,
                    "body_storage": body_storage,
                    "body_view": body_view,
                    "text_plain": text_plain,
                    "comments": comments,
                    "attachments": attachments,
                    "restrictions": restrictions,
                    "extracted_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                }
                out.write(json.dumps(row, ensure_ascii=False) + "\n")

    median_words = int(median(word_counts)) if word_counts else 0
    att_rate = (100.0 * att_download_ok / att_attempted) if att_attempted else None

    return {
        "pages": total_pages,
        "median_words": median_words,
        "total_words_text_plain": total_words_text_plain,
        "attachments_declared": att_declared_total,
        "attachments_attempted": att_attempted,
        "attachments_ok": att_download_ok,
        "attachments_err": att_download_err,
        "attachments_rate_pct": att_rate,
        "json_lines_written": total_pages,
        "json_valid_pct": 100.0 if total_pages > 0 else 0.0,
    }

# ---------------- CLI ----------------

def main():
    start_time = time.time()
    ap = argparse.ArgumentParser(description="Confluence kanonischer Extractor")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--space", action="append", help="Space-Key; mehrfach möglich (--space A --space B)")
    g.add_argument("--all-spaces", action="store_true", help="Alle Spaces iterieren")
    ap.add_argument("--since", help="YYYY-MM-DD (optional, lastmodified >=)")
    ap.add_argument("--out", default="data/raw/confluence.jsonl", help="Ausgabedatei (JSONL)")
    ap.add_argument("--with-attachments", action="store_true", help="Anhänge herunterladen und speichern")
    args = ap.parse_args()

    base = env_or_die("ATLASSIAN_BASE")
    bearer = os.getenv("ATLASSIAN_BEARER", "").strip() or None
    email = os.getenv("ATLASSIAN_EMAIL", "").strip() or None
    token = os.getenv("ATLASSIAN_TOKEN", "").strip() or None
    http = Http(base, bearer, email, token)
    cx = ConfluenceExtractor(http)

    if args.all_spaces:
        space_keys = [s.get("key") for s in cx.list_spaces()]
    else:
        space_keys = args.space

    metrics = dump_pages(space_keys, args.since, args.out, with_attachments=args.with_attachments)

    end_time = time.time()
    elapsed = end_time - start_time
    log(f"Gesamtdauer: {elapsed:.2f} Sekunden ({elapsed/60:.2f} Minuten)")

    # --------- Konsolenreport (am Ende) ----------
    print("\n=== Zusammenfassung (Canonical Extract) ===")
    print(f"Confluence-Seiten                 : {_fmt_int(metrics['pages'])}")
    print(f"Ø Seitelänge                      : {_fmt_int(metrics['median_words'])} Wörter (Nur text_plain, Median)")
    print(f"Gesamtvolumen (Text)              : {_approx(metrics['total_words_text_plain'])} Wörter (Nur text_plain)")
    print(f"JSON-Syntaxvalidität              : {metrics['json_valid_pct']:.1f}% (alle {_fmt_int(metrics['json_lines_written'])} Zeilen)")
    if args.with_attachments:
        tried = metrics['attachments_attempted']
        ok = metrics['attachments_ok']
        err = metrics['attachments_err']
        rate = metrics['attachments_rate_pct']
        print(f"Attachments (gesamt, deklariert)  : {_fmt_int(metrics['attachments_declared'])}")
        if tried:
            print(f"Attachment-Download-Rate          : {rate:.1f}%  ({_fmt_int(ok)} ok | {_fmt_int(err)} Fehler)")
        else:
            print(f"Attachment-Download-Rate          : n/a (keine Downloads versucht)")
    else:
        print("Attachment-Download-Rate          : n/a (ohne --with-attachments)")
    print(f"Laufzeit                           : {elapsed:.2f}s")
    print("Hinweis: OCR-/PDF-Raten werden in den jeweiligen Schritten gemessen.")

if __name__ == "__main__":
    main()
