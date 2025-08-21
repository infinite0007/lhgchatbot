#!/usr/bin/env python3
"""
Canonischer Confluence-Extractor (nur Erfassung) – vorbereitet für Finetuning, RAG und CAG

Was er tut
---------
- Lädt Seiten aus Confluence (Server/Cloud) inklusive: Titel, Space, Version, Timestamps, Labels,
  Ancestors (Breadcrumbs), Body im Storage-Format (HTML), gerenderter Plaintext,
  Kommentare (wenn möglich), Anhänge (mit Download-Links & MIME-Typen), optionale Restriktionen.
- Schreibt **eine JSONL-Datei** (1 Zeile = 1 Seite) als kanonische Quelle für weitere Pipelines.
- Keine Chunking/Embedding/QA – nur Erfassung & Normalisierung.

Installation
------------
  pip install requests beautifulsoup4 python-dotenv

Beispiele
---------
  python canonical_extractor.py --space SPACEKEY --since 2024-01-01 --out data/raw/confluence.jsonl
  python canonical_extractor.py --all-spaces --since 2024-01-01 --out data/raw/conf_all.jsonl
"""

from __future__ import annotations
import os, sys, json, time, base64, argparse, re
from typing import Dict, Any, Iterable, List, Optional

import requests
from bs4 import BeautifulSoup

from dotenv import load_dotenv
load_dotenv()

# ------------------------------ Helpers ------------------------------

def log(msg: str) -> None:
    print(f"[conf-extract] {msg}")

def env_or_die(key: str) -> str:
    v = os.getenv(key, "").strip()
    if not v:
        raise SystemExit(f"Umgebungsvariable fehlt: {key}")
    return v

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

    def get(self, path: str, params: Dict[str, Any] | None = None, tries: int = 5) -> requests.Response:
        url = f"{self.base}{path}" if path.startswith('/') else f"{self.base}/{path}"
        backoff = 1.0
        for _ in range(tries):
            r = self.sess.get(url, params=params, timeout=60)
            if r.status_code in (429, 502, 503, 504):
                time.sleep(backoff)
                backoff = min(backoff * 2, 16)
                continue
            r.raise_for_status()
            return r
        r.raise_for_status()
        return r

# --------------------------- Confluence API --------------------------

class ConfluenceExtractor:
    def __init__(self, http: Http):
        self.http = http

    # --- Spaces ---
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

    # --- Pages via CQL search (filter by since) ---
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

    # --- Page detail with expands ---
    def get_page(self, page_id: str) -> Dict[str, Any]:
        expands = [
            "body.storage",
            "version",
            "space",
            "metadata.labels",
            "ancestors",
            "history",
        ]
        r = self.http.get(f"/rest/api/content/{page_id}", params={"expand": ",".join(expands)})
        return r.json()

    def list_attachments(self, page_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        start = 0
        while True:
            r = self.http.get(f"/rest/api/content/{page_id}/child/attachment", params={"start": start, "limit": limit})
            data = r.json()
            results = data.get("results", [])
            if not results:
                break
            for a in results:
                links = a.get("_links", {})
                download = links.get("download")
                out.append({
                    "id": a.get("id"),
                    "title": a.get("title"),
                    "mediaType": a.get("metadata", {}).get("mediaType"),
                    "comment": a.get("metadata", {}).get("comment"),
                    "created": a.get("history", {}).get("createdDate"),
                    "creator": a.get("history", {}).get("createdBy", {}).get("displayName"),
                    "fileSize": a.get("extensions", {}).get("fileSize"),
                    "download": f"{self.http.base}{download}" if download else None,
                })
            start += limit
        return out

    def list_comments(self, page_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        start = 0
        while True:
            try:
                r = self.http.get(
                    f"/rest/api/content/{page_id}/child/comment",
                    params={"start": start, "limit": limit, "expand": "body.storage,version,history"}
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
                        "body_storage": ((c.get("body") or {}).get("storage") or {}).get("value"),
                    })
                start += limit
            except requests.HTTPError as e:
                if e.response is not None and e.response.status_code == 401:
                    log(f"WARN: Keine Berechtigung Kommentare für Seite {page_id} – überspringe")
                    return []
                raise
        return out

    def get_restrictions(self, page_id: str) -> Dict[str, Any]:
        try:
            r = self.http.get(
                f"/rest/api/content/{page_id}/restriction/byOperation",
                params={"expand": "read.restrictions.user,update.restrictions.user"}
            )
            return r.json()
        except Exception:
            return {}

# ---------------------------- Normalisierung -------------------------

def storage_to_text(storage_html: str) -> str:
    if not storage_html:
        return ""
    soup = BeautifulSoup(storage_html, "html.parser")
    # Markdown-ähnliche Kodierung für Codeblöcke beibehalten
    for code in soup.find_all(["ac:structured-macro", "code", "pre"]):
        txt = code.get_text(" ") or ""
        code.string = "\n```\n" + txt.strip() + "\n```\n"
    text = soup.get_text(" ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# ------------------------------- Dump --------------------------------

def dump_pages(space_keys: List[str], since: Optional[str], out_path: str) -> None:
    base = env_or_die("ATLASSIAN_BASE")
    bearer = os.getenv("ATLASSIAN_BEARER", "").strip() or None
    email = os.getenv("ATLASSIAN_EMAIL", "").strip() or None
    token = os.getenv("ATLASSIAN_TOKEN", "").strip() or None

    http = Http(base, bearer, email, token)
    cx = ConfluenceExtractor(http)

    total_pages = 0
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as out:
        for sk in space_keys:
            log(f"Space {sk}: suche Seiten… (since={since or 'ALL'})")
            for pid in cx.list_pages_cql(sk, since=since):
                p = cx.get_page(pid)
                body_storage = ((p.get("body", {}) or {}).get("storage", {}) or {}).get("value", "")
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

                comments = cx.list_comments(pid)
                attachments = cx.list_attachments(pid)
                restrictions = cx.get_restrictions(pid)

                text_plain = storage_to_text(body_storage)

                row = {
                    "source": "confluence",
                    "id": p.get("id"),
                    "space_key": space.get("key"),
                    "space_name": space.get("name"),
                    "title": p.get("title"),
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
                    "text_plain": text_plain,
                    "comments": comments,
                    "attachments": attachments,
                    "restrictions": restrictions,
                    "extracted_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                }
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                total_pages += 1
    log(f"Fertig. Seiten exportiert: {total_pages}. Datei: {out_path}")

# --------------------------------- CLI -------------------------------

def main():
    ap = argparse.ArgumentParser(description="Confluence kanonischer Extractor")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--space", action="append", help="Space-Key; mehrfach möglich (--space A --space B)")
    g.add_argument("--all-spaces", action="store_true", help="Alle Spaces iterieren")
    ap.add_argument("--since", help="YYYY-MM-DD (optional, lastmodified >=)")
    ap.add_argument("--out", default="data/raw/confluence.jsonl", help="Ausgabedatei (JSONL)")
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

    dump_pages(space_keys, args.since, args.out)

if __name__ == "__main__":
    main()
