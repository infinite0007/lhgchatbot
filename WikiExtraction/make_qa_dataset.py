#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import os
import sys
import re
import json
import time
import argparse
from typing import Dict, Any, Iterable, List, Optional

import requests

# Erstellt Q/A Paare für Finetuning als Datensatz aus Confluence-Seiten.
# Zuerst muss LLAMA gebaut oder als Release vorliegen (https://github.com/ggml-org/llama.cpp), da wir CUDA benutzen ist es wichtig nicht nur die llama-b5921XXX-bin-win-cpu-x64 sondern eben die korrekte cuda Version: llama-b6730XXX-bin-win-cuda-12.4-x64.zip sowie cudart-llama-bin-win-cuda-12.4-x64.zip alles im Ordner zusammenführen (wichtig CUDA Toolkit bei mir 13.0.1 installiert haben)
# Dann kann der Server gestartet werden im LLAMA Ordner mit: llama-server.exe -m "C:\Users\lhglij1\OneDrive - Liebherr\Desktop\Master\lhgchatbot\gemma-3-12b-instruct-gguf\gemma-3-it-12B-Q6_K.gguf" -c 4096 -ngl 999 --device CUDA0 --flash-attn on -b 640 -ub 160 -ctk q8_0 -ctv q8_0 --chat-template gemma --alias gemma-3-12b-instruct --host 127.0.0.1 --port 8080
# Falls ohne GPU das --device raus löschen. Um zu schauen welches Device man hat kann man das auch herausfinden mit: llama-server --list-devices
# Beispielaufruf um Fragen an die KI zu senden und genereren:
# python make_qa_dataset.py --input data/derivatives/joined_pages_full.jsonl --out data/derivatives/qa_dataset.jsonl --pairs-per-page 3 --endpoint http://127.0.0.1:8080/v1/chat/completions --model gemma-3-12b-instruct --max-chars 6000 --temperature 0.2
# ---------------------------- I/O Helfer ----------------------------

def jread(path: str) -> Iterable[Dict[str, Any]]:
    """JSONL zeilenweise lesen (robust gegen leere/fehlerhafte Zeilen)."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                yield json.loads(s)
            except Exception:
                continue


def clean_text(s: str, max_chars: int) -> str:
    """Whitespace normalisieren und hart auf max_chars begrenzen."""
    if not s:
        return ""
    s = re.sub(r"\s+", " ", s).strip()
    if max_chars and len(s) > max_chars:
        return s[:max_chars] + " …"
    return s


# ---------------------------- Prompts ----------------------------

SYSTEM_PROMPT = (
    "Du bist ein Data-Generator für supervised fine-tuning (SFT) im Domänenkontext "
    "Liebherr Software/Kühlgeräte. Aus dem gegebenen Confluence-Kontext erzeugst du "
    "hochwertige, hilfreiche, spezifische Q/A-Paare.\n"
    "REGELN:\n"
    "1) Antworte NUR mit Fakten aus dem Kontext. Keine Halluzinationen.\n"
    "2) Jede Antwort MUSS eine Quellenangabe enthalten (mindestens die Seiten-URL).\n"
    "3) Sprache: Deutsch, außer der Kontext ist eindeutig Englisch.\n"
    "4) Stil: präzise, knapp, sachlich.\n"
    "5) OUTPUT-FORMAT: NDJSON – gib GENAU N Zeilen aus; jede Zeile ist ein JSON-Objekt:\n"
    "   {\"question\": str, \"answer\": str, \"citations\": [{\"url\": str, \"title\": str, \"page_id\": str}], "
    "\"insufficient_evidence\": bool (optional)}\n"
    "6) Verwende die unten angegebene Seiten-URL als Quelle; nenne ggf. Attachment-Titel im Antworttext.\n"
)

USER_TEMPLATE = (
    "Erzeuge {n} Q/A-Paare aus diesem Confluence-Kontext.\n"
    "Seite: {title}\n"
    "URL: {url}\n"
    "Kontext (Auszug):\n"
    "{context}\n\n"
    "Nochmal: Gib GENAU {n} NDJSON-ZEILEN aus, jede Zeile ein JSON-Objekt.\n"
)


# ---------------------------- HTTP Aufruf ----------------------------

def call_chat(endpoint: str,
              model: str,
              messages: List[Dict[str, str]],
              temperature: float,
              max_tokens: int,
              timeout: int = 300,
              retries: int = 3) -> str:
    """
    Ruft einen OpenAI-kompatiblen Chat-Endpunkt auf (llama.cpp --api).
    - Kein Streaming
    - Stop-Tokens passend zu vielen Instrukt-Modellen
    - Retries mit Backoff
    """
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
        "stop": ["<eos>", "<end_of_turn>"]
    }
    last_err: Optional[Exception] = None
    for i in range(retries):
        try:
            r = requests.post(endpoint, json=payload, timeout=timeout)
            r.raise_for_status()
            data = r.json()
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            last_err = e
            # kleiner Backoff
            time.sleep(2 * (i + 1))
    # Wenn alle Retries scheitern:
    if last_err:
        raise last_err
    raise RuntimeError("Unbekannter Fehler beim Chat-Call")


# ---------------------------- Parsing ----------------------------

def extract_code_block(text: str) -> str:
    """
    Falls das Modell in einem Code-Block antwortet (```json ...```), nur den Block extrahieren.
    Ansonsten Rohtext zurückgeben.
    """
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    return fence.group(1).strip() if fence else text


def parse_ndjson_block(text: str, expect_n: int) -> List[Dict[str, Any]]:
    """
    Nimmt das Modellausgabe-Textfeld und extrahiert bis zu expect_n JSON-Objekte (NDJSON).
    Ignoriert Nicht-JSON-Zeilen robust.
    """
    out: List[Dict[str, Any]] = []
    t = extract_code_block(text)
    for line in t.splitlines():
        s = line.strip()
        if not s:
            continue
        # Manchmal liefert das Modell zusätzliche Präfixe -> inneres {...} matchen
        if not (s.startswith("{") and s.endswith("}")):
            m = re.search(r"(\{.*\})", s)
            if not m:
                continue
            s = m.group(1)
        try:
            obj = json.loads(s)
            if isinstance(obj, dict) and "question" in obj and "answer" in obj:
                out.append(obj)
        except Exception:
            continue
        if expect_n and len(out) >= expect_n:
            break
    return out[:expect_n] if expect_n else out


# ---------------------------- Hauptprogramm ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Pfad zur joined_pages_full.jsonl")
    ap.add_argument("--out", required=True, help="Ausgabe-JSONL mit Q/A-Paaren (NDJSON)")
    ap.add_argument("--endpoint", default="http://127.0.0.1:8080/v1/chat/completions",
                    help="OpenAI-kompatibles Chat-Endpoint (llama.cpp --api)")
    ap.add_argument("--model", default="gemma-3-12b-instruct", help="Model-Name (Server-seitig)")
    ap.add_argument("--pairs-per-page", type=int, default=3, dest="pairs_per_page")
    ap.add_argument("--max-chars", type=int, default=6000, dest="max_chars",
                    help="Maximale Zeichen aus dem Page-Text (inkl. Attachments) pro Prompt")
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--max-tokens", type=int, default=768, dest="max_tokens")
    ap.add_argument("--min-context-chars", type=int, default=400, dest="min_context_chars",
                    help="Seiten unterhalb dieser Länge werden übersprungen")
    ap.add_argument("--limit-pages", type=int, default=0, dest="limit_pages",
                    help="Optional: nur die ersten N Seiten verarbeiten (0 = alle)")
    ap.add_argument("--timeout", type=int, default=300, help="HTTP Timeout pro Request (Sekunden)")
    ap.add_argument("--retries", type=int, default=3, help="HTTP Retries bei Fehlern")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    written = 0
    seen_pages = 0

    with open(args.out, "w", encoding="utf-8") as fout:
        for page in jread(args.input):
            seen_pages += 1

            # Kontext auswählen & kürzen
            ctx = page.get("text_plain_plus_attachments") or page.get("text_plain") or ""
            ctx = clean_text(ctx, args.max_chars)
            if len(ctx) < args.min_context_chars:
                # zu wenig Substanz -> skip
                continue

            title = page.get("title") or ""
            url = page.get("url") or ""
            page_id = str(page.get("id") or "")

            sys_msg = {"role": "system", "content": SYSTEM_PROMPT}
            usr_msg = {
                "role": "user",
                "content": USER_TEMPLATE.format(
                    n=args.pairs_per_page,
                    title=title,
                    url=url,
                    context=ctx
                )
            }
            messages = [sys_msg, usr_msg]

            try:
                raw = call_chat(
                    endpoint=args.endpoint,
                    model=args.model,
                    messages=messages,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                    timeout=args.timeout,
                    retries=args.retries,
                )
                rows = parse_ndjson_block(raw, expect_n=args.pairs_per_page)
                if not rows:
                    # kein gültiger Output -> zur nächsten Seite
                    continue

                base_cite = {"url": url, "title": title, "page_id": page_id}

                for r in rows:
                    # Citations reparieren/ergänzen
                    cites = r.get("citations") or []
                    if not isinstance(cites, list):
                        cites = []
                    if not any(isinstance(c, dict) and c.get("url") == url for c in cites):
                        cites.insert(0, base_cite)
                    r["citations"] = cites

                    # Quelle in den Antworttext einfügen, wenn noch nicht vorhanden
                    if isinstance(r.get("answer"), str) and "[Quelle:" not in r["answer"]:
                        r["answer"] = r["answer"].rstrip() + f"\n\n[Quelle: {title}]({url})"

                    # Rückverweise für spätere Nachvollziehbarkeit
                    r["source_page_id"] = page_id
                    r["source_url"] = url
                    r["source_title"] = title

                    fout.write(json.dumps(r, ensure_ascii=False) + "\n")
                    written += 1

            except requests.HTTPError as e:
                print(f"[make_qa] HTTP-Error Seite {page_id}: {e}", file=sys.stderr)
            except requests.RequestException as e:
                print(f"[make_qa] Netzwerkfehler Seite {page_id}: {e}", file=sys.stderr)
            except Exception as e:
                print(f"[make_qa] Fehler Seite {page_id}: {e}", file=sys.stderr)

            if args.limit_pages and seen_pages >= args.limit_pages:
                break

    print(f"[make_qa] Fertig. Beispiele geschrieben: {written} -> {args.out}")


if __name__ == "__main__":
    main()
