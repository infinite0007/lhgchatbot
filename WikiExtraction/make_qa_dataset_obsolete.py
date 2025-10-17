#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Erstellt Q/A Paare für Finetuning als Datensatz aus Confluence-Seiten. Aber genau das ist das Problem - erstellt nur Fragen und immer X viele zu einer Page fässt Kapitel mit gleichen Themen nie zusammen und kann somit nie perfekten Verständnis kreieren. Was hier zudem fehlt ist eine Funktion die z. B. bei einer erstellten Frage diese nochmals in X ausführungen macht aber selbe Antwort so haben wir mehr Datensätze und die KI versteht die Antworten - ansonsten kein lernen möglich bei immer nur einer Frage/Antwort und nebenbei ist es besser Input Output zu haben nicht nur Fragen denn Input kann alles sein auch nur ein Wort oder Wert - weshalb wir KG erstellt haben (dieses dann nutzen)
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
# Dann kann der Server gestartet werden im LLAMA Ordner mit: llama-server.exe -m "C:\\Users\\lhglij1\\OneDrive - Liebherr\\Desktop\\Master\\lhgchatbot\\gemma-3-12b-instruct-gguf\\gemma-3-it-12B-Q6_K.gguf" -c 4096 -ngl 999 --device CUDA0 --flash-attn on -b 640 -ub 160 -ctk q8_0 -ctv q8_0 --chat-template gemma --alias gemma-3-12b-instruct --host 127.0.0.1 --port 8080
# Falls ohne GPU das --device raus löschen. Um zu schauen welches Device man hat kann man das auch herausfinden mit: llama-server --list-devices
# Beispielaufruf um Fragen an die KI zu senden und genereren:
# python make_qa_dataset.py --input data/derivatives/joined_pages_full.jsonl --out data/derivatives/qa_dataset.jsonl --endpoint http://127.0.0.1:8080/v1/chat/completions --model gemma-3-12b-instruct --auto-pairs --pairs-max-per-page 8 --let-model-decide --pairs-min-per-page 0 --max-chars 12000 --min-context-chars 200 --max-tokens 1280 --temperature 0.3
# lass model selber entscheiden wieviele Fragen er generieren soll ohen min und max grenze
# python make_qa_dataset.py --input data/derivatives/joined_pages_full.jsonl --out data/derivatives/qa_dataset_nodupes.jsonl --endpoint http://127.0.0.1:8080/v1/chat/completions --model gemma-3-12b-instruct --auto-pairs --let-model-decide  --max-chars 12000 --min-context-chars 200 --max-tokens 1280 --temperature 0.3 
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
# NEU: Englischer System-Prompt für Support-Kontext (Liebherr-Hausgeräte),
# Pflicht-Quellen, Stil & NDJSON-Format – exakt N Zeilen.
SYSTEM_PROMPT = (
    "You are a data generator for supervised fine-tuning (SFT).\n"
    "ROLE: A professional support assistant for Liebherr-Hausgeräte Ochsenhausen GmbH (software and refrigeration devices).\n"
    "COMPANY INFO (use only when contextually helpful): Liebherr-Hausgeräte Ochsenhausen GmbH, Memminger Str. 77–79, 88416 Ochsenhausen; Phone: 07352 9280; Opening hours: Mon–Fri 09:00–17:00; Sat Closed; Sun Closed.\n\n"
    "RULES:\n"
    "1) Use ONLY facts from the provided Confluence context. No hallucinations.\n"
    "2) Language: English (even if the source text is German).\n"
    "3) Style: concise, precise, support-oriented; include steps or bullet points when useful.\n"
    "4) Every answer MUST include source citations as a markdown link block at the end, e.g. 'Sources: [Title](URL)'. If multiple relevant pages are used, cite them all.\n"
    "5) OUTPUT FORMAT: NDJSON — output EXACTLY N lines; each line is a JSON object with keys:\n"
    '   {"question": str, "answer": str, "citations": [{"url": str, "title": str, "page_id": str}]}\n'
    "6) Ground answers strictly in the provided page. If a detail is not in the context, omit it.\n"
)

# Diversifizierte Frageformen, damit nicht immer „What is the purpose …“ entsteht
USER_TEMPLATE_FIXED = (
    "Create {n} diverse, high-quality Q/A pairs grounded ONLY in the page content.\n"
    "Vary question styles across: definition, how-to (step-by-step), constraints/limits, parameters & defaults, timing/thresholds, troubleshooting, best practices, do/don'ts, glossary/term, short rationale/why.\n"
    "Avoid repeating the same opening (e.g., not always 'What is the purpose'). Prefer action-oriented phrasing.\n"
    "Page title: {title}\n"
    "URL: {url}\n"
    "Context (excerpt):\n"
    "{context}\n\n"
    "Output EXACTLY {n} NDJSON lines; each line is ONE JSON object with keys: "
    '{{"question": "str", "answer": "str", "citations": [{{"url": "str", "title": "str", "page_id": "str"}}]}}.'
)

# NEU: Range-Template, wenn das Modell die Menge selbst wählen soll (0..N)
USER_TEMPLATE_RANGE = (
    "Create BETWEEN {n_min} and {n_max} diverse, high-quality Q/A pairs grounded ONLY in the page content.\n"
    "If content is sparse or redundant, you MAY output fewer pairs (including zero). Do NOT pad with generic or duplicated items.\n"
    "Vary question styles across: definition, how-to (step-by-step), constraints/limits, parameters & defaults, timing/thresholds, troubleshooting, best practices, do/don'ts, glossary/term, short rationale/why.\n"
    "Avoid repeating the same opening. Prefer action-oriented phrasing.\n"
    "Page title: {title}\n"
    "URL: {url}\n"
    "Context (excerpt):\n"
    "{context}\n\n"
    "Return NDJSON: each line ONE JSON object with keys: "
    '{{"question": "str", "answer": "str", "citations": [{{"url": "str", "title": "str", "page_id": "str"}}]}}. '
    "No duplicates."
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
            time.sleep(2 * (i + 1))
    if last_err:
        raise last_err
    raise RuntimeError("Unbekannter Fehler beim Chat-Call")


# ---------------------------- Parsing ----------------------------

def extract_code_block(text: str) -> str:
    """Falls das Modell ```json ...``` nutzt, nur den Block extrahieren."""
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    return fence.group(1).strip() if fence else text


def parse_ndjson_block(text: str, expect_n: int, mode_keys=("question","answer")) -> List[Dict[str, Any]]:
    """
    Robuste Auswertung:
    - Akzeptiert kompletten JSON-Array ODER NDJSON (eine Zeile = ein Objekt)
    - Filtert nur Objekte, die die geforderten Keys enthalten
    - Begrenzt auf expect_n Elemente, falls gesetzt (>0)
    """
    out: List[Dict[str, Any]] = []
    t = extract_code_block(text).strip()

    # 1) kompletter JSON-Array?
    try:
        maybe = json.loads(t)
        if isinstance(maybe, list):
            for obj in maybe:
                if isinstance(obj, dict) and all(k in obj for k in mode_keys):
                    out.append(obj)
            return out[:expect_n] if expect_n else out
    except Exception:
        pass

    # 2) NDJSON fallback
    for line in t.splitlines():
        s = line.strip()
        if not s:
            continue
        if not (s.startswith("{") and s.endswith("}")):
            m = re.search(r"(\{.*\})", s)
            if not m:
                continue
            s = m.group(1)
        try:
            obj = json.loads(s)
            if isinstance(obj, dict) and all(k in obj for k in mode_keys):
                out.append(obj)
        except Exception:
            continue
        if expect_n and len(out) >= expect_n:
            break
    return out[:expect_n] if expect_n else out


# ---------------------------- (kleines) Dedupe ----------------------------

def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())

def dedupe_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    out = []
    for r in rows:
        key = (_norm(r.get("question")), _norm(r.get("answer")))
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


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
    ap.add_argument("--max-tokens", type=int, default=1024, dest="max_tokens")
    ap.add_argument("--min-context-chars", type=int, default=200, dest="min_context_chars",
                    help="Seiten unterhalb dieser Länge werden übersprungen")
    ap.add_argument("--limit-pages", type=int, default=0, dest="limit_pages",
                    help="Optional: nur die ersten N Seiten verarbeiten (0 = alle)")
    ap.add_argument("--timeout", type=int, default=300, help="HTTP Timeout pro Request (Sekunden)")
    ap.add_argument("--retries", type=int, default=3, help="HTTP Retries bei Fehlern")

    # NEU: Automatisch mehr Q/As pro Seite erzeugen, je nach Seitenlänge
    ap.add_argument("--auto-pairs", action="store_true",
                    help="Wenn gesetzt, wird die Anzahl Q/As pro Seite anhand der Kontextlänge skaliert.")
    ap.add_argument("--pairs-scale-min-chars", type=int, default=1200,
                    help="Ab dieser Kontextlänge wird hochskaliert.")
    ap.add_argument("--pairs-scale-max-chars", type=int, default=6000,
                    help="Ab hier ist maximale Skalierung erreicht (Deckel).")
    ap.add_argument("--pairs-max-per-page", type=int, default=8,
                    help="Harte Obergrenze, wenn --auto-pairs aktiv ist.")

    # NEU: Modell entscheidet Anzahl (Range 0..N)
    ap.add_argument("--let-model-decide", action="store_true",
                    help="Modell darf zwischen min und max selbst wählen, inkl. 0.")
    ap.add_argument("--pairs-min-per-page", type=int, default=0,
                    help="Nur relevant mit --let-model-decide: untere Grenze (kann 0 sein).")

    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    written = 0
    seen_pages = 0

    skipped_short = 0
    zero_rows = 0
    api_fail = 0
    invalid_json = 0

    with open(args.out, "w", encoding="utf-8") as fout:
        for page in jread(args.input):
            seen_pages += 1

            # Kontext auswählen & kürzen
            ctx = page.get("text_plain_plus_attachments") or page.get("text_plain") or ""
            ctx = clean_text(ctx, args.max_chars)
            if len(ctx) < args.min_context_chars:
                skipped_short += 1
                continue

            title = page.get("title") or ""
            url = page.get("url") or ""
            page_id = str(page.get("id") or "")

            # Zielanzahl N bestimmen (max)
            N = args.pairs_per_page
            if args.auto_pairs:
                lo = args.pairs_scale_min_chars
                hi = max(args.pairs_scale_max_chars, lo + 1)
                if len(ctx) <= lo:
                    target = args.pairs_per_page
                elif len(ctx) >= hi:
                    target = args.pairs_max_per_page
                else:
                    frac = (len(ctx) - lo) / (hi - lo)
                    target = args.pairs_per_page + frac * (args.pairs_max_per_page - args.pairs_per_page)
                N = int(round(target))
                N = max(args.pairs_per_page, min(args.pairs_max_per_page, N))

            # Prompt bauen – fixed vs. range
            sys_msg = {"role": "system", "content": SYSTEM_PROMPT}
            if args.let_model_decide:
                n_min = max(0, args.pairs_min_per_page)
                n_max = max(n_min, N)
                usr_msg = {
                    "role": "user",
                    "content": USER_TEMPLATE_RANGE.format(
                        n_min=n_min, n_max=n_max, title=title, url=url, context=ctx
                    ),
                }
                messages = [sys_msg, usr_msg]
            else:
                usr_msg = {
                    "role": "user",
                    "content": USER_TEMPLATE_FIXED.format(
                        n=N, title=title, url=url, context=ctx
                    ),
                }
                messages = [sys_msg, usr_msg]

            try:
                if args.let_model_decide:
                    # Ein einzelner Call; Modell darf 0..N liefern (wir kappen auf N)
                    raw = call_chat(
                        endpoint=args.endpoint,
                        model=args.model,
                        messages=messages,
                        temperature=args.temperature,
                        max_tokens=args.max_tokens,
                        timeout=args.timeout,
                        retries=args.retries,
                    )
                    rows = parse_ndjson_block(raw, expect_n=0, mode_keys=("question","answer"))
                    rows = dedupe_rows(rows)[:N]  # dedupe + kappen auf N (max)
                else:
                    # Refill-Loop wie bisher, bis N erreicht (falls Modell weniger liefert)
                    needed = N
                    rows: List[Dict[str, Any]] = []
                    attempt = 0
                    while len(rows) < needed and attempt < 3:
                        raw = call_chat(
                            endpoint=args.endpoint,
                            model=args.model,
                            messages=messages,
                            temperature=args.temperature,
                            max_tokens=args.max_tokens,
                            timeout=args.timeout,
                            retries=args.retries,
                        )
                        got = parse_ndjson_block(raw, expect_n=needed - len(rows), mode_keys=("question","answer"))
                        if not isinstance(got, list):
                            invalid_json += 1
                            got = []
                        rows.extend(got)
                        rows = dedupe_rows(rows)
                        attempt += 1

                if not rows:
                    zero_rows += 1
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

                    # Quelle in den Antworttext einfügen, wenn noch nicht vorhanden (ENGLISCH)
                    if isinstance(r.get("answer"), str) and "[Source:" not in r["answer"] and "Sources:" not in r["answer"]:
                        r["answer"] = r["answer"].rstrip()
                        if cites:
                            src_block = "Sources: " + ", ".join(
                                f"[{c.get('title','source')}]({c['url']})" for c in cites if c.get("url")
                            )
                            r["answer"] += f"\n\n{src_block}"
                        else:
                            r["answer"] += f"\n\n[Source: {title}]({url})"

                    # Rückverweise für spätere Nachvollziehbarkeit
                    r["source_page_id"] = page_id
                    r["source_url"] = url
                    r["source_title"] = title

                    fout.write(json.dumps(r, ensure_ascii=False) + "\n")
                    written += 1

            except requests.HTTPError as e:
                api_fail += 1
                print(f"[make_qa] HTTP-Error Seite {page_id}: {e}", file=sys.stderr)
            except requests.RequestException as e:
                api_fail += 1
                print(f"[make_qa] Netzwerkfehler Seite {page_id}: {e}", file=sys.stderr)
            except Exception as e:
                invalid_json += 1
                print(f"[make_qa] Fehler Seite {page_id}: {e}", file=sys.stderr)

            if args.limit_pages and seen_pages >= args.limit_pages:
                break

    print(
        f"[make_qa] Fertig. Beispiele geschrieben: {written} -> {args.out} | "
        f"pages_seen={seen_pages}, skipped_short={skipped_short}, zero_rows={zero_rows}, "
        f"api_fail={api_fail}, invalid_json={invalid_json}"
    )


if __name__ == "__main__":
    main()
