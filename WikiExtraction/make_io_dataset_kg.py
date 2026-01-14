#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
make_io_dataset_kg.py
------------------------------------

ZWECK
  Dieses Skript baut aus Confluence-Seiten (JSONL) zunächst ein leichtes
  Wissensmodell (Knowledge Graph, kurz KG) aus Triples (subj, pred, obj).
  Danach erzeugt es pro Entität (optional in thematische FACETTEN gesplittet)
  ein SFT-Dataset aus INPUT→OUTPUT-Paaren für Completion-only-Loss Training
  (z. B. mit TRL SFTTrainer + `completion_only_loss` via Masking).

NEU / FIXES
  - Filtert uninformative Entitäten (IDs, reine Zahlen, kurze Codes).
  - Optionaler Kanonisierungsschritt per LLM (selber Endpoint) normalisiert
    Entitätsnamen („Sabbath mode“, „child lock“, …) und repariert Triples.
  - IO-Generator erstellt IMMER exakt N dissimilare Inputs; wenn zu wenige
    kommen, füllt er via Paraphrasen + Templates auf — Output bleibt identisch.
  - Outputs gehen „in die Tiefe“: Definition, Preconditions, Parameter + Units,
    Edge-Cases, „Case A/Case B“, kurze Beispiele, Klartext bei Ambiguitäten.
  - Inputs müssen die Entität explizit nennen (keine vagen Fragen).

EINGABE (JSONL pro Zeile – z. B. „joined pages“):
  {
    "source": "confluence",
    "id": "160863104",
    "space_key": "...",
    "title": "REQ Door Heaters - Humidity and Ambient Temperature Based",
    "url": "https://helpd-doc....",
    "text_plain_plus_attachments": "...",  # bevorzugter langer Text
    "text_plain": "..."                     # Fallback
  }

AUSGABE (NDJSON, eine Zeile pro INPUT→OUTPUT-Variante):
  Pflichtfelder:
    - input:      str
    - output:     str  (kanonisch, identisch für alle Varianten einer Gruppe)
    - citations:  List[{"url": str, "title": str, "page_id": str}]
  Metafelder:
    - entity, facet, variation_group, auto_or_fixed_*,
      auto_variants_count, outputs_k_for_entity,
      triples_in_facet, triples_in_entity,
      insufficient_evidence (optional), conflict_notes (optional)

# Erstellt Q/A Paare für Finetuning als Datensatz aus Confluence-Seiten.
# Zuerst muss LLAMA gebaut oder als Release vorliegen (https://github.com/ggml-org/llama.cpp), da wir CUDA benutzen ist es wichtig nicht nur die llama-b5921XXX-bin-win-cpu-x64 sondern eben die korrekte cuda Version: llama-b6730XXX-bin-win-cuda-12.4-x64.zip sowie cudart-llama-bin-win-cuda-12.4-x64.zip alles im Ordner zusammenführen (wichtig CUDA Toolkit bei mir 13.0.1 installiert haben)
# Dann kann der Server gestartet werden im LLAMA Ordner mit: llama-server.exe -m "C:\\Users\\lhglij1\\OneDrive - Liebherr\\Desktop\\Master\\lhgchatbot\\gemma-3-12b-instruct-gguf\\gemma-3-it-12B-Q6_K.gguf" -c 4096 -ngl 999 --device CUDA0 --flash-attn on -b 640 -ub 160 -ctk q8_0 -ctv q8_0 --chat-template gemma --alias gemma-3-12b-instruct --host 127.0.0.1 --port 8080
# Falls ohne GPU das --device raus löschen. Um zu schauen welches Device man hat kann man das auch herausfinden mit: llama-server --list-devices
# Beispielaufruf um Fragen an die KI zu senden und genereren:
  python make_io_dataset_kg.py --input data/derivatives/joined_pages_full.jsonl --out data/derivatives/io_dataset.jsonl --endpoint http://127.0.0.1:8080/v1/chat/completions --model gemma-3-12b-instruct --kg-out data/derivatives/kg_triples.jsonl --top-entities 0 --facet-split --min-triples-per-entity 1 --min-triples-per-facet 1 --outputs-per-entity 0 --variants-per-output 10 --generator-temperature 0.2

  python make_io_dataset_kg.py \
    --input data/derivatives/joined_pages_full.jsonl \
    --out data/derivatives/io_dataset.jsonl \
    --endpoint http://127.0.0.1:8080/v1/chat/completions \
    --model gemma-3-12b-instruct \
    --kg-out data/derivatives/kg_triples.jsonl \
    --top-entities 0 \
    --facet-split \
    --min-triples-per-entity 1 \
    --min-triples-per-facet 1 \
    --outputs-per-entity 0 \
    --variants-per-output 6 \
    --generator-temperature 0.2
"""

from __future__ import annotations
import os, re, json, time, argparse, hashlib, math, random
from typing import Dict, Any, Iterable, List, Optional, Tuple, Set
import requests

# ──────────────────────────────────────────────────────────────────────────────
# [I/O HELFER]
# ──────────────────────────────────────────────────────────────────────────────

def jread(path: str) -> Iterable[Dict[str, Any]]:
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
    if not s:
        return ""
    s = re.sub(r"\s+", " ", s).strip()
    if max_chars and len(s) > max_chars:
        return s[:max_chars] + " …"
    return s

def ensure_dir(path: str) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)

def norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip().lower()

# ──────────────────────────────────────────────────────────────────────────────
# [CHAT-API]
# ──────────────────────────────────────────────────────────────────────────────

def call_chat(endpoint: str,
              model: str,
              messages: List[Dict[str, str]],
              temperature: float,
              max_tokens: int,
              timeout: int = 300,
              retries: int = 3,
              stop: Optional[List[str]] = None) -> str:
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    if stop:
        payload["stop"] = stop

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
    raise RuntimeError("Unknown error in call_chat")

# ──────────────────────────────────────────────────────────────────────────────
# [PARSING & ÄHNLICHKEIT]
# ──────────────────────────────────────────────────────────────────────────────

def extract_code_block(text: str) -> str:
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    return fence.group(1).strip() if fence else text

def parse_ndjson_objects(text: str, min_keys: Set[str]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    t = extract_code_block(text)
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
            if isinstance(obj, dict) and min_keys.issubset(obj.keys()):
                out.append(obj)
        except Exception:
            continue
    return out

def extract_json_array(text: str) -> List[Any]:
    t = extract_code_block(text).strip()
    m = re.search(r"(\[.*\])", t, flags=re.DOTALL)
    if not m:
        return []
    try:
        arr = json.loads(m.group(1))
        return arr if isinstance(arr, list) else []
    except Exception:
        return []

def strip_sources_block(s: str) -> str:
    return re.sub(r"\n\s*Sources:\s*[\s\S]*$", "", s, flags=re.I).strip()

def wordset(s: str) -> Set[str]:
    s = re.sub(r"[^\w\s]", " ", s.lower())
    toks = [t for t in s.split() if t]
    return set(toks)

def too_similar(a: str, b: str, jaccard_threshold: float = 0.8) -> bool:
    A, B = wordset(a), wordset(b)
    if not A or not B:
        return True
    inter = len(A & B)
    union = len(A | B)
    return (inter / union) >= jaccard_threshold

# ──────────────────────────────────────────────────────────────────────────────
# [KG PROMPTS]
# ──────────────────────────────────────────────────────────────────────────────

SYSTEM_KG_EXTRACTOR = (
    "You are an expert information extraction assistant. From the provided page text, "
    "extract precise factual triples (subject, predicate, object) relevant to household "
    "refrigeration software, electronics, parameters, modes, sensors, timings, thresholds, units.\n"
    "QUALITY RULES:\n"
    "1) Language: English. Use canonical, human-meaningful entity names: e.g., 'Sabbath mode', "
    "'child lock', 'ambient temperature sensor'.\n"
    "2) REJECT non-informative subjects/objects: IDs, pure numbers, single codes like '12345', "
    "'CFG_01', '{GUID}', or raw filenames. Prefer the meaningful feature or parameter name instead.\n"
    "3) Keep units explicit for example units like (°C, %, ms/s/min/h) or others.\n"
    "4) OUTPUT FORMAT: NDJSON; each line JSON:\n"
    "   {\"subj\": str, \"pred\": str, \"obj\": str, \"page_id\": str, \"page_title\": str, \"page_url\": str}\n"
    "5) One fact per line. Only include facts present in the text. No hallucinations.\n"
)

USER_KG_EXTRACTOR = (
    "Page title: {title}\n"
    "URL: {url}\n"
    "Page ID: {page_id}\n"
    "Context (excerpt, normalized):\n"
    "{context}\n\n"
    "Extract up to {max_triples} high-quality factual triples as NDJSON."
)

SYSTEM_KG_NORMALIZER = (
    "You are a knowledge-graph triple normalizer. Given raw triples, rewrite SUBJECT/OBJECT into "
    "concise, canonical, human-meaningful entities (e.g., 'Sabbath mode', 'child lock').\n"
    "RULES:\n"
    "- DROP any triple where subject/object are IDs, numbers, or meaningless codes.\n"
    "- KEEP units and exact values in the object when informative.\n"
    "- Return NDJSON with the same keys."
)

USER_KG_NORMALIZER = (
    "Normalize the following triples:\n{triples}\n\n"
    "Return NDJSON lines; exclude low-quality triples."
)

# ──────────────────────────────────────────────────────────────────────────────
# [IO GENERATOR PROMPTS]
# ──────────────────────────────────────────────────────────────────────────────

SUPPORT_PERSONA_BLOCK = (
    "You are a helpful support assistant for Liebherr-Hausgeräte Ochsenhausen GmbH "
    "(Memminger Str. 77–79, 88416 Ochsenhausen, Germany). Phone: +49 7352 9280. "
    "Opening hours: Mon–Fri 09:00–17:00; Sat–Sun closed."
)

SYSTEM_GENERATOR = (
    "You generate INPUT→OUTPUT training pairs from a small knowledge subgraph and its sources.\n"
    "Soft guidance (no hard rules):\n"
    "- Produce natural, strongly varied INPUTs: sometimes questions, sometimes short instructions, sometimes plain keyword-like or statement-style lines.\n"
    "- Vary the **opening** of inputs explicitly (do not always start with the same leading word). Mix starts like 'How/Which/Why/When', imperative verbs ('Configure/Explain/Outline'), or bare phrases.\n"
    "- Prefer **unlabeled** inputs: avoid leading headings like 'Keywords:', 'Describe:', 'Question:', 'Task:' at the start. Write the phrase itself instead (e.g., 'Liebherr cooling, emergency mode, zone off').\n"
    "- Keep the single OUTPUT identical across all variants for the same subgraph.\n"
    "- OUTPUT depth: start with a 1–2 sentence summary, then compact bullets or a short step-by-step covering purpose, preconditions, parameters with units, limits/edge-cases, and a tiny example.\n"
    "- If anything is ambiguous, state what is needed instead of guessing.\n"
    "- Every OUTPUT must end with:\n"
    "  \\n\\nSources:\\n- [<Title>](<URL>)\n"
    "- Return strictly NDJSON: one JSON object per line:\n"
    " {\"input\": str, \"output\": str, \"citations\": [{\"url\": str, \"title\": str, \"page_id\": str}]}\n"
    "Do not use code fences.\n"
)

USER_GENERATOR = (
    "Topic: {entity}\n"
    "Facet (optional): {facet_name}\n"
    "Knowledge subgraph (triples):\n"
    "{triples_text}\n\n"
    "Sources:\n"
    "{sources_text}\n\n"
    "Generate EXACTLY {n} NDJSON lines with **strongly varied INPUT openings** (some questions, some instructions, some statements/keyword-style). "
    "Avoid leading labels like 'Keywords:' / 'Describe:' / 'Question:' — write the phrase directly. "
    "Do NOT wrap in a code block. One JSON object per line."
)

SYSTEM_PARAPHRASE = (
    "You paraphrase user prompts into mutually dissimilar forms while preserving intent and constraints. "
    "Avoid trivial synonym swaps and near-duplicates."
)
USER_PARAPHRASE = (
    "Entity/topic: {entity}\n"
    "Canonical answer (for intent alignment):\n{output}\n\n"
    "Existing prompts:\n{existing}\n\n"
    "Generate {need} NEW prompts (question/instruction/keywords/scenario). "
    "Each MUST explicitly reference the entity/topic.\n"
    "Return ONLY a JSON array of strings."
)

# ──────────────────────────────────────────────────────────────────────────────
# [KG UTILS]
# ──────────────────────────────────────────────────────────────────────────────

class Triple(Dict[str, Any]):
    """subj, pred, obj, page_id, page_title, page_url"""
    pass

_BAD_ENTITY_RX = re.compile(
    r"^(?:[0-9]+|[A-Za-z]*\d+[A-Za-z]*|[A-F0-9\-]{6,}|[_.\-]+)$"
)

def _is_informative_entity(s: str) -> bool:
    s = (s or "").strip()
    if len(s) < 3:
        return False
    if not re.search(r"[A-Za-z]{3,}", s):
        return False
    if _BAD_ENTITY_RX.match(s):
        return False
    if re.search(r"\.(bin|hex|cfg|ini|json|txt|csv|log)$", s, flags=re.I):
        return False
    return True

def _filter_and_clean_triples(triples: List[Triple]) -> List[Triple]:
    out: List[Triple] = []
    for t in triples:
        subj, obj = t.get("subj","").strip(), t.get("obj","").strip()
        pred = t.get("pred","").strip()
        if not _is_informative_entity(subj) or not _is_informative_entity(obj):
            continue
        if len(pred) < 2:
            continue
        out.append(t)
    return out

def extract_triples_for_page(endpoint: str, model: str, title: str, url: str, page_id: str,
                             context: str, max_triples: int, temperature: float,
                             max_tokens: int, timeout: int, retries: int) -> List[Triple]:
    sys_msg = {"role": "system", "content": SYSTEM_KG_EXTRACTOR}
    usr_msg = {"role": "user", "content": USER_KG_EXTRACTOR.format(
        title=title, url=url, page_id=page_id, context=context, max_triples=max_triples)}
    raw = call_chat(endpoint, model, [sys_msg, usr_msg], temperature, max_tokens, timeout, retries,
                    stop=["<eos>", "<end_of_turn>"])
    objs = parse_ndjson_objects(raw, min_keys={"subj", "pred", "obj"})
    triples: List[Triple] = []
    for o in objs[:max_triples]:
        t: Triple = {
            "subj": str(o.get("subj", "")).strip(),
            "pred": str(o.get("pred", "")).strip(),
            "obj": str(o.get("obj", "")).strip(),
            "page_id": str(o.get("page_id") or page_id),
            "page_title": str(o.get("page_title") or title),
            "page_url": str(o.get("page_url") or url),
        }
        if len(t["subj"]) < 2 or len(t["pred"]) < 2 or len(t["obj"]) < 1:
            continue
        triples.append(t)
    triples = _filter_and_clean_triples(triples)
    return triples

def normalize_triples(endpoint: str, model: str, triples: List[Triple],
                      temperature: float, max_tokens: int, timeout: int, retries: int) -> List[Triple]:
    if not triples:
        return triples
    need = any(len(t["subj"]) < 6 or len(t["obj"]) < 6 for t in triples)
    if not need:
        return triples
    chunk = triples[:40]
    nd = "\n".join(json.dumps(t, ensure_ascii=False) for t in chunk)
    sys_msg = {"role": "system", "content": SYSTEM_KG_NORMALIZER}
    usr_msg = {"role": "user", "content": USER_KG_NORMALIZER.format(triples=nd)}
    try:
        raw = call_chat(endpoint, model, [sys_msg, usr_msg], temperature, max_tokens, timeout, retries,
                        stop=["<eos>", "<end_of_turn>"])
        objs = parse_ndjson_objects(raw, min_keys={"subj", "pred", "obj"})
        objs = _filter_and_clean_triples([{
            "subj": str(o.get("subj","")).strip(),
            "pred": str(o.get("pred","")).strip(),
            "obj": str(o.get("obj","")).strip(),
            "page_id": str(o.get("page_id","")),
            "page_title": str(o.get("page_title","")),
            "page_url": str(o.get("page_url","")),
        } for o in objs])
        return objs if objs else triples
    except Exception:
        return triples

def index_triples_by_entity(triples: List[Triple]) -> Dict[str, List[Triple]]:
    idx: Dict[str, List[Triple]] = {}
    for t in triples:
        for ent in (t["subj"], t["obj"]):
            if not ent:
                continue
            idx.setdefault(ent, []).append(t)
    return idx

def top_entities(idx: Dict[str, List[Triple]], k: int) -> List[str]:
    items = sorted(idx.items(), key=lambda kv: len(kv[1]), reverse=True)
    if k <= 0:
        return [ent for ent, _ in items]
    return [ent for ent, _ in items[:k]]

def triples_to_text(triples: List[Triple], max_lines: int = 120) -> str:
    return "\n".join(f"- ({t['subj']}) --{t['pred']}--> ({t['obj']})  [{t['page_title']}]"
                     for t in triples[:max_lines])

def sources_from_triples(triples: List[Triple]) -> List[Dict[str, str]]:
    seen: Set[Tuple[str, str]] = set()
    out: List[Dict[str, str]] = []
    for t in triples:
        k = (t["page_url"], t["page_title"])
        if k in seen:
            continue
        seen.add(k)
        out.append({"url": t["page_url"], "title": t["page_title"], "page_id": t["page_id"]})
    return out

def render_sources_md(sources: List[Dict[str, str]], max_sources: int = 20) -> str:
    return "\n".join(f"- [{s.get('title','Source')}]({s.get('url','')}) (ID: {s.get('page_id','')})"
                     for s in sources[:max_sources])

# ──────────────────────────────────────────────────────────────────────────────
# [FACETS]
# ──────────────────────────────────────────────────────────────────────────────

FACET_RULES = {
    "purpose": re.compile(r"(purpose|definition|used for|behavior|behaviour|overview|function)", re.I),
    "activation": re.compile(r"(enable|disable|activation|deactivation|\bon\b|\boff\b|state|mode)", re.I),
    "temperature": re.compile(r"(temp|°c|setpoint|threshold|limit|hysteresis|warming|cooling)", re.I),
    "ui": re.compile(r"(ui|button|display|menu|setting|user interface|screen)", re.I),
    "robustness": re.compile(r"(error|alarm|fault|robust|safety|fallback)", re.I),
}

def split_into_facets(triples: List[Triple], min_triples_per_facet: int) -> Dict[str, List[Triple]]:
    buckets: Dict[str, List[Triple]] = {k: [] for k in FACET_RULES.keys()}
    buckets["misc"] = []
    for t in triples:
        text = f"{t['pred']} {t['obj']}"
        placed = False
        for name, rx in FACET_RULES.items():
            if rx.search(text):
                buckets[name].append(t)
                placed = True
                break
        if not placed:
            buckets["misc"].append(t)
    buckets = {k: v for k, v in buckets.items() if len(v) >= min_triples_per_facet}
    if not buckets:
        return {"misc": triples}
    return buckets

# ──────────────────────────────────────────────────────────────────────────────
# [INPUT-QUALITÄT / DUPLIKATE / FALLBACKS]
# ──────────────────────────────────────────────────────────────────────────────

def entity_tokens(entity: str) -> Set[str]:
    toks = [t for t in re.split(r"[^\w]+", entity or "") if len(t) >= 3]
    return set(t.lower() for t in toks)

def is_low_quality_input(q: str, ent_toks: Set[str]) -> bool:
    if len(q.strip()) < 8 or len(q.split()) < 3:
        return True
    ql = q.lower()
    if not any(t in ql for t in ent_toks):
        vague = re.search(r"\b(build|version|date|when|what)\b", ql) is not None
        return vague
    return False

def fixup_input(q: str, entity: str) -> str:
    q = q.strip()
    if not q.endswith("?") and q.lower().startswith(("what", "when", "how", "where", "why")):
        q += "?"
    if entity and entity.lower() not in q.lower():
        joiner = " for " if " for " not in q.lower() else " "
        q = q.rstrip(".") + f"{joiner}{entity}".rstrip() + "."
    return q

def paraphrase_inputs(endpoint: str, model: str, entity: str, canonical_output: str,
                      existing: List[str], need: int,
                      temperature: float, max_tokens: int, timeout: int, retries: int) -> List[str]:
    if need <= 0:
        return []
    sys_msg = {"role": "system", "content": SYSTEM_PARAPHRASE}
    existing_block = "\n".join(f"- {x}" for x in existing) if existing else "(none)"
    usr_msg = {"role": "user", "content": USER_PARAPHRASE.format(
        entity=entity, output=canonical_output, existing=existing_block, need=need)}
    raw = call_chat(endpoint, model, [sys_msg, usr_msg],
                    temperature=max(temperature, 0.55),
                    max_tokens=max(256, max_tokens // 4),
                    timeout=timeout, retries=retries,
                    stop=["<eos>", "<end_of_turn>"])
    arr = extract_json_array(raw)
    return [str(x).strip() for x in arr if str(x).strip()]

def template_prompts(entity: str, facet_name: str, k: int) -> List[str]:
    entity = (entity or "the feature").strip()
    facet = (facet_name or "").strip()
    tag = f"{entity}" if facet in ("", "misc") else f"{entity} {facet}"
    base = [
        f"What should I know about {tag}?",
        f"Give me a concise explanation of {tag}.",
        f"Explain how {tag} works.",
        f"Instruction: summarize {tag}.",
        f"{tag} quick reference.",
        f"{tag} — key parameters and constraints?",
        f"How to configure {tag}?",
        f"Troubleshooting notes for {tag}?",
        f"Best practices for {tag}?",
        f"{tag} keywords and purpose.",
    ]
    random.shuffle(base)
    return base[:max(0, k)]

# ──────────────────────────────────────────────────────────────────────────────
# [GENERATION & HARMONISIERUNG]
# ──────────────────────────────────────────────────────────────────────────────

_MIN_INPUT_JACCARD = 0.8

def _harmonize_rows(rows: List[Dict[str, Any]],
                    subgraph_sources: List[Dict[str, str]],
                    n_variants: int,
                    max_sources_per_output: Optional[int]) -> List[Dict[str, Any]]:
    # 1) Dissimile Inputs wählen
    uniq_rows: List[Dict[str, Any]] = []
    for r in rows:
        inp = str(r.get("input", "")).strip()
        if not inp:
            continue
        if any(too_similar(inp, u.get("input",""), _MIN_INPUT_JACCARD) for u in uniq_rows):
            continue
        uniq_rows.append(r)
        if len(uniq_rows) >= n_variants:
            break
    if not uniq_rows:
        return []

    # 2) Kanonischer Output
    canonical_output = ""
    for r in uniq_rows:
        out = str(r.get("output", "")).strip()
        if out:
            canonical_output = out
            break
    if not canonical_output:
        canonical_output = "No information available based on the provided sources."

    # 3) Zitate zusammenführen
    merged_map: Dict[Tuple[str, str], Dict[str, str]] = {}
    def add_cites(cites):
        if not isinstance(cites, list):
            return
        for c in cites:
            if not isinstance(c, dict):
                continue
            url = c.get("url", "")
            pid = c.get("page_id", "")
            title = c.get("title") or "Source"
            if not url:
                continue
            merged_map[(url, pid)] = {"url": url, "title": title, "page_id": pid}

    for r in uniq_rows:
        add_cites(r.get("citations"))
    add_cites(subgraph_sources)

    merged_cites = list(merged_map.values())
    if max_sources_per_output and max_sources_per_output > 0:
        merged_cites = merged_cites[:max_sources_per_output]

    # 4) Sources-Block sicherstellen
    src_lines = "\n".join(f"- [{c['title']}]({c['url']})" for c in merged_cites if c.get("url"))
    out_text = canonical_output.strip()
    if "Sources:" not in out_text and src_lines:
        out_text = out_text.rstrip() + "\n\nSources:\n" + src_lines

    # 5) Variation Group
    vg = hashlib.sha256(out_text.encode("utf-8")).hexdigest()[:8]

    final_rows: List[Dict[str, Any]] = []
    for r in uniq_rows:
        final_rows.append({
            "input": str(r.get("input", "")).strip(),
            "output": out_text,
            "citations": merged_cites,
            "variation_group": vg,
        })
    return final_rows

def generate_variants_for_facet(endpoint: str,
                                model: str,
                                entity: str,
                                facet_name: str,
                                facet_triples: List[Triple],
                                n_variants: int,
                                temperature: float,
                                max_tokens: int,
                                timeout: int,
                                retries: int,
                                max_sources_per_output: Optional[int]) -> List[Dict[str, Any]]:
    sys_msg = {"role": "system", "content": SYSTEM_GENERATOR}
    triples_txt = triples_to_text(facet_triples, max_lines=120)
    srcs = sources_from_triples(facet_triples)
    usr_msg = {"role": "user", "content": USER_GENERATOR.format(
        entity=entity,
        facet_name=facet_name,
        triples_text=triples_txt,
        sources_text=render_sources_md(srcs, max_sources=20),
        n=n_variants,
    )}
    raw = call_chat(endpoint, model, [sys_msg, usr_msg],
                    temperature, max_tokens, timeout, retries,
                    stop=["<eos>", "<end_of_turn>"])
    rows = parse_ndjson_objects(raw, min_keys={"input", "output"})
    rows = _harmonize_rows(rows, srcs, n_variants, max_sources_per_output)

    # Robust: Auffüllen bis N
    if len(rows) < n_variants:
        ent_toks = entity_tokens(entity)
        canonical_output = rows[0]["output"] if rows else ""
        existing_inputs = [r["input"] for r in rows]
        need = n_variants - len(rows)

        new_inputs = paraphrase_inputs(
            endpoint=endpoint, model=model, entity=entity,
            canonical_output=canonical_output or f"About {entity} / facet {facet_name}.",
            existing=existing_inputs, need=need,
            temperature=temperature, max_tokens=max_tokens,
            timeout=timeout, retries=retries
        )

        fixed_new: List[str] = []
        for q in new_inputs:
            q0 = q
            if is_low_quality_input(q0, ent_toks):
                q0 = fixup_input(q0, entity)
            if not any(too_similar(q0, e, _MIN_INPUT_JACCARD) for e in existing_inputs + fixed_new):
                fixed_new.append(q0)

        if len(fixed_new) < need:
            templ = template_prompts(entity, facet_name, need * 2)
            for q in templ:
                q0 = q
                if is_low_quality_input(q0, ent_toks):
                    q0 = fixup_input(q0, entity)
                if not any(too_similar(q0, e, _MIN_INPUT_JACCARD) for e in existing_inputs + fixed_new):
                    fixed_new.append(q0)
                if len(fixed_new) >= need:
                    break

        if not canonical_output:
            src_lines = "\n".join(f"- [{c['title']}]({c['url']})" for c in srcs if c.get("url"))
            canonical_output = f"Information about {entity} ({facet_name}). Refer to the sources below.\n\nSources:\n{src_lines}"

        vg = hashlib.sha256(canonical_output.encode("utf-8")).hexdigest()[:8]
        cites = rows[0]["citations"] if rows else srcs

        for q in fixed_new[:need]:
            rows.append({
                "input": q,
                "output": canonical_output,
                "citations": cites,
                "variation_group": vg,
            })

    return rows[:n_variants]

# ──────────────────────────────────────────────────────────────────────────────
# [DETECTOR] (optional)
# ──────────────────────────────────────────────────────────────────────────────

def detect_conflict(endpoint: str, model: str, facet_triples: List[Triple], output_text: str,
                    temperature: float, max_tokens: int, timeout: int, retries: int) -> Tuple[bool, str]:
    if not output_text.strip():
        return False, ""
    sys_msg = {"role": "system", "content": (
        "You are a strict fact consistency checker. Given a knowledge subgraph (triples) and a draft OUTPUT, "
        "return a single JSON line with keys: {\"has_conflict\": bool, \"notes\": str}. "
        "Set 'has_conflict' to true if any claim in OUTPUT contradicts or overgeneralizes the subgraph."
    )}
    usr_msg = {"role": "user", "content": (
        "Subgraph:\n{triples_text}\n\nDraft OUTPUT:\n{output_text}\n"
        "Return ONE JSON object only."
    ).format(triples_text=triples_to_text(facet_triples, max_lines=120), output_text=output_text)}
    try:
        raw = call_chat(endpoint, model, [sys_msg, usr_msg],
                        temperature, max_tokens, timeout, retries,
                        stop=["<eos>", "<end_of_turn>"])
        objs = parse_ndjson_objects(raw, min_keys={"has_conflict", "notes"})
        if objs:
            return bool(objs[0].get("has_conflict", False)), str(objs[0].get("notes", "")).strip()
    except Exception:
        pass
    return False, ""

# ──────────────────────────────────────────────────────────────────────────────
# [AUTO HEURISTIK]
# ──────────────────────────────────────────────────────────────────────────────

DEFAULT_AUTO_OUTPUTS_TARGET_TRIPLES = 6

def decide_outputs_per_entity_auto(total_triples_for_entity: int,
                                   num_facets_after_filter: int,
                                   target_triples: int = DEFAULT_AUTO_OUTPUTS_TARGET_TRIPLES) -> int:
    if num_facets_after_filter <= 0:
        return 0
    est = max(1, math.ceil(max(1, total_triples_for_entity) / max(1, target_triples)))
    return min(est, num_facets_after_filter)

def decide_variants_per_output_auto(triples_in_facet: int) -> int:
    v = round(1.6 * math.log2(1 + max(0, triples_in_facet)) + 1.4)
    return max(2, min(v, 8))

# ──────────────────────────────────────────────────────────────────────────────
# [MAIN]
# ──────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to joined_pages_full.jsonl")
    ap.add_argument("--out", required=True, help="Output NDJSON of IO pairs")
    ap.add_argument("--kg-out", default="", help="(Optional) Save extracted triples NDJSON here")
    ap.add_argument("--endpoint", default="http://127.0.0.1:8080/v1/chat/completions",
                    help="OpenAI-compatible chat endpoint")
    ap.add_argument("--model", default="gemma-3-12b-instruct", help="Model name (server-side)")

    # Paging / Textgröße
    ap.add_argument("--max-page-chars", type=int, default=10000)
    ap.add_argument("--min-context-chars", type=int, default=400)

    # Extractor
    ap.add_argument("--extractor-max-triples-per-page", type=int, default=30)
    ap.add_argument("--extractor-temperature", type=float, default=0.0)
    ap.add_argument("--extractor-max-tokens", type=int, default=768)

    # Entities & Facets
    ap.add_argument("--top-entities", type=int, default=0, help="0 = all entities")
    ap.add_argument("--min-triples-per-entity", type=int, default=1)
    ap.add_argument("--facet-split", action="store_true")
    ap.add_argument("--min-triples-per-facet", type=int, default=1)

    # Steuerung Facetten/Varianten
    ap.add_argument("--outputs-per-entity", type=int, default=0,
                    help=">0 fixed, 0 auto")
    ap.add_argument("--variants-per-output", type=int, default=5,
                    help=">0 fixed, 0 auto")

    # Generator
    ap.add_argument("--generator-temperature", type=float, default=0.2)
    ap.add_argument("--generator-max-tokens", type=int, default=1024)
    ap.add_argument("--max-sources-per-output", type=int, default=10)

    # Detector (optional)
    ap.add_argument("--use-detector", action="store_true")
    ap.add_argument("--detector-temperature", type=float, default=0.0)
    ap.add_argument("--detector-max-tokens", type=int, default=256)

    # Networking / Caps
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--limit-pages", type=int, default=0)
    ap.add_argument("--max-total-pairs", type=int, default=0)

    args = ap.parse_args()

    ensure_dir(args.out)
    if args.kg_out:
        ensure_dir(args.kg_out)

    # STEP 1: KG-Extraktion
    all_triples: List[Triple] = []
    page_count = 0

    for page in jread(args.input):
        page_count += 1
        if args.limit_pages and page_count > args.limit_pages:
            break

        body = page.get("text_plain_plus_attachments") or page.get("text_plain") or ""
        body = clean_text(body, args.max_page_chars)
        if len(body) < args.min_context_chars:
            continue

        title = page.get("title") or ""
        url = page.get("url") or ""
        page_id = str(page.get("id") or page.get("page_id") or "")

        try:
            triples = extract_triples_for_page(
                endpoint=args.endpoint, model=args.model,
                title=title, url=url, page_id=page_id, context=body,
                max_triples=args.extractor_max_triples_per_page,
                temperature=args.extractor_temperature,
                max_tokens=args.extractor_max_tokens,
                timeout=args.timeout, retries=args.retries,
            )
            triples = normalize_triples(
                endpoint=args.endpoint, model=args.model, triples=triples,
                temperature=args.extractor_temperature,
                max_tokens=args.extractor_max_tokens,
                timeout=args.timeout, retries=args.retries,
            )
            all_triples.extend(triples)
        except requests.HTTPError as e:
            print(f"[KG] HTTP error page {page_id}: {e}")
        except requests.RequestException as e:
            print(f"[KG] Network error page {page_id}: {e}")
        except Exception as e:
            print(f"[KG] Error page {page_id}: {e}")

    print(f"[KG] Triples extracted: {len(all_triples)}")

    if args.kg_out:
        with open(args.kg_out, "w", encoding="utf-8") as fkg:
            for t in all_triples:
                fkg.write(json.dumps(t, ensure_ascii=False) + "\n")
        print(f"[KG] Saved triples → {args.kg_out}")

    if not all_triples:
        print("[KG] No triples extracted. Nothing to generate.")
        return

    # STEP 2: Entitäten & (optional) Facetten
    ent_index = index_triples_by_entity(all_triples)
    entities = top_entities(ent_index, args.top_entities)
    print(f"[KG] Entities considered: {len(entities)} (top={args.top_entities})")

    # STEP 3: IO-Generierung
    written = 0
    cap_sources = args.max_sources_per_output if args.max_sources_per_output > 0 else None

    with open(args.out, "w", encoding="utf-8") as fout:
        for ent in entities:
            subgraph = ent_index.get(ent, [])
            if len(subgraph) < args.min_triples_per_entity:
                continue

            if args.facet_split:
                facets = split_into_facets(subgraph, min_triples_per_facet=args.min_triples_per_facet)
            else:
                facets = {"misc": subgraph}

            facet_items = sorted(facets.items(), key=lambda kv: len(kv[1]), reverse=True)
            num_facets_available = len(facet_items)
            total_triples = len(subgraph)

            if args.outputs_per_entity > 0:
                outputs_k = min(args.outputs_per_entity, num_facets_available)
            else:
                outputs_k = decide_outputs_per_entity_auto(
                    total_triples_for_entity=total_triples,
                    num_facets_after_filter=num_facets_available,
                    target_triples=DEFAULT_AUTO_OUTPUTS_TARGET_TRIPLES,
                )
            facet_items = facet_items[:outputs_k]

            for facet_name, facet_triples in facet_items:
                try:
                    if args.variants_per_output > 0:
                        n_variants = args.variants_per_output
                    else:
                        n_variants = decide_variants_per_output_auto(len(facet_triples))

                    rows = generate_variants_for_facet(
                        endpoint=args.endpoint, model=args.model,
                        entity=ent, facet_name=facet_name, facet_triples=facet_triples,
                        n_variants=n_variants,
                        temperature=args.generator_temperature,
                        max_tokens=args.generator_max_tokens,
                        timeout=args.timeout, retries=args.retries,
                        max_sources_per_output=cap_sources,
                    )

                    if args.use_detector and rows:
                        has_conflict, notes = detect_conflict(
                            endpoint=args.endpoint, model=args.model,
                            facet_triples=facet_triples, output_text=rows[0]["output"],
                            temperature=args.detector_temperature,
                            max_tokens=args.detector_max_tokens,
                            timeout=args.timeout, retries=args.retries,
                        )
                        if has_conflict:
                            for r in rows:
                                r["insufficient_evidence"] = True
                                if notes:
                                    r["conflict_notes"] = notes

                    for r in rows:
                        if isinstance(r.get("output"), str) and "Sources:" not in r["output"]:
                            src_lines = "\n".join(
                                f"- [{c.get('title','Source')}]({c.get('url','')})"
                                for c in r.get("citations", []) if c.get("url")
                            )
                            if src_lines:
                                r["output"] = r["output"].rstrip() + "\n\nSources:\n" + src_lines

                        # Meta
                        r["entity"] = ent
                        r["facet"] = facet_name
                        r["auto_or_fixed_outputs"] = ("fixed" if args.outputs_per_entity > 0 else "auto")
                        r["auto_or_fixed_variants"] = ("fixed" if args.variants_per_output > 0 else "auto")
                        r["auto_variants_count"] = n_variants
                        r["outputs_k_for_entity"] = outputs_k
                        r["triples_in_facet"] = len(facet_triples)
                        r["triples_in_entity"] = total_triples

                        fout.write(json.dumps(r, ensure_ascii=False) + "\n")
                        written += 1

                        if args.max_total_pairs and written >= args.max_total_pairs:
                            break

                except requests.HTTPError as e:
                    print(f"[GEN] HTTP error entity '{ent}' facet '{facet_name}': {e}")
                except requests.RequestException as e:
                    print(f"[GEN] Network error entity '{ent}' facet '{facet_name}': {e}")
                except Exception as e:
                    print(f"[GEN] Error entity '{ent}' facet '{facet_name}': {e}")

                if args.max_total_pairs and written >= args.max_total_pairs:
                    break

            if args.max_total_pairs and written >= args.max_total_pairs:
                break

    print(f"[IO] Done. Pairs written: {written} → {args.out}")

if __name__ == "__main__":
    main()
