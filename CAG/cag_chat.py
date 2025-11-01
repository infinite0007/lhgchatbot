#!/usr/bin/env python
# -*- coding: utf-8 -*-
# run with: streamlit run .\cag_chat.py

import os, json, re
from pathlib import Path
from typing import List, Tuple, Any, Dict, Optional

import numpy as np
import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_community.llms import HuggingFacePipeline
from sentence_transformers import SentenceTransformer

# Optional FAISS (für sehr große Caches und/oder GPU Acceleration)
try:
    import faiss  # pip install faiss-cpu  (oder faiss-gpu)
    HAS_FAISS = True
except Exception:
    HAS_FAISS = False

# Optional PyTorch für CUDA-Matmul (falls keine FAISS-GPU)
try:
    import torch
    HAS_TORCH = True
except Exception:
    HAS_TORCH = False

# =========================
# KONFIG – hier anpassen
# =========================
CACHE_DIR        = "cag_cache"                  # Output von deinem CAG-Indexer (embeddings.npy, texts.jsonl, meta.jsonl)
EMBEDDING_MODEL  = "sentence-transformers/all-MiniLM-L6-v2"
HF_MODEL_PATH    = "../gemma-3-4b-Instruct"     # lokales Instruct-Modell

TOP_K        = 6
TEMPERATURE  = 0.1
MAX_TOKENS   = 450
SNIPPET_CHARS = 900  # max. Zeichen pro Treffer im Prompt
USE_FAISS      = True  # wenn verfügbar, nutze FAISS (CPU/GPU). Fallback: NumPy / Torch

SYSTEM_PROMPT = (
    "You are a precise assistant for enterprise knowledge.\n"
    "Follow these steps strictly and in order:\n"
    "1. Before searching, check the question for spelling or typing errors and silently correct them.\n"
    "2. Answer ONLY using the provided context.\n"
    "3. If no relevant information can be found, reply only with: 'Not in context.' and do not cite any sources.\n"
    "4. If the question can be partially answered, provide only the part that is supported by the context.\n"
    "5. Keep answers concise and factual.\n"
    "6. Add cites into the text but only if it's available in the provided context. Use [1], [2], ... markers.\n"
    "7. Do NOT invent sources or add information not in the context.\n"
    "8. Do NOT ask follow-up questions. Stop after giving the answer and sources.\n"
    "This is important — follow the steps exactly and do not hallucinate."
)

# =========================
# UI – identisch zur RAG-Optik
# =========================
st.set_page_config(page_title="Liebherr Chatbot", page_icon="🦚", layout="wide")
st.title("🦚 CAG Chat – Liebherr Software")
st.caption("Cache-Augmented Generation (CAG) with a local embedding cache. It answers from cached embeddings of the [Confluence Software](https://helpd-doc.liebherr.com/spaces/SWEIG/pages/43424891/SW-Platform-Development+Home+E2020) space — by [Julian Lingnau](https://de.linkedin.com/in/julian-lingnau-05b623162).")

with st.sidebar:
    st.subheader("Settings")
    st.text_input("DB path", value=CACHE_DIR, disabled=True)
    st.text_input("Collection", value="-", disabled=True)
    st.text_input("Embeddings", value=EMBEDDING_MODEL, disabled=True)
    st.text_input("HF model path", value=HF_MODEL_PATH, disabled=True)
    k          = st.slider("Top-K", 1, 20, TOP_K, 1)
    temp       = st.slider("Temperature", 0.0, 1.5, TEMPERATURE, 0.05)
    max_tokens = st.slider("Max new tokens", 64, 1024, MAX_TOKENS, 32)
    # Neu: identische Toggles wie im RAG-Chat (nur hinzugefügt; sonst nichts verändert)
    follow_up_active = st.toggle("Follow-up question", value=True)
    debug_mode       = st.toggle("Further debugging", value=False)

# =========================
# Cache laden (schnell & speicherschonend)
# =========================
@st.cache_resource(show_spinner=True)
def load_cag_cache(cache_dir: str):
    cdir = Path(cache_dir)
    emb = np.load(cdir / "embeddings.npy", mmap_mode="r")  # (N, D), angenommen bereits L2-normalized gespeichert
    # texts.jsonl: pro Zeile ein Chunk-Text
    texts: List[str] = []
    with open(cdir / "texts.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                obj = json.loads(line)
                texts.append(obj["text"])
    # meta.jsonl: parallele Metadaten-Liste
    metas: List[dict] = []
    with open(cdir / "meta.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                metas.append(json.loads(line))
    # config.json (optional)
    cfg = {}
    cfg_path = cdir / "config.json"
    if cfg_path.exists():
        try:
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        except Exception:
            cfg = {}

    # np.memmap → float32 Array in RAM (einmalig). embeddings.npy ist bereits normalized. # Speicherschonend (echtes memmap) – auskommentierte Variante Array bleibt gemappt. RAM-Peak klein, aber Zugriff ist langsamer (Page-Faults, I/O). Sinnvoll, wenn das Embedding-File größer als euer RAM ist. Aber das wird aktuell nicht nötig sein.
    emb_arr = np.array(emb, dtype=np.float32, copy=True)
    return emb_arr, texts, metas, cfg

try:
    EMB, TEXTS, METAS, CFG = load_cag_cache(CACHE_DIR)
    n_chunks, dim = EMB.shape
    st.info(f"📦 Cache geladen: **{n_chunks}** Chunks, **{dim}**-D Embeddings")
except Exception as e:
    st.error(f"Cache konnte nicht geladen werden aus '{CACHE_DIR}': {e}")
    st.stop()

# =========================
# Query-Encoder (gleiches Modell)
# =========================
@st.cache_resource(show_spinner=True)
def load_query_encoder(model_name: str) -> SentenceTransformer:
    return SentenceTransformer(model_name)

query_encoder = load_query_encoder(EMBEDDING_MODEL)

# =========================
# LLM lazy & gecacht laden
# =========================
@st.cache_resource(show_spinner=True)
def load_llm(model_path: str, max_new_tokens: int, temperature: float, debug: bool) -> HuggingFacePipeline:
    tok = AutoTokenizer.from_pretrained(model_path, use_fast=True, local_files_only=True)
    mdl = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype="auto",
        local_files_only=True,
    )
    gen = pipeline(
        "text-generation",
        model=mdl,
        tokenizer=tok,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=bool(temperature > 0.0),
        return_full_text=True,
        pad_token_id=tok.eos_token_id if tok.eos_token_id is not None else None,
    )
    return HuggingFacePipeline(pipeline=gen)

llm = load_llm(HF_MODEL_PATH, max_tokens, temp, debug_mode)

# =========================
# Retrieval (Cosine) – mit optionaler FAISS / CUDA-Beschleunigung
# =========================
def _faiss_build_index(emb: np.ndarray):
    # Cosine via inner product bei l2-normalized Vektoren
    index = faiss.IndexFlatIP(emb.shape[1])
    if faiss.get_num_gpus() > 0:
        _ = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_all_gpus(index)  # verteilt auf alle GPUs
    index.add(emb.astype(np.float32))
    return index

@st.cache_resource(show_spinner=False)
def get_search_backend(emb: np.ndarray):
    """
    Liefert einen callable search(qvec, top_k) -> (scores, idx) mit dem schnellsten verfügbaren Backend.
    Reihenfolge: FAISS (GPU/CPU) > Torch CUDA > NumPy.
    """
    if USE_FAISS and HAS_FAISS:
        try:
            index = _faiss_build_index(emb)
            def _search_faiss(qv: np.ndarray, top_k: int):
                D, I = index.search(qv.reshape(1, -1).astype(np.float32), top_k)
                return D[0].tolist(), I[0].tolist()
            return _search_faiss, "FAISS"
        except Exception:
            pass

    if HAS_TORCH and torch.cuda.is_available():
        t_emb = torch.from_numpy(emb).to("cuda")
        def _search_torch(qv: np.ndarray, top_k: int):
            t_q = torch.from_numpy(qv.reshape(1, -1)).to("cuda")
            scores = torch.matmul(t_emb, t_q.t()).squeeze(1)  # (N,)
            topk = min(top_k, scores.shape[0])
            vals, idx = torch.topk(scores, k=topk, largest=True, sorted=True)
            return vals.detach().cpu().numpy().tolist(), idx.detach().cpu().numpy().tolist()
        return _search_torch, "TorchCUDA"

    # Fallback: NumPy (schnell genug für mittlere Größen)
    def _search_numpy(qv: np.ndarray, top_k: int):
        s = EMB @ qv  # (N,), Cosine = Dot (l2-normalized)
        topk = min(top_k, s.shape[0])
        idx = np.argpartition(-s, topk - 1)[:topk]
        idx = idx[np.argsort(-s[idx])]
        return s[idx].tolist(), idx.tolist()
    return _search_numpy, "NumPy"

SEARCH, BACKEND = get_search_backend(EMB)
st.caption(f"Retrieval Backend: **{BACKEND}**")

# =========================
# Hilfsfunktionen: Zitate & Anzeige wie im RAG
# =========================
RE_CITATIONS = re.compile(r"\[(\d+)\]")

def canonicalize_url(u: str) -> str:
    if not u:
        return u
    u = u.strip().split("#")[0]
    return u[:-1] if u.endswith("/") else u

def extract_used_indices(text: str) -> set:
    return set(int(m) for m in RE_CITATIONS.findall(text))

def remove_invalid_citations(text: str, invalid: set[int]) -> str:
    for i in sorted(invalid, reverse=True):
        text = re.sub(rf"(?<!\d)\[{i}\](?!\d)", "", text)
    return re.sub(r"\s{2,}", " ", text).strip()

def _score_key(title: str, url: str) -> Tuple[str, str]:
    return (title or "").strip(), canonicalize_url(url or "")

def format_context(idxs: List[int], scores: List[float]) -> Tuple[str, List[Tuple[int, str, str, float]], List[Tuple[str, str, float]]]:
    """
    Returns:
      - context_text: nummerierte Einträge [1] ... [n] (inkl. SCORE)
      - index_map: [(index, title, url, score)]
      - topk_list: deduped [(title, url, score)] für Top-K-Anzeige
    """
    seen = set()
    lines = []
    index_map = []
    topk_list = []
    for rank, (i, sc) in enumerate(zip(idxs, scores), start=1):
        meta = METAS[i] if i < len(METAS) else {}
        title = meta.get("title") or f"Source {rank}"
        url   = canonicalize_url(meta.get("url") or meta.get("source") or "")
        snippet = (TEXTS[i] if i < len(TEXTS) else "").strip()
        if SNIPPET_CHARS and len(snippet) > SNIPPET_CHARS:
            snippet = snippet[:SNIPPET_CHARS] + "…"
        lines.append(f"[{rank}] {title} — {url} (SCORE={float(sc):.4f})\n{snippet}\n")
        index_map.append((rank, title, url, float(sc)))
        key = (title, url)
        if key not in seen and (title or url):
            seen.add(key)
            topk_list.append((title, url, float(sc)))
    return "\n".join(lines), index_map, topk_list

def extract_answer_block(full_text: str) -> str:
    m = re.search(r"Answer:\s*(.*)", full_text, re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else full_text.strip()

def build_sources_section(valid_indices: List[int], index_map: List[Tuple[int, str, str, float]]) -> str:
    if not valid_indices:
        return "Sources used:\n- (no valid cited sources found in retrieved context)"
    idx_to_meta = {i: (t, u) for i, t, u, _ in index_map}
    used_sources = []
    seen_urls = set()
    for i in valid_indices:
        t, u = idx_to_meta[i]
        u_c = canonicalize_url(u)
        if u_c not in seen_urls:
            seen_urls.add(u_c)
            used_sources.append(f"- [{i}] {t}: {u}")
    return "Sources used:\n\n" + "\n".join(used_sources)

def build_topk_section(topk_list: List[Tuple[str, str, float]]) -> str:
    if not topk_list:
        return ""
    lines = []
    for title, url, score in topk_list:
        lines.append(f"- {title}: {url} (SCORE={score:.4f})")
    return "Top-K searched in:\n\n" + "\n".join(lines)

# ========= History-aware Query-Rewriting (nur wenn Follow-up aktiv) =========
def _history_window(messages: List[Dict[str, str]], max_pairs: int = 3) -> List[Dict[str, str]]:
    """Nimmt die letzten max_pairs User/Assistant-Paare (ohne den aktuellen Input)."""
    hist = []
    for m in messages:
        if m.get("role") in ("user", "assistant"):
            hist.append({"role": m["role"], "content": m["content"]})
    return hist[-(max_pairs*2):] if hist else []

def _rewrite_with_history(original_q: str, history_snippets: List[Dict[str, str]]) -> str:
    """
    Nutzt dasselbe LLM deterministisch (temp=0.0), um Folgefragen zu disambiguieren.
    Gibt bei Fehlschlag die Originalfrage zurück.
    """
    if not history_snippets:
        return original_q
    try:
        hist_txt = "\n".join(
            [f"{h['role'].capitalize()}: {h['content']}" for h in history_snippets[-6:]]
        )
        prompt = (
            "Rewrite the user's question so it is fully self-contained and unambiguous, "
            "based only on the chat history. Do NOT answer the question. Output only the rewritten question.\n\n"
            f"Chat history:\n{hist_txt}\n\n"
            f"User question: {original_q}\n\nRewritten:"
        )
        gen = llm.pipeline
        resp = gen(prompt, max_new_tokens=96, temperature=0.0, do_sample=False, return_full_text=False)
        text = str(resp[0]["generated_text"]).strip()
        text = " ".join(text.split()).strip().strip('"').strip("'")
        if len(text) < 3:
            return original_q
        return text
    except Exception:
        return original_q

# =========================
# CAG QA-Pipeline
# =========================
def encode_query(q: str) -> np.ndarray:
    qv = query_encoder.encode([q], convert_to_numpy=True, normalize_embeddings=True)  # (1, D)
    return qv[0].astype(np.float32)

def answer_with_sources(question: str, top_k: int, debug: bool, follow_up: bool) -> Tuple[str, List[Tuple[int, str, str, float]]]:
    # (A) optional: History-aware Rewrite
    history_used = _history_window(st.session_state.get("messages", []), max_pairs=3)
    original_q = question
    effective_q = question
    if follow_up:
        effective_q = _rewrite_with_history(original_q, history_used)

    # (B) Retrieval
    qv = encode_query(effective_q)
    scores, idxs = SEARCH(qv, top_k=max(top_k, 6))  # klein wenig breiter holen

    # (C) Kontext formatieren (inkl. SCORE)
    context_text, index_map, topk_list = format_context(idxs[:top_k], scores[:top_k])

    # (D) Prompt + Generate
    prompt = f"{SYSTEM_PROMPT}\n\nContext:\n{context_text}\n\nQuestion: {effective_q}\n\nAnswer:"
    out = llm.invoke(prompt)
    full_text = str(out)

    # (E) Nur den eigentlichen Answer-Block isolieren
    answer = extract_answer_block(full_text)

    # (F) Zitate aus Answer-Block extrahieren + validieren
    used_indices = extract_used_indices(answer)
    available_indices = {i for i, _, _, _ in index_map}
    invalid_indices = used_indices - available_indices
    if invalid_indices:
        answer = remove_invalid_citations(answer, invalid_indices)
        used_indices = extract_used_indices(answer)
    valid_indices = sorted(list(used_indices & available_indices))

    # (G) Fallback: Kontext vorhanden aber keine Zitate gesetzt → [1] an 1. Satz
    if not valid_indices and index_map and answer.strip().lower() != "not in context.":
        answer = re.sub(r"([.!?])(\s|$)", r" [1]\1\2", answer, count=1)
        valid_indices = [1]

    # (H) Sections bauen
    sources_section = build_sources_section(valid_indices, index_map)

    debug_sections = ""
    if debug:
        topk_section = build_topk_section(topk_list)
        full_context_dump = f"— Full Context (as given to LLM) —\n{context_text}"
        hist_lines = []
        if history_used:
            for h in history_used:
                role = "User" if h["role"] == "user" else "Assistant"
                content = h["content"].strip()
                if len(content) > 500:
                    content = content[:500] + "…"
                hist_lines.append(f"{role}: {content}")
        history_dump = "\n".join(hist_lines) if hist_lines else "(none)"
        rewrite_dump = (
            "— Follow-up rewrite applied: YES —\n"
            f"— Original question —\n{original_q}\n\n"
            f"— Effective question —\n{effective_q}\n\n"
            f"— History used (window=3) —\n{history_dump}"
            if follow_up else
            "— Follow-up rewrite applied: NO —"
        )
        debug_sections = "\n\n" + (topk_section + "\n\n" if topk_section else "") + rewrite_dump + "\n\n" + full_context_dump

    # (I) Finale Antwort
    final_text = f"{answer}\n\n{sources_section}{debug_sections}"
    return final_text, index_map

# =========================
# Chat-Verlauf & UI
# =========================
if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    role = "user" if m["role"] == "user" else "assistant"
    with st.chat_message(role):
        st.write(m["content"])

user_q = st.chat_input("Frage zu deinen Daten …")
if user_q:
    st.session_state.messages.append({"role": "user", "content": user_q})
    with st.chat_message("user"):
        st.write(user_q)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving (CAG) & generating …"):
            try:
                ans, _ = answer_with_sources(user_q, k, debug_mode, follow_up_active)
            except Exception as e:
                st.error(f"Fehler: {e}")
                st.stop()
            st.write(ans)
    st.session_state.messages.append({"role": "assistant", "content": ans})
