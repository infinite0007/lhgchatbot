#!/usr/bin/env python
# -*- coding: utf-8 -*-
# run with: streamlit run .\cag_chat.py

import os, json
from pathlib import Path
from typing import List, Tuple, Any

import numpy as np
import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_community.llms import HuggingFacePipeline
from sentence_transformers import SentenceTransformer

# =========================
# KONFIG – hier anpassen
# =========================
CACHE_DIR        = "cag_cache"                  # Output von deinem CAG-Indexer (embeddings.npy, texts.jsonl, meta.jsonl)
EMBEDDING_MODEL  = "sentence-transformers/all-MiniLM-L6-v2"
HF_MODEL_PATH    = "../gemma-3-4b-Instruct"     # lokales Instruct-Modell

TOP_K        = 6
TEMPERATURE  = 0.1
MAX_TOKENS   = 300
SNIPPET_CHARS = 900  # max. Zeichen pro Treffer im Prompt

SYSTEM_PROMPT = (
    "You are a precise assistant for enterprise knowledge.\n"
    "Answer ONLY using the provided context. If the answer is not in the context, reply: 'Not in context.'\n"
    "Keep answers concise. Cite sources as URLs at the end if available.\n"
    "Do NOT ask or continue with new questions. Stop after giving the answer."
)

# =========================
# UI – identisch zum RAG-Chat
# =========================
st.set_page_config(page_title="Liebherr Chatbot", page_icon="🦚", layout="wide")
st.title("🦚 CAG Chat – Liebherr Software")
st.caption("Cache-Augmented Generation (CAG) with a local embedding cache. It answers from cached embeddings of the [Confluence Software](https://helpd-doc.liebherr.com/spaces/SWEIG/pages/43424891/SW-Platform-Development+Home+E2020) space — by [Julian Lingnau](https://de.linkedin.com/in/julian-lingnau-05b623162).")

with st.sidebar:
    st.subheader("Settings")
    # gleiche Felder wie im RAG-Chat:
    st.text_input("DB path", value=CACHE_DIR, disabled=True)
    st.text_input("Collection", value="-", disabled=True)
    st.text_input("Embeddings", value=EMBEDDING_MODEL, disabled=True)
    st.text_input("HF model path", value=HF_MODEL_PATH, disabled=True)
    k          = st.slider("Top-K", 1, 20, TOP_K, 1)
    temp       = st.slider("Temperature", 0.0, 1.5, TEMPERATURE, 0.05)
    max_tokens = st.slider("Max new tokens", 64, 1024, MAX_TOKENS, 32)

# =========================
# Cache laden (einheitlich)
# =========================
@st.cache_resource(show_spinner=True)
def load_cag_cache(cache_dir: str):
    cdir = Path(cache_dir)
    # Embeddings speichere großfreundlich mit mmap
    emb = np.load(cdir / "embeddings.npy", mmap_mode="r")  # shape: (N, D)
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
    # Vorabnormierung der Embeddings (Cosine = Dot bei L2=1)
    # mmap ist read-only; daher kopieren wir in float32 und normalisieren einmalig
    emb_arr = np.array(emb, dtype=np.float32, copy=True)
    norms = np.linalg.norm(emb_arr, axis=1, keepdims=True) + 1e-12
    emb_arr /= norms
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
def load_llm(model_path: str, max_new_tokens: int, temperature: float) -> HuggingFacePipeline:
    tok = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    mdl = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype="auto",
    )
    gen = pipeline(
        "text-generation",
        model=mdl,
        tokenizer=tok,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=bool(temperature > 0.0),
        return_full_text=False, # False => gibt nur Antwort, nicht Prompt/Quellen zusätzlich - bei True sieht man auch woher er die Infos im Text zusammengestellt hat
        pad_token_id=tok.eos_token_id if tok.eos_token_id is not None else None,
    )
    return HuggingFacePipeline(pipeline=gen)

llm = load_llm(HF_MODEL_PATH, max_tokens, temp)

# =========================
# CAG: Cosine-Search + Prompting
# =========================
def encode_query(q: str) -> np.ndarray:
    qv = query_encoder.encode([q], convert_to_numpy=True, normalize_embeddings=True)  # (1, D), L2-normalized
    return qv[0]

def topk_indices_cosine(qvec: np.ndarray, top_k: int) -> List[int]:
    # EMB ist L2-normalisiert, qvec auch -> Cosine = Dot
    scores = EMB @ qvec  # shape: (N,)
    top_k = min(top_k, scores.shape[0])
    if top_k <= 0:
        return []
    idx = np.argpartition(-scores, top_k - 1)[:top_k]
    # sortiert nach Score absteigend
    idx = idx[np.argsort(-scores[idx])]
    return idx.tolist()

def format_context(idxs: List[int]) -> Tuple[str, List[Tuple[str, str]]]:
    lines, sources = [], []
    seen = set()
    for i in idxs:
        meta = METAS[i] if i < len(METAS) else {}
        title = meta.get("title") or "Source"
        url   = meta.get("url") or meta.get("source") or ""
        snippet = (TEXTS[i] if i < len(TEXTS) else "").strip()
        if SNIPPET_CHARS and len(snippet) > SNIPPET_CHARS:
            snippet = snippet[:SNIPPET_CHARS] + "…"
        lines.append(f"### {title} — {url}\n{snippet}\n")
        key = (title, url)
        if key not in seen and (title or url):
            seen.add(key)
            sources.append(key)
    return "\n".join(lines), sources

def answer_with_sources(question: str, top_k: int) -> Tuple[str, List[Tuple[str, str]]]:
    qv = encode_query(question)
    idxs = topk_indices_cosine(qv, top_k)
    context_text, sources = format_context(idxs)
    prompt = f"{SYSTEM_PROMPT}\n\nContext:\n{context_text}\n\nQuestion: {question}\n\nAnswer:"
    out = llm.invoke(prompt)
    text = str(out)
    if sources:
        src_lines = []
        for title, url in sources:
            src_lines.append(f"- {title}: {url}" if url else f"- {title}")
        text += "\n\nSources:\n" + "\n".join(src_lines)
    return text, sources

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
                ans, _ = answer_with_sources(user_q, k)
            except Exception as e:
                st.error(f"Fehler: {e}")
                st.stop()
            st.write(ans)
    st.session_state.messages.append({"role": "assistant", "content": ans})
