#!/usr/bin/env python
# -*- coding: utf-8 -*-
# run with: streamlit run .\cag_chat.py

import os, json, pickle
from typing import List, Any, Tuple
from pathlib import Path

import numpy as np
from scipy import sparse
import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_community.llms import HuggingFacePipeline

# =========================
# KONFIG – hier anpassen
# =========================
CACHE_DIR      = "cag_cache"           # Output von cag_build_cache.py
HF_MODEL_PATH  = "../gemma-3-4b-Instruct"  # dein lokales Instruct-Modell

TOP_K        = 6
TEMPERATURE  = 0.1
MAX_TOKENS   = 300
SNIPPET_CHARS = 900  # pro Treffer

SYSTEM_PROMPT = (
    "You are a precise assistant for enterprise knowledge.\n"
    "Answer ONLY using the provided context. If the answer is not in the context, reply: 'Not in context.'\n"
    "Keep answers concise. Cite sources as URLs at the end if available.\n"
    "Do NOT ask or continue with new questions. Stop after giving the answer."
)

# =========================
# UI
# =========================
st.set_page_config(page_title="Liebherr Chatbot (CAG)", page_icon="🧠", layout="wide")
st.title("🧠 CAG Chat – Liebherr Software")
st.caption("Cache-Augmented Generation auf Basis deines JSONL-Caches (ohne Vektor-DB).")

with st.sidebar:
    st.subheader("Settings")
    st.text_input("Cache dir", value=CACHE_DIR, disabled=True)
    st.text_input("HF model path", value=HF_MODEL_PATH, disabled=True)
    k          = st.slider("Top-K", 1, 20, TOP_K, 1)
    temp       = st.slider("Temperature", 0.0, 1.5, TEMPERATURE, 0.05)
    max_tokens = st.slider("Max new tokens", 64, 1024, MAX_TOKENS, 32)

# =========================
# Cache laden
# =========================
@st.cache_resource(show_spinner=True)
def load_cag_cache(cache_dir: str):
    cache_dir = Path(cache_dir)
    X = sparse.load_npz(cache_dir / "tfidf_matrix.npz")
    with open(cache_dir / "vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    # Metadaten
    metas = []
    with open(cache_dir / "meta.jsonl", "r", encoding="utf-8") as fp:
        for line in fp:
            if line.strip():
                metas.append(json.loads(line))
    return X.tocsr(), vectorizer, metas

try:
    X, VEC, METAS = load_cag_cache(CACHE_DIR)
    num_docs = X.shape[0]
    vocab_sz = len(VEC.vocabulary_)
    st.info(f"📦 Cache geladen: **{num_docs}** Dokumente, **{vocab_sz}** Features")
except Exception as e:
    st.error(f"Cache konnte nicht geladen werden aus '{CACHE_DIR}': {e}")
    st.stop()

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
        return_full_text=False,
        pad_token_id=tok.eos_token_id if tok.eos_token_id is not None else None,
    )
    return HuggingFacePipeline(pipeline=gen)

llm = load_llm(HF_MODEL_PATH, max_tokens, temp)

# =========================
# CAG-Helfer
# =========================
def tfidf_search(query: str, top_k: int) -> List[int]:
    """Liefert Top-K Doc-Indices nach Cosine-Similarity (TF-IDF)."""
    q_vec = VEC.transform([query])            # 1 x vocab
    # Cosine = (q @ X.T) / (||q|| * ||X||)
    # Für Sparse effizient: wir normalisieren VEC standardmäßig L2; scikit tf-idf gibt L2-normierte Zeilen zurück,
    # sodass dot-Produkt ~ Cosine ist. Sonst explizit normieren.
    scores = q_vec @ X.T                      # shape: (1, n_docs)
    scores = scores.toarray().ravel()
    if top_k >= len(scores):
        return np.argsort(-scores).tolist()
    idx = np.argpartition(-scores, top_k)[:top_k]
    idx = idx[np.argsort(-scores[idx])]
    return idx.tolist()

def load_original_texts(doc_indices: List[int], dataset_path: str) -> List[Tuple[str, str, str]]:
    """
    Optional: Originaltext für Kontextblöcke aus JSONL nachladen.
    Rückgabe: [(title, url, snippet), ...]
    """
    # Leichtgewicht: Metas enthalten URL/Titel; Snippet holen wir aus Cache nicht direkt,
    # deshalb (für Genauigkeit) lesen wir betroffene Zeilen erneut aus dem JSONL.
    # Dazu speichern wir im Meta die page_id – hier stark vereinfacht:
    title_url = [(METAS[i].get("title","Source"), METAS[i].get("url","")) for i in doc_indices]
    # In vielen Fällen genügt es, die Snippets aus TF-IDF nicht erneut zu lesen.
    # Wir erzeugen Snippets, indem wir den TF-IDF-Vektor nutzen -> hier einfacher:
    # Für wirklich präzise Snippets kannst du deine Originaltexte in einer separaten Datei zwischenspeichern.
    # Hier nehmen wir eine „Light“-Variante: kein Volltext-Reload, sondern Markierung.
    out = []
    for (t, u) in title_url:
        # Warnhinweis: ohne Volltext-Reload kein exakter Auszug – optional verbessern:
        out.append((t, u, "[Snippet im Cache – Volltext optional nachladen]"))
    return out

def format_context(doc_indices: List[int]) -> Tuple[str, List[Tuple[str, str]]]:
    lines, sources = [], []
    for i in doc_indices:
        title = METAS[i].get("title") or "Source"
        url   = METAS[i].get("url") or ""
        # Optional: exakten Snippet liefern (siehe Kommentar in load_original_texts)
        snippet = f"(CAG Match {i})"
        lines.append(f"### {title} — {url}\n{snippet}\n")
        sources.append((title, url))
    return "\n".join(lines), sources

def answer_with_sources(question: str, top_k: int) -> Tuple[str, List[Tuple[str, str]]]:
    idxs = tfidf_search(question, top_k)
    context_text, sources = format_context(idxs)
    prompt = f"{SYSTEM_PROMPT}\n\nContext:\n{context_text}\n\nQuestion: {question}\n\nAnswer:"
    out = llm.invoke(prompt)
    text = str(out)
    if sources:
        src_lines = []
        seen = set()
        for title, url in sources:
            key = (title, url)
            if key in seen: 
                continue
            seen.add(key)
            if url:
                src_lines.append(f"- {title}: {url}")
            else:
                src_lines.append(f"- {title}")
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
        with st.spinner("Searching cache & generating …"):
            try:
                ans, _ = answer_with_sources(user_q, k)
            except Exception as e:
                st.error(f"Fehler: {e}")
                st.stop()
            st.write(ans)
    st.session_state.messages.append({"role": "assistant", "content": ans})
