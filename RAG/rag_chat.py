#!/usr/bin/env python
# -*- coding: utf-8 -*-
# run with: streamlit run .\rag_chat.py
import os
from typing import List, Any, Tuple

import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain_core.runnables import RunnablePassthrough

# =========================
# KONFIG – hier anpassen
# =========================
DATABASE_LOCATION = "chroma_db"
COLLECTION_NAME   = "rag_data"
EMBEDDING_MODEL   = "sentence-transformers/all-MiniLM-L6-v2"
HF_MODEL_PATH     = "../Falcon3-1B-Instruct"  # dein lokales Modell

TOP_K        = 6
USE_MMR      = False
TEMPERATURE  = 0.1
MAX_TOKENS   = 300

SYSTEM_PROMPT = (
    "You are a precise assistant for enterprise knowledge.\n"
    "Answer ONLY using the provided context. If the answer is not in the context, reply: 'Not in context.'\n"
    "Keep answers concise. Cite sources as URLs at the end if available."
)

# =========================
# UI früh rendern
# =========================
st.set_page_config(page_title="Liebherr Chatbot", page_icon="🦜", layout="wide")
st.title("🦜 RAG Chat – Liebherr Software")
st.caption("Has all infos under Confluence Software: https://helpd-doc.liebherr.com/spaces/SWEIG/pages/43424891/SW-Platform-Development+Home+E2020 by Julian Lingnau")

with st.sidebar:
    st.subheader("Settings")
    st.text_input("DB path", value=DATABASE_LOCATION, disabled=True)
    st.text_input("Collection", value=COLLECTION_NAME, disabled=True)
    st.text_input("Embeddings", value=EMBEDDING_MODEL, disabled=True)
    st.text_input("HF model path", value=HF_MODEL_PATH, disabled=True)
    k = st.slider("Top-K", 1, 20, TOP_K, 1)
    temp = st.slider("Temperature", 0.0, 1.5, TEMPERATURE, 0.05)
    max_tokens = st.slider("Max new tokens", 64, 1024, MAX_TOKENS, 32)

# =========================
# Vectorstore + Embeddings
# =========================
@st.cache_resource(show_spinner=True)
def load_vectorstore(db_dir: str, coll: str, emb_model: str):
    embeddings = HuggingFaceEmbeddings(model_name=emb_model)
    vs = Chroma(collection_name=coll, embedding_function=embeddings, persist_directory=db_dir)
    return vs

vector_store = load_vectorstore(DATABASE_LOCATION, COLLECTION_NAME, EMBEDDING_MODEL)

# Basics anzeigen / leer?
try:
    info = vector_store.get()
    num_vecs = len(info.get("ids", []))
except Exception as e:
    st.error(f"Chroma konnte nicht gelesen werden: {e}")
    st.stop()

st.info(f"📦 Vektoren in DB: **{num_vecs}**")
if num_vecs == 0:
    st.error("Die Chroma-DB ist leer. Bitte zuerst indexieren (rag_indexdb.py).")
    st.stop()

# Retriever
def build_retriever(vs: Chroma, top_k: int, use_mmr: bool):
    if not use_mmr:
        return vs.as_retriever(search_kwargs={"k": top_k})
    return vs.as_retriever(search_type="mmr", search_kwargs={"k": top_k, "fetch_k": max(top_k * 3, 20)})

retriever = build_retriever(vector_store, k, USE_MMR)

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
        pad_token_id=tok.eos_token_id if tok.eos_token_id is not None else None,
    )
    return HuggingFacePipeline(pipeline=gen)

llm = load_llm(HF_MODEL_PATH, max_tokens, temp)

# =========================
# RAG-Helfer
# =========================
def format_docs(docs: List[Any]) -> Tuple[str, List[Tuple[str, str]]]:
    """Kontext für Prompt + deduplizierte Quellenliste (Titel, URL)."""
    seen = set()
    lines, sources = [], []
    for d in docs:
        title = d.metadata.get("title") or "Source"
        url   = d.metadata.get("url") or d.metadata.get("source") or ""
        key = (title, url)
        snippet = (d.page_content or "").strip()
        snippet = snippet[:900] + ("…" if len(snippet) > 900 else "")
        lines.append(f"### {title} — {url}\n{snippet}\n")
        if key not in seen and (title or url):
            seen.add(key)
            sources.append(key)
    return "\n".join(lines), sources

def answer_with_sources(question: str, top_k: int) -> Tuple[str, List[Tuple[str, str]]]:
    rv = build_retriever(vector_store, top_k, USE_MMR)
    docs = rv.get_relevant_documents(question)
    context_text, sources = format_docs(docs)
    prompt = f"{SYSTEM_PROMPT}\n\nContext:\n{context_text}\n\nQuestion: {question}\n\nAnswer:"
    out = llm.invoke(prompt)
    text = str(out)
    # Quellen anhängen
    if sources:
        src_lines = []
        for title, url in sources:
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
        with st.spinner("Retrieving & generating …"):
            try:
                ans, _ = answer_with_sources(user_q, k)
            except Exception as e:
                st.error(f"Fehler: {e}")
                st.stop()
            st.write(ans)
    st.session_state.messages.append({"role": "assistant", "content": ans})
