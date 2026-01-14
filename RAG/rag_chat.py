#!/usr/bin/env python
# -*- coding: utf-8 -*-
# run with: streamlit run .\rag_chat.py

import os, re, json
from typing import List, Any, Tuple, Dict, Optional

import numpy as np
import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline, LlamaCpp
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer

# =========================
# CONFIG
# =========================
DATABASE_LOCATION = "chroma_db"
COLLECTION_NAME   = "rag_data"
EMBEDDING_MODEL   = "sentence-transformers/all-MiniLM-L6-v2"
HF_MODEL_PATH     = "../Qwen2.5-1.5B-Instruct"

TOP_K         = 6
USE_MMR       = False # https://www.evidentlyai.com/ranking-metrics/mean-reciprocal-rank-mrr
TEMPERATURE   = 0.0
MAX_TOKENS    = 450
SNIPPET_CHARS = 900

# ---- Quantifizierte Einstellungen GGUF ----
USE_GGUF        =  # True = ungenauer und Antwortqualität leidet under GGUF
GGUF_MODEL_PATH = "../FinetuneLLM/finetunedmodels/QWENTest/Qwen2.5-1.5B-Instruct-lora-unsloth-liebherr-2ep_var5-out/gguf/Qwen2.5-1.5B-Instruct-lora-unsloth-liebherr-2ep_var5.gguf"  # Quantifiziertes Modell
N_CTX           = 8192   # je nach Bedarf/VRAM
N_GPU_LAYERS    = -1     # -1 = max. Offload auf CUDA (GPU) wenn man es anpasst, verteilt es sich dann perfekt auf CPU/GPU auf
N_THREADS       = os.cpu_count() or 8

SYSTEM_PROMPT = (
    "You are a precise assistant for enterprise knowledge.\n"
    "Follow these steps strictly and in order:\n"
    "1. Before searching, check the question for spelling or typing errors and silently correct them.\n"
    "2. Search only within the provided context to find an answer.\n"
    "3. If no relevant information can be found, reply only with: 'Not in context.' and do not cite any sources.\n"
    "4. If the question can be partially answered, provide only the part that is supported by the context.\n"
    "5. Keep answers concise and factual.\n"
    "6. Add source citations as [1], [2], etc. into the text (if supported by the context).\n"
    "7. Do NOT write any 'Sources used:' or similar headings like 'Sources used: [1]'. Citations must only appear inline like [1], [2], etc.\n"
    "8. Do NOT invent sources or add information not in the context.\n"
    "9. Do NOT ask follow-up questions. Stop after giving the answer and sources."
)

# =========================
# UI
# =========================
st.set_page_config(page_title="Liebherr Chatbot", page_icon="🦜", layout="wide")
st.title("🦜 RAG Chat – Liebherr Software")
st.caption("Retrieval-Augmented Generation (RAG) with a vector database. It retrieves enterprise knowledge from the [Confluence Software](https://helpd-doc.liebherr.com/spaces/SWEIG/pages/43424891/SW-Platform-Development+Home+E2020) space — by [Julian Lingnau](https://de.linkedin.com/in/julian-lingnau-05b623162).")

with st.sidebar:
    st.subheader("Settings")
    st.text_input("DB path", value=DATABASE_LOCATION, disabled=True)
    st.text_input("Collection", value=COLLECTION_NAME, disabled=True)
    st.text_input("Embeddings", value=EMBEDDING_MODEL, disabled=True)
    # Modellanzeige dynamisch je nach Modus
    model_name_display = (
        os.path.basename(GGUF_MODEL_PATH) if USE_GGUF else os.path.basename(HF_MODEL_PATH)
    )
    st.text_input("Model", value=model_name_display, disabled=True)

    k          = st.slider("Top-K", 1, 20, TOP_K, 1)
    temp       = st.slider("Temperature", 0.0, 1.5, TEMPERATURE, 0.05)
    max_tokens = st.slider("Max new tokens", 64, 1024, MAX_TOKENS, 32)
    spellfix_active = st.toggle("Spell checker", value=True)
    debug_mode      = st.toggle("Further debugging", value=False)
    st.markdown("---")
    follow_up_active = st.toggle("Follow-up questions", value=True)
    carry_docs_k     = st.slider("Carryover (docs)", 0, 10, 3, 1)
    history_turns    = st.slider("History (turns)", 0, 10, 3, 1)

# =========================
# Vectorstore + Embeddings
# =========================
@st.cache_resource(show_spinner=True)
def load_vectorstore(db_dir: str, coll: str, emb_model: str):
    embeddings = HuggingFaceEmbeddings(model_name=emb_model)
    vs = Chroma(collection_name=coll, embedding_function=embeddings, persist_directory=db_dir)
    return vs

vector_store = load_vectorstore(DATABASE_LOCATION, COLLECTION_NAME, EMBEDDING_MODEL)

try:
    info = vector_store.get()
    num_vecs = len(info.get("ids", []))
except Exception as e:
    st.error(f"Chroma error: {e}")
    st.stop()

st.info(f"📦 Vectors in DB: **{num_vecs}**")
if num_vecs == 0:
    st.error("Chroma DB is empty. Please index first (rag_indexdb.py).")
    st.stop()

mode = "MMR" if USE_MMR else "Top-K cosine"
st.caption(f"Vector DB: **Chroma** • Retrieval: **{mode}** • Top-K: **{k}**")

def build_retriever(vs: Chroma, top_k: int, use_mmr: bool):
    if not use_mmr:
        return vs.as_retriever(search_kwargs={"k": top_k})
    return vs.as_retriever(search_type="mmr", search_kwargs={"k": top_k, "fetch_k": max(top_k * 3, 20)})

# =========================
# LLM (cached)
# =========================
@st.cache_resource(show_spinner=True)
def load_llm(use_gguf: bool, hf_model_path: str, gguf_model_path: str,
             max_new_tokens: int, temperature: float):
    if not use_gguf:
        tok = AutoTokenizer.from_pretrained(hf_model_path, use_fast=True, local_files_only=True)
        mdl = AutoModelForCausalLM.from_pretrained(
            hf_model_path, device_map="auto", torch_dtype="auto", local_files_only=True
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

    # GGUF-Pfad wählen -> LlamaCpp (CUDA-Offload via n_gpu_layers)
    llm = LlamaCpp(
        model_path=gguf_model_path,
        n_ctx=N_CTX, # In llama.cpp muss Prompt-Tokens + max_tokens ≤ n_ctx sein, sonst ERROR also abgelehnt wenn: prompt_tokens + max_tokens > n_ctx
        verbose=True,
        use_mlock=True,
        n_gpu_layers=N_GPU_LAYERS,
        n_threads=N_THREADS
    )
    return llm

llm = load_llm(USE_GGUF, HF_MODEL_PATH, GGUF_MODEL_PATH, MAX_TOKENS, TEMPERATURE)

# =========================
# Optional: Query encoder
# =========================
@st.cache_resource(show_spinner=False)
def load_query_encoder(name: str) -> SentenceTransformer:
    return SentenceTransformer(name)

query_encoder = load_query_encoder(EMBEDDING_MODEL)

# =========================
# Helpers
# =========================
RE_CITATIONS = re.compile(r"\[(\d+)\]")
SOURCES_RE = re.compile(r"(Sources\s+(used:|:))(.|\s)*?$", re.IGNORECASE)

def strip_llm_sources_block(text: str) -> str:
    """Schneidet vom LLM generierte 'Sources used:'-Abschnitte am Ende weg."""
    return re.sub(SOURCES_RE, "", text).rstrip()

def extract_used_indices(text: str) -> set:
    return set(int(m) for m in RE_CITATIONS.findall(text))

def remove_invalid_citations(text: str, invalid: set[int]) -> str:
    for i in sorted(invalid, reverse=True):
        text = re.sub(rf"(?<!\d)\[{i}\](?!\d)", "", text)
    return re.sub(r"\s{2,}", " ", text).strip()

def canonicalize_url(u: str) -> str:
    if not u:
        return u
    u = u.strip().split("#")[0]
    return u[:-1] if u.endswith("/") else u

def _score_key(title: str, url: str) -> Tuple[str, str]:
    return (title or "").strip(), canonicalize_url(url or "")

def format_docs(
    docs: List[Document],
    score_by_key: Optional[Dict[Tuple[str, str], float]] = None
) -> Tuple[str, List[Tuple[int, str, str]], List[Tuple[str, str, float]]]:
    seen = set()
    lines, index_map, topk_list = [], [], []
    for i, d in enumerate(docs):
        idx = i + 1
        title = (d.metadata or {}).get("title") or f"Source {idx}"
        url   = canonicalize_url((d.metadata or {}).get("url") or (d.metadata or {}).get("source") or "")
        snippet = (d.page_content or "").strip()
        if SNIPPET_CHARS and len(snippet) > SNIPPET_CHARS:
            snippet = snippet[:SNIPPET_CHARS] + "…"
        score = None
        if score_by_key is not None:
            score = score_by_key.get(_score_key(title, url))
        score_txt = f" (SCORE={float(score):.4f})" if isinstance(score, (int, float)) else ""
        lines.append(f"[{idx}] {title} — {url}{score_txt}\n{snippet}\n")
        index_map.append((idx, title, url))
        key = (title, url)
        if key not in seen and (title or url):
            seen.add(key)
            topk_list.append((title, url, float(score) if isinstance(score, (int, float)) else float('nan')))
    return "\n".join(lines), index_map, topk_list

def extract_answer_block(full_text: str) -> str:
    m = re.search(r"Answer:\s*(.*)", full_text, re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else full_text.strip()

def build_sources_section(valid_indices: List[int], index_map: List[Tuple[int, str, str]]) -> str:
    if not valid_indices:
        return "Sources used:\n- (no valid cited sources found in retrieved context)"
    idx_to_meta = {i: (t, u) for i, t, u in index_map}
    used_sources, seen_urls = [], set()
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
        if score == score:
            lines.append(f"- {title}: {url} (SCORE={score:.4f})")
        else:
            lines.append(f"- {title}: {url}")
    return "Top-K searched in:\n\n" + "\n".join(lines)

# Spelling-only fix (no paraphrase)
def _fix_spelling_only_llm(text: str) -> str:
    try:
        prompt = (
            "Correct only obvious spelling mistakes in the following text. "
            "Do not paraphrase, remove, add, or reorder words. "
            "If there are no mistakes, return it exactly as-is.\n\n"
            f"Text: {text}\n\nCorrected:"
        )

        if hasattr(llm, "pipeline"):
            # HuggingFace pipeline
            resp = llm.pipeline(
                prompt,
                max_new_tokens=64,
                temperature=0.0,
                do_sample=False,
                return_full_text=False
            )
            out = str(resp[0]["generated_text"]).strip()
        else:
            # GGUF (LlamaCpp)
            out = str(llm.invoke(prompt)).strip()

        if len(out) == 0 or out.lower() in {"n/a", "none"}:
            return text
        return out
    except Exception:
        return text

# History / Carryover
def _history_snippets(messages: List[Dict[str, str]], turns: int) -> str:
    if turns <= 0 or not messages:
        return ""
    hist = []
    i = len(messages) - 1
    if i >= 0 and messages[i].get("role") == "user":
        i -= 1
    collected_pairs = 0
    while i >= 1 and collected_pairs < turns:
        if messages[i].get("role") == "assistant" and messages[i-1].get("role") == "user":
            u = messages[i-1].get("content", "").strip()
            a = strip_llm_sources_block(messages[i].get("content", "").strip())
            hist.append(f"User: {u}\nAssistant: {a}")
            collected_pairs += 1
            i -= 2
        else:
            i -= 1
    hist.reverse()
    return "\n\n".join(hist)

def _dedup_docs(docs: List[Document]) -> List[Document]:
    seen, out = set(), []
    for d in docs:
        title = (d.metadata or {}).get("title") or ""
        url   = canonicalize_url((d.metadata or {}).get("url") or (d.metadata or {}).get("source") or "")
        key = (title.strip(), url)
        if key not in seen:
            seen.add(key)
            out.append(d)
    return out

# =========================
# QA-Pipeline
# =========================
def answer_with_sources(
    question: str,
    top_k: int,
    debug: bool,
    follow_up_on: bool,
    carry_k: int,
    hist_turns: int
) -> Tuple[str, List[Tuple[int, str, str]]]:

    original_q = question
    if spellfix_active:
        question = _fix_spelling_only_llm(question)

    rv = build_retriever(vector_store, top_k, USE_MMR)
    fresh_docs: List[Document] = rv.get_relevant_documents(question)

    score_by_key: Dict[Tuple[str, str], float] = {}
    try:
        docs_scores = vector_store.similarity_search_with_score(question, k=max(top_k, len(fresh_docs)))
        for d, s in docs_scores:
            title = (d.metadata or {}).get("title") or ""
            url   = canonicalize_url((d.metadata or {}).get("url") or (d.metadata or {}).get("source") or "")
            score_by_key[_score_key(title, url)] = float(s)
    except Exception:
        pass

    # Follow-up OFF ⇒ no carryover, no history
    last_docs: List[Document] = st.session_state.get("last_docs", []) if follow_up_on else []
    carry_docs = (last_docs[:carry_k] if (follow_up_on and last_docs and carry_k > 0) else [])
    merged_docs = _dedup_docs(carry_docs + fresh_docs)[:max(top_k, len(carry_docs))]
    show_docs = merged_docs[:top_k] if len(merged_docs) >= top_k else merged_docs

    context_text, index_map, topk_list = format_docs(show_docs, score_by_key=score_by_key)

    hist_block = _history_snippets(st.session_state.get("messages", []), hist_turns) if follow_up_on else ""
    history_section = f"\n\n---\nRecent chat (for reference only):\n{hist_block}\n\n---\n" if hist_block else ""

    prompt = f"{SYSTEM_PROMPT}\n\nContext:\n{context_text}{history_section}\nQuestion: {question}\n\nAnswer:"
    out = llm.invoke(prompt)
    full_text = str(out)

    answer = extract_answer_block(full_text)
    is_no_context = (answer.strip().lower() == "not in context.")

    valid_indices: List[int] = []
    sources_section = ""
    if not is_no_context:
        used_indices = extract_used_indices(answer)
        available_indices = {i for i, _, _ in index_map}
        invalid_indices = used_indices - available_indices
        if invalid_indices:
            answer = remove_invalid_citations(answer, invalid_indices)
            used_indices = extract_used_indices(answer)
        valid_indices = sorted(list(used_indices & available_indices))
        if not valid_indices and index_map:
            answer = re.sub(r"([.!?])(\s|$)", r" [1]\1\2", answer, count=1)
            valid_indices = [1]
        sources_section = build_sources_section(valid_indices, index_map)

    # Debug (and ONLY show Top-K section when debugging)
    debug_sections = ""
    if debug:
        topk_section = build_topk_section(topk_list)
        carry_info = f"Carryover docs used: {len(carry_docs) if follow_up_on else 0} / {carry_k if follow_up_on else 0}"
        hist_info  = f"History turns shown: {hist_turns if follow_up_on else 0}"
        meta_dump = f"— Meta —\n{carry_info}\n{hist_info}\n"
        rew_dump  = f"— Original question —\n{original_q}\n\n— Spell-fixed —\n{question}\n"
        full_context_dump = f"— Full Context (as given to LLM) —\n{context_text}"
        debug_sections = "\n\n" + (topk_section + "\n\n" if topk_section else "") + meta_dump + rew_dump + "\n" + full_context_dump

    st.session_state["last_docs"] = show_docs if follow_up_on else []

    if is_no_context:
        final_text = f"{answer}{debug_sections}"
    else:
        final_text = f"{answer}\n\n{sources_section}{debug_sections}"

    return final_text, index_map

# =========================
# Chat UI
# =========================
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_docs" not in st.session_state:
    st.session_state.last_docs = []

for m in st.session_state.messages:
    role = "user" if m["role"] == "user" else "assistant"
    with st.chat_message(role):
        st.write(m["content"])

user_q = st.chat_input("Ask about your data …")
if user_q:
    st.session_state.messages.append({"role": "user", "content": user_q})
    with st.chat_message("user"):
        st.write(user_q)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving & generating …"):
            try:
                ans, _ = answer_with_sources(
                    question=user_q,
                    top_k=k,
                    debug=debug_mode,
                    follow_up_on=follow_up_active,
                    carry_k=carry_docs_k,
                    hist_turns=history_turns
                )
            except Exception as e:
                st.error(f"Error: {e}")
                st.stop()
            st.write(ans)

    st.session_state.messages.append({"role": "assistant", "content": ans})
