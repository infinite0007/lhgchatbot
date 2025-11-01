#!/usr/bin/env python
# -*- coding: utf-8 -*-
# run with: streamlit run .\rag_chat.py
import re
from typing import List, Any, Tuple, Dict, Optional

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
HF_MODEL_PATH     = "../gemma-3-4b-Instruct"  # dein lokales Modell Instruct Modelle empfohlen da diese wissen wan man aufhören muss und genau auf Frage Antwort trainiert sind. Base Modelle generieren oft nach Antwort neue Fragen und sind nicht speziell fürs Chatten trainiert also Anwendungsfall ab 4b Modelle empfohlen um auch komplexe Eingaben oder kombinierte zu verstehen

TOP_K        = 6
USE_MMR      = False
TEMPERATURE  = 0.1
MAX_TOKENS   = 450

SYSTEM_PROMPT = (
    "You are a precise assistant for enterprise knowledge.\n"
    "Follow these steps strictly and in order:\n"
    "1. Before searching, check the question for spelling or typing errors and silently correct them.\n"
    "2. Search only within the provided context to find an answer.\n"
    "3. If no relevant information can be found, reply only with: 'Not in context.' and do not cite any sources.\n"
    "4. If the question can be partially answered, provide only the part that is supported by the context.\n"
    "5. Keep answers concise and factual.\n"
    "6. Add cites into the text but only if it's available in the provided context. Use [1], [2], ... markers.\n"
    "7. Do NOT invent sources or add information not in the context.\n"
    "8. Do NOT ask follow-up questions. Stop after giving the answer and sources.\n"
    "This is important — follow the steps exactly and do not hallucinate."
)

# =========================
# UI früh rendern
# =========================
st.set_page_config(page_title="Liebherr Chatbot", page_icon="🦜", layout="wide")
st.title("🦜 RAG Chat – Liebherr Software")
st.caption("Retrieval-Augmented Generation (RAG) with a vector database. It retrieves enterprise knowledge from the [Confluence Software](https://helpd-doc.liebherr.com/spaces/SWEIG/pages/43424891/SW-Platform-Development+Home+E2020) space — by [Julian Lingnau](https://de.linkedin.com/in/julian-lingnau-05b623162).")

with st.sidebar:
    st.subheader("Settings")
    st.text_input("DB path", value=DATABASE_LOCATION, disabled=True)
    st.text_input("Collection", value=COLLECTION_NAME, disabled=True)
    st.text_input("Embeddings", value=EMBEDDING_MODEL, disabled=True)
    st.text_input("HF model path", value=HF_MODEL_PATH, disabled=True)
    k = st.slider("Top-K", 1, 20, TOP_K, 1)
    temp = st.slider("Temperature", 0.0, 1.5, TEMPERATURE, 0.05)
    max_tokens = st.slider("Max new tokens", 64, 1024, MAX_TOKENS, 32)
    follow_up_active = st.toggle("Follow-up question", value=True)
    debug_mode = st.toggle("Further debugging", value=False)

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
def load_llm(model_path: str, max_new_tokens: int, temperature: float, debug: bool) -> HuggingFacePipeline:
    tok = AutoTokenizer.from_pretrained(model_path, use_fast=True, local_files_only=True,)
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
        # False => gibt nur Antwort, nicht Prompt/Quellen zusätzlich ins LLM - bei True sieht man auch woher er die Infos im Text zusammengestellt hat
        return_full_text=True,
        pad_token_id=tok.eos_token_id if tok.eos_token_id is not None else None,
    )
    return HuggingFacePipeline(pipeline=gen)

llm = load_llm(HF_MODEL_PATH, max_tokens, temp, debug_mode)

# =========================
# RAG-Helfer
# =========================
RE_CITATIONS = re.compile(r"\[(\d+)\]")

def extract_used_indices(text: str) -> set:
    return set(int(m) for m in RE_CITATIONS.findall(text))

def remove_invalid_citations(text: str, invalid: set[int]) -> str:
    """Entfernt alle [x] aus dem Text, deren x ungültig ist."""
    for i in sorted(invalid, reverse=True):
        text = re.sub(rf"(?<!\d)\[{i}\](?!\d)", "", text)
    text = re.sub(r"\s{2,}", " ", text).strip()
    return text

def canonicalize_url(u: str) -> str:
    """Einfache Kanonisierung für Dedupe (remove trailing slash, fragments, whitespace)."""
    if not u:
        return u
    u = u.strip()
    u = u.split("#")[0]
    if u.endswith("/"):
        u = u[:-1]
    return u

def _score_key(title: str, url: str) -> Tuple[str, str]:
    return (title or "").strip(), canonicalize_url(url or "")

def format_docs(
    docs: List[Any],
    score_by_key: Optional[Dict[Tuple[str, str], float]] = None
) -> Tuple[str, List[Tuple[int, str, str]], List[Tuple[str, str, float]]]:
    """
    Returns:
      - context_text: string with numbered entries [1] ... [n] (inkl. SCORE wenn vorhanden)
      - index_map: list of tuples (index, title, url) for every shown index
      - topk_list: deduped list of (title, url, score) for the Top-K display
    """
    seen = set()
    lines = []
    index_map = []
    topk_list = []
    for i, d in enumerate(docs):
        idx = i + 1
        title = d.metadata.get("title") or f"Source {idx}"
        url   = canonicalize_url(d.metadata.get("url") or d.metadata.get("source") or "")
        snippet = (d.page_content or "").strip()
        snippet = snippet[:900] + ("…" if len(snippet) > 900 else "")
        score = None
        if score_by_key is not None:
            score = score_by_key.get(_score_key(title, url))
        score_txt = f" (SCORE={score:.4f})" if isinstance(score, (int, float)) else ""
        lines.append(f"[{idx}] {title} — {url}{score_txt}\n{snippet}\n")
        index_map.append((idx, title, url))
        key = (title, url)
        if key not in seen and (title or url):
            seen.add(key)
            topk_list.append((title, url, float(score) if isinstance(score, (int, float)) else float('nan')))
    return "\n".join(lines), index_map, topk_list

def extract_answer_block(full_text: str) -> str:
    """Nimmt alles NACH 'Answer:'; Fallback: gesamt."""
    m = re.search(r"Answer:\s*(.*)", full_text, re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else full_text.strip()

def build_sources_section(valid_indices: List[int], index_map: List[Tuple[int, str, str]]) -> str:
    if not valid_indices:
        return "Sources used:\n- (no valid cited sources found in retrieved context)"
    idx_to_meta = {i: (t, u) for i, t, u in index_map}
    used_sources = []
    seen_urls = set()
    for i in valid_indices:
        t, u = idx_to_meta[i]
        u_c = canonicalize_url(u)
        if u_c not in seen_urls:
            seen_urls.add(u_c)
            used_sources.append(f"[{i}] {t}: {u}")
    return "Sources used:\n\n" + "\n".join(f"- {s}" for s in used_sources)

def build_topk_section(topk_list: List[Tuple[str, str, float]]) -> str:
    if not topk_list:
        return ""
    lines = []
    for title, url, score in topk_list:
        if score == score:  # not NaN
            lines.append(f"- {title}: {url} (SCORE={score:.4f})")
        else:
            lines.append(f"- {title}: {url}")
    return "Top-K searched in:\n\n" + "\n".join(lines)

# ========= History-aware Query-Rewriting (nur wenn Follow-up aktiv) =========
def _history_window(messages: List[Dict[str, str]], max_pairs: int = 3) -> List[Dict[str, str]]:
    """Nimmt die letzten max_pairs User/Assistant-Paare (ohne den aktuellen Input)."""
    hist = []
    for m in messages:
        if m.get("role") in ("user", "assistant"):
            hist.append({"role": m["role"], "content": m["content"]})
    # letztes Element ist meist die letzte Assistant-Antwort (vor der neuen Frage)
    return hist[-(max_pairs*2):] if hist else []

def _rewrite_with_history(original_q: str, history_snippets: List[Dict[str, str]]) -> str:
    """
    Nutzt dasselbe LLM deterministisch (temp=0.0), um Folgefragen zu disambiguieren.
    Gibt bei Fehlschlag die Originalfrage zurück.
    """
    if not history_snippets:
        return original_q

    # Kompakter Prompt, nur Umschreiben – keine Antwort erzeugen.
    # Wir verwenden die bestehende Pipeline, aber überschreiben Parameter für deterministische, kurze Outputs.
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
        # Kurz & deterministisch generieren (kein cache-busting an deiner Pipeline)
        gen = llm.pipeline
        resp = gen(prompt, max_new_tokens=96, temperature=0.0, do_sample=False, return_full_text=False)
        text = str(resp[0]["generated_text"]).strip()
        # Einfache Post-Korrektur: harte Zeilenumbrüche weg, Anführungen entfernen
        text = " ".join(text.split()).strip().strip('"').strip("'")
        # Leerer/zu kurzer Output → Fallback
        if len(text) < 3:
            return original_q
        return text
    except Exception:
        return original_q

# =========================
# QA-Pipeline
# =========================
def answer_with_sources(question: str, top_k: int, debug: bool, follow_up: bool) -> Tuple[str, List[Tuple[int, str, str]]]:
    # (A) optional: History-aware Rewrite
    history_used = _history_window(st.session_state.get("messages", []), max_pairs=3)
    original_q = question
    effective_q = question
    if follow_up:
        effective_q = _rewrite_with_history(original_q, history_used)

    # 1) Retrieve (für Texte)
    rv = build_retriever(vector_store, top_k, USE_MMR)
    docs = rv.get_relevant_documents(effective_q)

    # 1b) Parallel: Scores stabil über (Title, URL) holen
    score_by_key: Dict[Tuple[str, str], float] = {}
    try:
        docs_with_scores = vector_store.similarity_search_with_score(effective_q, k=max(top_k, len(docs)))
        for d, s in docs_with_scores:
            title = d.metadata.get("title") or ""
            url   = canonicalize_url(d.metadata.get("url") or d.metadata.get("source") or "")
            score_by_key[_score_key(title, url)] = float(s)
    except Exception:
        pass  # falls DB oder Backend Scores nicht liefert

    # 2) Kontexte vorbereiten (mit Scores)
    context_text, index_map, topk_list = format_docs(docs, score_by_key=score_by_key)

    # 3) Prompt + Generate
    prompt = f"{SYSTEM_PROMPT}\n\nContext:\n{context_text}\n\nQuestion: {effective_q}\n\nAnswer:"
    out = llm.invoke(prompt)
    full_text = str(out)

    # 4) Nur den eigentlichen Answer-Block isolieren
    answer = extract_answer_block(full_text)

    # 5) Zitate NUR aus dem Answer-Block extrahieren + validieren
    used_indices = extract_used_indices(answer)
    available_indices = {i for i, _, _ in index_map}
    invalid_indices = used_indices - available_indices
    if invalid_indices:
        answer = remove_invalid_citations(answer, invalid_indices)
        used_indices = extract_used_indices(answer)

    valid_indices = sorted(list(used_indices & available_indices))

    # 6) Fallback: wenn inhaltlich Kontext da ist, aber keine Zitate gesetzt wurden
    if not valid_indices and index_map and answer.strip().lower() != "not in context.":
        answer = re.sub(r"([.!?])(\s|$)", r" [1]\1\2", answer, count=1)
        valid_indices = [1]

    # 7) Sources used exakt aus valid_indices
    sources_section = build_sources_section(valid_indices, index_map)

    # 8) Debug-Extras (Top-K als Aufzählung + Full Context inkl. SCORE)
    debug_sections = ""
    if debug:
        topk_section = build_topk_section(topk_list)
        # History-Dump
        hist_lines = []
        if history_used:
            for h in history_used:
                role = "User" if h["role"] == "user" else "Assistant"
                # hart kürzen, damit UI nicht explodiert
                content = h["content"].strip()
                if len(content) > 500:
                    content = content[:500] + "…"
                hist_lines.append(f"{role}: {content}")
        history_dump = "\n".join(hist_lines) if hist_lines else "(none)"

        full_context_dump = f"— Full Context (as given to LLM) —\n{context_text}"
        rewrite_dump = (
            "— Follow-up rewrite applied: YES —\n"
            f"— Original question —\n{original_q}\n\n"
            f"— Effective question —\n{effective_q}\n\n"
            f"— History used (window=3) —\n{history_dump}"
            if follow_up else
            "— Follow-up rewrite applied: NO —"
        )
        debug_sections = "\n\n" + (topk_section + "\n\n" if topk_section else "") + rewrite_dump + "\n\n" + full_context_dump

    # 9) Finale Antwort zusammenbauen
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
        with st.spinner("Retrieving & generating …"):
            try:
                ans, _ = answer_with_sources(user_q, k, debug_mode, follow_up_active)
            except Exception as e:
                st.error(f"Fehler: {e}")
                st.stop()
            st.write(ans)
    st.session_state.messages.append({"role": "assistant", "content": ans})
