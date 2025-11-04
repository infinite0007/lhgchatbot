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
from langchain_community.llms import HuggingFacePipeline, LlamaCpp
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
# CONFIG
# =========================
CACHE_DIR        = "cag_cache"  # contains embeddings.npy, texts.jsonl, meta.jsonl
EMBEDDING_MODEL  = "sentence-transformers/all-MiniLM-L6-v2"
HF_MODEL_PATH    = "../gemma-3-4b-Instruct"

TOP_K         = 6
TEMPERATURE   = 0.1
MAX_TOKENS    = 450
SNIPPET_CHARS = 900 # max. Zeichen pro Treffer im Prompt
USE_FAISS     = True # wenn verfügbar, nutze FAISS (CPU/GPU). Fallback: NumPy / Torch

# ---- Quantifizierte Einstellungen GGUF ----
USE_GGUF        = False # True = ungenauer und Antwortqualität leidet under GGUF
GGUF_MODEL_PATH = "../gemma-3-4b-Instruct/GGUF/gemma3-4b-instruct.gguf"  # Quantifiziertes Modell
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
st.set_page_config(page_title="Liebherr Chatbot", page_icon="🦚", layout="wide")
st.title("🦚 CAG Chat – Liebherr Software")
st.caption("Cache-Augmented Generation (CAG) with a local embedding cache. It accesses cached enterprise knowledge from the [Confluence Software](https://helpd-doc.liebherr.com/spaces/SWEIG/pages/43424891/SW-Platform-Development+Home+E2020) space — by [Julian Lingnau](https://de.linkedin.com/in/julian-lingnau-05b623162).")

with st.sidebar:
    st.subheader("Settings")
    st.text_input("Cache path", value=CACHE_DIR, disabled=True)
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
# Cache loading
# =========================
@st.cache_resource(show_spinner=True)
def load_cag_cache(cache_dir: str):
    cdir = Path(cache_dir)
    emb = np.load(cdir / "embeddings.npy", mmap_mode="r")
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
    st.info(f"📦 Cache loaded: **{n_chunks}** chunks, **{dim}**-D embeddings")
except Exception as e:
    st.error(f"Cache load error from '{CACHE_DIR}': {e}")
    st.stop()

# =========================
# Query encoder
# =========================
@st.cache_resource(show_spinner=True)
def load_query_encoder(model_name: str) -> SentenceTransformer:
    return SentenceTransformer(model_name)

query_encoder = load_query_encoder(EMBEDDING_MODEL)

# =========================
# LLM
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
# Retrieval backends
# =========================
def _faiss_build_index(emb: np.ndarray):
    # Cosine via inner product bei l2-normalized Vektoren
    index = faiss.IndexFlatIP(emb.shape[1])
    if faiss.get_num_gpus() > 0:
        _ = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_all_gpus(index) # verteilt auf alle GPUs
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
            scores = torch.matmul(t_emb, t_q.t()).squeeze(1)
            topk = min(top_k, scores.shape[0])
            vals, idx = torch.topk(scores, k=topk, largest=True, sorted=True)
            return vals.detach().cpu().numpy().tolist(), idx.detach().cpu().numpy().tolist()
        return _search_torch, "TorchCUDA"
    def _search_numpy(qv: np.ndarray, top_k: int):
        s = EMB @ qv
        topk = min(top_k, s.shape[0])
        idx = np.argpartition(-s, topk - 1)[:topk]
        idx = idx[np.argsort(-s[idx])]
        return s[idx].tolist(), idx.tolist()
    return _search_numpy, "NumPy"

SEARCH, BACKEND = get_search_backend(EMB)
st.caption(f"Similarity backend: **{BACKEND}** • Metric: **cosine** • Top-K: **{k}**")

# =========================
# Helpers (citations & formatting)
# =========================
RE_CITATIONS = re.compile(r"\[(\d+)\]")
SOURCES_RE = re.compile(r"(Sources\s+(used:|:))(.|\s)*?$", re.IGNORECASE)

def strip_llm_sources_block(text: str) -> str:
    """Schneidet vom LLM generierte 'Sources used:'-Abschnitte am Ende weg."""
    return re.sub(SOURCES_RE, "", text).rstrip()

def canonicalize_url(u: str) -> str:
    if not u: return u
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
    seen, lines, index_map, topk_list = set(), [], [], []
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
    lines = [f"- {t}: {u} (SCORE={s:.4f})" for t, u, s in topk_list]
    return "Top-K searched in:\n\n" + "\n".join(lines)

# Spelling-only fix
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

# =========================
# QA-Pipeline
# =========================
def encode_query(q: str) -> np.ndarray:
    qv = query_encoder.encode([q], convert_to_numpy=True, normalize_embeddings=True)
    return qv[0].astype(np.float32)

def answer_with_sources(
    question: str,
    top_k: int,
    debug: bool,
    follow_up_on: bool,
    carry_k: int,
    hist_turns: int
) -> Tuple[str, List[Tuple[int, str, str, float]]]:

    original_q = question
    if spellfix_active:
        question = _fix_spelling_only_llm(question)

    qv = encode_query(question)
    scores, idxs = SEARCH(qv, top_k=max(top_k, 6))
    # Follow-up OFF ⇒ ignore any previous context
    if not follow_up_on:
        # Use only fresh retrieval results
        idxs = idxs[:top_k]
        scores = scores[:top_k]

    context_text, index_map, topk_list = format_context(idxs[:top_k], scores[:top_k])

    # History only if follow-up ON
    if follow_up_on:
        # build small history window for reference only (same format as RAG)
        def _history_snippets(messages: List[Dict[str, str]], turns: int) -> str:
            if turns <= 0 or not messages:
                return ""
            hist = []
            i = len(messages) - 1
            if i >= 0 and messages[i].get("role") == "user":
                i -= 1
            collected = 0
            while i >= 1 and collected < turns:
                if messages[i].get("role") == "assistant" and messages[i-1].get("role") == "user":
                    u = messages[i-1].get("content", "").strip()
                    a = strip_llm_sources_block(messages[i].get("content", "").strip())
                    hist.append(f"User: {u}\nAssistant: {a}")
                    collected += 1
                    i -= 2
                else:
                    i -= 1
            hist.reverse()
            return "\n\n".join(hist)
        hist_block = _history_snippets(st.session_state.get("messages", []), hist_turns)
    else:
        hist_block = ""

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
        available_indices = {i for i, _, _, _ in index_map}
        invalid_indices = used_indices - available_indices
        if invalid_indices:
            answer = remove_invalid_citations(answer, invalid_indices)
            used_indices = extract_used_indices(answer)
        valid_indices = sorted(list(used_indices & available_indices))
        if not valid_indices and index_map:
            answer = re.sub(r"([.!?])(\s|$)", r" [1]\1\2", answer, count=1)
            valid_indices = [1]
        sources_section = build_sources_section(valid_indices, index_map)

    debug_sections = ""
    if debug:
        topk_section = build_topk_section(topk_list)
        carry_info = f"Carryover docs used: {len(index_map) if follow_up_on else 0} / {carry_k if follow_up_on else 0}"
        hist_info  = f"History turns shown: {hist_turns if follow_up_on else 0}"
        meta_dump = f"— Meta —\n{carry_info}\n{hist_info}\n"
        rew_dump  = f"— Original question —\n{original_q}\n\n— Spell-fixed —\n{question}\n"
        full_context_dump = f"— Full Context (as given to LLM) —\n{context_text}"
        debug_sections = "\n\n" + (topk_section + "\n\n" if topk_section else "") + meta_dump + rew_dump + "\n" + full_context_dump

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
        with st.spinner("Searching cache & generating …"):
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
