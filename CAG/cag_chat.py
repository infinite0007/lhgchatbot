#!/usr/bin/env python
# -*- coding: utf-8 -*-
# run with: streamlit run ./cag_chat_hot.py

import os, re, json
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import torch
import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer

# ----- Optional FAISS -----
try:
    import faiss
    HAS_FAISS = True
except Exception:
    HAS_FAISS = False

# --------------- CONFIG ---------------
CACHE_DIR        = "cag_cache1.5b"
HF_MODEL_PATH    = "../Qwen2.5-1.5B-Instruct"
TOP_K            = 6
TEMPERATURE      = 0.0
MAX_TOKENS       = 450
SNIPPET_CHARS    = 900
USE_FAISS        = True

st.set_page_config(page_title="Liebherr Chatbot", page_icon="🦚", layout="wide")
st.title("🦚 CAG Chat – Liebherr Software")
st.caption("Cache-Augmented Generation (CAG) with a local embedding cache. It accesses cached enterprise knowledge from the [Confluence Software](https://helpd-doc.liebherr.com/spaces/SWEIG/pages/43424891/SW-Platform-Development+Home+E2020) space — by [Julian Lingnau](https://de.linkedin.com/in/julian-lingnau-05b623162).")

with st.sidebar:
    st.subheader("Settings")
    st.text_input("Cache path", value=CACHE_DIR, disabled=True)
    st.text_input("Model", value=os.path.basename(HF_MODEL_PATH), disabled=True)
    k          = st.slider("Top-K", 1, 20, TOP_K, 1)
    temp       = st.slider("Temperature", 0.0, 1.5, TEMPERATURE, 0.05)
    max_tokens = st.slider("Max new tokens", 64, 1024, MAX_TOKENS, 32)
    spellfix_active = st.toggle("Spell checker", value=True)
    debug_mode      = st.toggle("Further debugging", value=False)

# --------------- Cache laden ---------------
@st.cache_resource(show_spinner=True)
def load_embed_cache(cache_dir: str):
    cdir = Path(cache_dir)
    emb_mmap = np.load(cdir / "embeddings.npy", mmap_mode="r")
    EMB = np.asarray(emb_mmap, dtype=np.float32)  # NumPy 2.x safe

    texts, metas = [], []
    with open(cdir / "texts.jsonl", "r", encoding="utf-8") as f:
        for ln in f:
            if ln.strip():
                obj = json.loads(ln); texts.append(obj["text"])
    with open(cdir / "meta.jsonl", "r", encoding="utf-8") as f:
        for ln in f:
            if ln.strip():
                metas.append(json.loads(ln))

    cfg = {}
    cfg_path = cdir / "config.json"
    if cfg_path.exists():
        try:
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        except Exception:
            cfg = {}
    hotset = []
    hot_path = cdir / "kv_hotset.json"
    if hot_path.exists():
        try:
            hotset = json.loads(hot_path.read_text(encoding="utf-8")).get("hot_chunk_ids", [])
        except Exception:
            hotset = []
    return EMB, texts, metas, cfg, set(hotset)

EMB, TEXTS, METAS, CFG, HOTSET = load_embed_cache(CACHE_DIR)
st.info(f"📦 Cache: {EMB.shape[0]} Chunks • {EMB.shape[1]}-D • Hot-Set: {len(HOTSET)}")

# --------------- Similarity backend ---------------
def _faiss_backend(emb: np.ndarray):
    index = faiss.IndexFlatIP(emb.shape[1])
    if faiss.get_num_gpus() > 0:
        _ = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_all_gpus(index)
    index.add(emb.astype(np.float32))
    def _search(qv: np.ndarray, top_k: int):
        D, I = index.search(qv.reshape(1, -1).astype(np.float32), top_k)
        return D[0].tolist(), I[0].tolist()
    return _search

def _numpy_backend(emb: np.ndarray):
    emb_local = emb
    def _search(qv: np.ndarray, top_k: int):
        s = emb_local @ qv
        topk = min(top_k, s.shape[0])
        idx = np.argpartition(-s, topk - 1)[:topk]
        idx = idx[np.argsort(-s[idx])]
        return s[idx].tolist(), idx.tolist()
    return _search

@st.cache_resource(show_spinner=False)
def get_search_backend(emb: np.ndarray):
    if USE_FAISS and HAS_FAISS:
        return _faiss_backend(emb), "FAISS"
    return _numpy_backend(emb), "NumPy"

SEARCH, BACKEND = get_search_backend(EMB)
st.caption(f"Similarity backend: **{BACKEND}** • Metric: cosine")

# --------------- Query encoder (wie Build) ---------------
@st.cache_resource(show_spinner=True)
def load_query_encoder(name: str) -> SentenceTransformer:
    return SentenceTransformer(name, device="cuda")

embed_model_name = CFG.get("embed_model", "sentence-transformers/all-MiniLM-L6-v2")
query_encoder = load_query_encoder(embed_model_name)

def encode_query(q: str) -> np.ndarray:
    # mit device="cuda" ist das trotzdem ok – .encode nutzt GPU
    v = query_encoder.encode([q], convert_to_numpy=True, normalize_embeddings=True)[0]
    return v.astype(np.float32)

# --------------- LLM + Tokenizer ---------------
@st.cache_resource(show_spinner=True)
def load_llm(model_path: str):
    tok = AutoTokenizer.from_pretrained(model_path, use_fast=True, local_files_only=True, trust_remote_code=True)
    mdl = AutoModelForCausalLM.from_pretrained(
        model_path, device_map="auto", torch_dtype="auto", local_files_only=True, trust_remote_code=True
    ).eval()
    return tok, mdl

tok, mdl = load_llm(HF_MODEL_PATH)
MODEL_DEVICE = next(mdl.parameters()).device
MODEL_DTYPE  = next(mdl.parameters()).dtype

# --------------- KV Laden/Mappen ---------------
def _map_legacy_to_device_dtype(kv_legacy: List[Tuple[torch.Tensor, torch.Tensor]]):
    out = []
    for k, v in kv_legacy:
        out.append((
            k.to(device=MODEL_DEVICE, dtype=MODEL_DTYPE, non_blocking=True),
            v.to(device=MODEL_DEVICE, dtype=MODEL_DTYPE, non_blocking=True),
        ))
    return out

@st.cache_resource(show_spinner=False)
def load_chunk_kv_legacy(cache_dir: str, chunk_id: str):
    path = Path(cache_dir) / "kv" / f"{chunk_id}.pt"
    try:
        obj = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        obj = torch.load(path, map_location="cpu")
    except Exception:
        obj = torch.load(path, map_location="cpu", weights_only=False)
    kv_legacy = obj["kv_legacy"]
    seq_len   = int(obj["seq_len"])
    return kv_legacy, seq_len

# Hot-Set KV in RAM/VRAM vorladen (dict: chunk_id -> mapped legacy list)
@st.cache_resource(show_spinner=True)
def warm_hotset(cache_dir: str, hot_ids: set):
    table: Dict[str, List[Tuple[torch.Tensor, torch.Tensor]]] = {}
    loaded = 0
    for cid in hot_ids:
        try:
            kv_legacy, _ = load_chunk_kv_legacy(cache_dir, cid)
            table[cid] = _map_legacy_to_device_dtype(kv_legacy)
            loaded += 1
        except Exception:
            pass
    return table, loaded

HOT_KV, HOT_LOADED = warm_hotset(CACHE_DIR, HOTSET)
st.success(f"🔥 Hot-Set geladen: {HOT_LOADED}/{len(HOTSET)} KV-Chunks im Speicher")

# --------------- Helper (Quellen/Anzeige) ---------------
RE_CIT = re.compile(r"\[(\d+)\]")

def canonicalize_url(u: str) -> str:
    if not u: return u
    u = u.strip().split("#")[0]
    return u[:-1] if u.endswith("/") else u

def format_context(idxs: List[int], scores: List[float]):
    lines, index_map = [], []
    for rank, (i, sc) in enumerate(zip(idxs, scores), start=1):
        meta = METAS[i] if i < len(METAS) else {}
        title = meta.get("title") or f"Source {rank}"
        url   = canonicalize_url(meta.get("url") or meta.get("source") or "")
        snippet = (TEXTS[i] if i < len(TEXTS) else "").strip()
        if SNIPPET_CHARS and len(snippet) > SNIPPET_CHARS:
            snippet = snippet[:SNIPPET_CHARS] + "…"
        lines.append(f"[{rank}] {title} — {url} (SCORE={float(sc):.4f})\n{snippet}\n")
        index_map.append((rank, title, url, float(sc)))
    return "\n".join(lines), index_map

def remove_invalid_citations(text: str, valid_indices: set) -> str:
    used = {int(x) for x in RE_CIT.findall(text)}
    invalid = used - valid_indices
    for i in sorted(invalid, reverse=True):
        text = re.sub(rf"(?<!\d)\[{i}\](?!\d)", "", text)
    return re.sub(r"\s{2,}", " ", text).strip()

# --------------- Spellfix (optional, kurz) ---------------
def _spellfix_llm(text: str) -> str:
    try:
        prompt = (
            "Correct only obvious spelling mistakes in the following text. "
            "Do not paraphrase, remove, add, or reorder words. "
            "If there are no mistakes, return it exactly as-is.\n\n"
            f"Text: {text}\n\nCorrected:"
        )
        with torch.no_grad():
            out = mdl.generate(
                **tok(prompt, return_tensors="pt").to(MODEL_DEVICE),
                max_new_tokens=64, do_sample=False, temperature=0.0,
                pad_token_id=tok.eos_token_id
            )
        s = tok.decode(out[0], skip_special_tokens=True)
        return s.split("Corrected:")[-1].strip() or text
    except Exception:
        return text

# --------------- KV-gestützte Generation (legacy list[(k,v)]) ---------------
def generate_with_kv(question: str, kv_legacy_mapped: List[Tuple[torch.Tensor, torch.Tensor]],
                     max_new_tokens: int, temperature: float) -> str:
    tail = question + "\n\nAnswer:"
    tail_ids = tok(tail, return_tensors="pt", add_special_tokens=False).input_ids.to(MODEL_DEVICE)

    with torch.no_grad():
        out = mdl(input_ids=tail_ids, past_key_values=kv_legacy_mapped, use_cache=True, return_dict=True)
        logits = out.logits[:, -1, :]
        past  = out.past_key_values  # legacy list bleibt kompatibel

    def _next_token(logits_tensor):
        if temperature and temperature > 0.0:
            probs = torch.softmax(logits_tensor / temperature, dim=-1)
            return torch.multinomial(probs, num_samples=1)
        return torch.argmax(logits_tensor, dim=-1, keepdim=True)

    generated, cur_ids = [], None
    for _ in range(max_new_tokens):
        with torch.no_grad():
            if cur_ids is None:
                token = _next_token(logits)
            else:
                out = mdl(input_ids=cur_ids, past_key_values=past, use_cache=True, return_dict=True)
                past = out.past_key_values
                token = _next_token(out.logits[:, -1, :])

        tid = int(token.item())
        if tok.eos_token_id is not None and tid == tok.eos_token_id:
            break
        generated.append(tid)
        cur_ids = token

    return tok.decode(generated, skip_special_tokens=True)

# --------------- QA-Pipeline (Hot-Set First) ---------------
def answer_with_sources(question: str, top_k: int, debug: bool) -> str:
    original_q = question
    if spellfix_active:
        question = _spellfix_llm(question)

    qv = encode_query(question)
    scores, idxs = SEARCH(qv, top_k=max(top_k, 6))
    idxs, scores = idxs[:top_k], scores[:top_k]
    if not idxs:
        return "Not in context."

    # 1) bevorzugt Chunks aus dem Hot-Set nutzen (0-I/O)
    best_idx = None
    for i in idxs:
        if METAS[i]["chunk_id"] in HOT_KV:
            best_idx = i; break
    if best_idx is None:
        best_idx = idxs[0]

    cid = METAS[best_idx]["chunk_id"]
    if cid in HOT_KV:
        kv = HOT_KV[cid]
    else:
        # einmalig von Disk holen (kalter Pfad)
        kv_legacy, _ = load_chunk_kv_legacy(CACHE_DIR, cid)
        kv = _map_legacy_to_device_dtype(kv_legacy)

    context_text, index_map = format_context(idxs, scores)
    valid_indices = {i for i, *_ in index_map}

    ans = generate_with_kv(question, kv, max_tokens, temp).strip()
    if ans.lower() == "not in context.":
        return "Not in context."

    if "[1]" not in ans:
        ans = re.sub(r"([.!?])(\s|$)", r" [1]\1\2", ans, count=1)
    ans = remove_invalid_citations(ans, valid_indices)

    lines = [f"- [{idx}] {title}: {canonicalize_url(url)}" for idx, title, url, _ in index_map]
    sources = "Sources used:\n\n" + "\n".join(lines)

    if debug:
        dbg = f"\n\n— Full Context —\n{context_text}\n\n— Original —\n{original_q}\n— Spell-fixed —\n{question}"
        return f"{ans}\n\n{sources}{dbg}"
    return f"{ans}\n\n{sources}"

# --------------- Chat UI ---------------
if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message("user" if m["role"] == "user" else "assistant"):
        st.write(m["content"])

user_q = st.chat_input("Ask about your data …")
if user_q:
    st.session_state.messages.append({"role": "user", "content": user_q})
    with st.chat_message("user"):
        st.write(user_q)
    with st.chat_message("assistant"):
        with st.spinner("Hot-Set KV & generating …"):
            try:
                ans = answer_with_sources(user_q, k, debug_mode)
            except Exception as e:
                st.error(f"Error: {e}")
                st.stop()
            st.write(ans)
    st.session_state.messages.append({"role": "assistant", "content": ans})
