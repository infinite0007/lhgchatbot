#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
from itertools import combinations
from typing import Dict, List, Tuple

import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from sentence_transformers import SentenceTransformer
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# -------------------------
# RAG-Konfiguration
# -------------------------
DATABASE_LOCATION = "chroma_db"
COLLECTION_NAME   = "rag_data"
EMBEDDING_MODEL   = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K             = 6

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

# -------------------------
# Modelle: lokale HF-Pfade
# -------------------------
MODELS: Dict[str, str] = {
    "Phi-4-mini-instruct":   "../Phi-4-mini-instruct",
    "Qwen2.5-3B-Instruct":   "../Qwen2.5-3B-Instruct",
    "Falcon3-3B-Instruct":   "../Falcon3-3B-Instruct",
    "Llama-3.2-3B-Instruct": "../Llama-3.2-3B-Instruct",
    "gemma-3-4B-Instruct":   "../gemma-3-4b-Instruct",
}

EVAL_TEMPERATURE = 0.0
MAX_TOKENS       = 450

# -------------------------
# Fragenkatalog
# -------------------------
QUESTIONS: List[str] = [
    "Please explain me the basic architecture of the Powerboard Software.",
    "Explain the goals of the electronic platform (name it and how it can be extended).",
    "What is the ToD and what is it used for?",
    "How can the Powerboard software be stripped down to specific project needs?",
    "What is the purpose of the LPF?",
    "What is a Snapshot and for what is it used?",
    "How does the bus addressing work? Explain Function (Fct) and Group (Grp) IDs.",
    "Explain the SW-Split. Which Splits are done?",
    "Explain the different MainModes of the application.",
    "Which zone_type must be used for the IceCenter?",
    "What is the purpose of the HeatingTimeMonitoring. What is the pupose of parMaxSensorBasedHeatingTime and parMinimumHeatingOffTime?",
    "How does the configPowerProvisionCompressor affects the cooling control?",
    "What is the purpose of parRobustnessTempRetriggerDeltaFallingTemp?",
    "How does the b-value correction work for the Timebased Fan Control?",
    "When is the Cooling Compressor Speed Increasing not active?",
    "Please explain the Presentation Light functionality.",
    "What is the effect of stateLowPassFiltering on the display temperature?",
    "What does the parameter parTimerToSetTheAbsolutRangeAfterDoorOpenORSetTemperatureChange do?",
    "What states does a zone reminder have and how can it be reset?",
    "Gets the door alarm influenced by the autodoor feature?",
    "What happens if a reminder gets quitted on the UI or via CM. How is the stateAirFilterReminder handled. Who is responsible for changing from REMINDER_RESET to the nexxt status and what is the next status.",
    "There is a parameter for limiting the maximum time in cleaning mode. Is the function implemented identically for LH and Miele, as Miele appears to have included a door dependency? Can you please tell me exactly how the function is implemented?",
    "Explain me the zone dependency configuration in detail.",
    "What is the purpose of sabbath mode?",
    "What is the different between EMERGENCY_CONTROL_DEFAULT_COOLING and EMERGENCY_CONTROL_CYCLIC_COOLING?",
    "What is the purpose of parDefrostRecoveryTime in the context of automatic adaptive cooling capacity?",
    "Explain me the feature robustness temp retrigger. What is the purpose for the parRobustnessTempRetriggerDeltaFallingTemp?",
]


# -------------------------
# Helper-Funktionen
# -------------------------

def normalize_answer(text: str) -> str:
    """
    Bereinigt LLM-Antworten vor dem Embedding:
    - entfernt 'Answer:'-Prefix
    - entfernt Zitate wie [1], [23]
    - normalisiert Whitespace
    """
    if text is None:
        return ""
    t = text.strip()
    t = re.sub(r"^Answer:\s*", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\[\d+\]", "", t)
    t = re.sub(r"\s+", " ", t)
    return t.strip()


def load_vectorstore() -> Chroma:
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vs = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=DATABASE_LOCATION,
    )
    return vs


def build_retriever(vs: Chroma, top_k: int):
    return vs.as_retriever(search_kwargs={"k": top_k})


def format_docs_for_prompt(docs) -> str:
    """Einfaches Context-Format: Titel + Snippet."""
    lines = []
    for i, d in enumerate(docs, start=1):
        title   = (d.metadata or {}).get("title") or f"Source {i}"
        content = (d.page_content or "").strip()
        if len(content) > 900:
            content = content[:900] + "…"
        lines.append(f"[{i}] {title}\n{content}\n")
    return "\n".join(lines)


def extract_answer_block(full_text: str) -> str:
    """Falls 'Answer:' im Output vorkommt, nur den Antwortteil extrahieren."""
    m = re.search(r"Answer:\s*(.*)", full_text, re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else full_text.strip()


def load_llm(hf_model_path: str):
    tok = AutoTokenizer.from_pretrained(
        hf_model_path,
        use_fast=True,
        local_files_only=True
    )
    mdl = AutoModelForCausalLM.from_pretrained(
        hf_model_path,
        device_map="auto",
        torch_dtype="auto",
        local_files_only=True,
    )
    gen = pipeline(
        "text-generation",
        model=mdl,
        tokenizer=tok,
        max_new_tokens=MAX_TOKENS,
        temperature=EVAL_TEMPERATURE,
        do_sample=False,
        return_full_text=True,
        pad_token_id=tok.eos_token_id if tok.eos_token_id is not None else None,
    )
    from langchain_community.llms import HuggingFacePipeline
    return HuggingFacePipeline(pipeline=gen)


def run_rag_answer(llm, retriever, question: str) -> str:
    docs = retriever.get_relevant_documents(question)
    context = format_docs_for_prompt(docs)
    prompt = f"{SYSTEM_PROMPT}\n\nContext:\n{context}\n\nQuestion: {question}\n\nAnswer:"
    out = llm.invoke(prompt)
    return extract_answer_block(str(out))


# -------------------------
# Main
# -------------------------

def main() -> None:
    # 1) RAG vorbereiten
    vs        = load_vectorstore()
    retriever = build_retriever(vs, TOP_K)

    # 2) Antworten sammeln: answers[modell][i] = Text
    answers: Dict[str, List[str]] = {m: [] for m in MODELS.keys()}

    for model_name, model_path in MODELS.items():
        print(f"=== Evaluating model: {model_name} ===")
        llm = load_llm(model_path)
        for q in QUESTIONS:
            ans = run_rag_answer(llm, retriever, q)
            answers[model_name].append(ans)

    # 3) Antworten einbetten
    sim_encoder = SentenceTransformer(EMBEDDING_MODEL)
    model_names: List[str] = list(MODELS.keys())
    N: int = len(QUESTIONS)

    # ans_embeds[modell][i] = np.ndarray
    ans_embeds: Dict[str, List[np.ndarray]] = {m: [] for m in model_names}
    for m in model_names:
        for ans in answers[m]:
            cleaned = normalize_answer(ans)
            v = sim_encoder.encode(cleaned, normalize_embeddings=True)
            ans_embeds[m].append(v)

    # 4) Similarity-Statistik pro Modell + paarweise Modell-Ähnlichkeiten
    sims_per_model: Dict[str, List[float]] = {m: [] for m in model_names}
    pairwise_vals: Dict[Tuple[str, str], List[float]] = {
        (m1, m2): [] for m1, m2 in combinations(model_names, 2)
    }

    for i in range(N):
        for m1, m2 in combinations(model_names, 2):
            v1 = ans_embeds[m1][i]
            v2 = ans_embeds[m2][i]
            s = float(np.dot(v1, v2))  # Cosine-Sim wegen normalize_embeddings=True
            sims_per_model[m1].append(s)
            sims_per_model[m2].append(s)
            pairwise_vals[(m1, m2)].append(s)

    print("\n=== Similarity-Statistik (pro Modell) ===")
    for m in model_names:
        vals = np.array(sims_per_model[m], dtype=float)
        mean = float(vals.mean())
        std  = float(vals.std())
        print(f"{m:25s}  mean={mean:.4f}  std={std:.4f}")

    print("\n=== Paarweise Modell-Ähnlichkeiten (Mean Cosine Similarity über alle Fragen) ===")
    for (m1, m2), sims in pairwise_vals.items():
        vals = np.array(sims, dtype=float)
        mean = float(vals.mean())
        print(f"{m1:25s} vs {m2:25s}  mean={mean:.4f}")


if __name__ == "__main__":
    main()
