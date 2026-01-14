#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Vergleicht zwei Chat-Modelle (z.B. Qwen-Base vs Qwen-Instruct)
auf einem festen Fragenkatalog.

- Keine RAG-Pipeline, nur direkte Chat-Prompts im Format:
  <human>: {question}
  <assistant>:

- Lädt nacheinander Modell A und Modell B (speicherschonend),
  generiert Antworten für alle Fragen und speichert alles in einer JSONL-Datei.

- Ausgabe: Für jede Frage werden Frage + Antwort von Modell A und B
  in der Konsole angezeigt.
"""

import json
from pathlib import Path
from typing import Dict, List

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# ------------------------------
# KONFIGURATION
# ------------------------------

# Modelle: HIER anpassen je Run
# Beispiel 1: roh Base vs roh Instruct
MODEL_A_NAME = "Qwen2.5-3B-Base (finetuned)"
MODEL_A_PATH = "../FinetuneLLM/finetunedmodels/Qwen2.5-3B-Base-lora-unsloth-liebherr-1ep_var5-out/merged_model_for_gguf_convert"

MODEL_B_NAME = "Qwen2.5-3B-Instruct (finetuned)"
MODEL_B_PATH = "../FinetuneLLM/finetunedmodels/Qwen2.5-3B-Instruct-lora-unsloth-liebherr-1ep_var5-out/merged_model_for_gguf_convert"

# Beispiel 2 (später):
# MODEL_A_NAME = "Qwen2.5-3B-Base (ft-e1-v5)"
# MODEL_A_PATH = "FinetuneLLM/qwen_base_e1_var5"
# MODEL_B_NAME = "Qwen2.5-3B-Instruct (ft-e1-v5)"
# MODEL_B_PATH = "FinetuneLLM/qwen_instruct_e1_var5"

OUT_FILE = "compare_two_models_finetuned_qwen_raw.jsonl"  # Dateiname anpassen je Run

# Generierungs-Parameter
MAX_NEW_TOKENS   = 400
EVAL_TEMPERATURE = 0.0  # deterministisch
TEMPLATE = "<human>: {user}\n<assistant>:"

# Fragenkatalog (deine 27 Fragen)
QUESTIONS: List[str] = [
    "Please explain me the basic architecture of the Powerboard Software.",
    "What is the purpose of sabbath mode?",
    "What is the ToD and what is it used for?",
    "What is the purpose of the LPF?",
    "What is a Snapshot and for what is it used?"
]


# ------------------------------
# HELFERFUNKTIONEN
# ------------------------------

def load_model_and_pipeline(model_path: str):
    """Lädt Tokenizer + Modell + HF-Pipeline für ein Chat-Modell."""
    print(f"\n[Lade Modell] {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,     # bei CPU ggf. float32 nutzen
        device_map="auto",
    )

    gen = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
    )
    return gen, tokenizer


def generate_answer(gen, tokenizer, question: str) -> str:
    """Erzeugt eine Antwort auf eine Frage im <human>/<assistant>-Format."""
    prompt = TEMPLATE.format(user=question)

    outputs = gen(
        prompt,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,                 # deterministisch
        temperature=EVAL_TEMPERATURE,
        top_p=1.0,
        top_k=0,
        return_full_text=True,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    full_text = outputs[0]["generated_text"]
    if "<assistant>:" in full_text:
        answer = full_text.split("<assistant>:")[1].strip()
    else:
        answer = full_text.strip()
    return answer


# ------------------------------
# HAUPTLOGIK
# ------------------------------

def main():
    out_path = Path(OUT_FILE)
    records: List[Dict] = []

    # 1) Modell A beantworten lassen
    gen_a, tok_a = load_model_and_pipeline(MODEL_A_PATH)
    answers_a: Dict[int, str] = {}

    for qid, q in enumerate(QUESTIONS, start=1):
        print(f"\n[{MODEL_A_NAME}] Frage {qid}/{len(QUESTIONS)}")
        print("Q:", q)
        ans = generate_answer(gen_a, tok_a, q)
        answers_a[qid] = ans

    # GPU-Speicher aufräumen
    del gen_a
    torch.cuda.empty_cache()

    # 2) Modell B beantworten lassen
    gen_b, tok_b = load_model_and_pipeline(MODEL_B_PATH)
    answers_b: Dict[int, str] = {}

    for qid, q in enumerate(QUESTIONS, start=1):
        print(f"\n[{MODEL_B_NAME}] Frage {qid}/{len(QUESTIONS)}")
        print("Q:", q)
        ans = generate_answer(gen_b, tok_b, q)
        answers_b[qid] = ans

    del gen_b
    torch.cuda.empty_cache()

    # 3) Konsolidierte Ausgabe + JSONL
    print("\n==================== VERGLEICH PRO FRAGE ====================\n")

    for qid, q in enumerate(QUESTIONS, start=1):
        a_ans = answers_a.get(qid, "")
        b_ans = answers_b.get(qid, "")

        print(f"========== Frage {qid} ==========")
        print("Frage:")
        print(q)
        print("\n---", MODEL_A_NAME, "---")
        print(a_ans)
        print("\n---", MODEL_B_NAME, "---")
        print(b_ans)
        print("=================================\n")

        rec = {
            "qid": qid,
            "question": q,
            "model_a_name": MODEL_A_NAME,
            "model_a_path": MODEL_A_PATH,
            "model_a_answer": a_ans,
            "model_b_name": MODEL_B_NAME,
            "model_b_path": MODEL_B_PATH,
            "model_b_answer": b_ans,
        }
        records.append(rec)

    # JSONL speichern
    with out_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\nFertig. Ergebnisse gespeichert in: {out_path.resolve()}")


if __name__ == "__main__":
    main()
