#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, json, math
from typing import Dict, Any, List
from datasets import load_dataset, Dataset, DatasetDict
import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig,
    TrainingArguments
)
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer, SFTConfig

# ------------------------------
# Konfiguration
# ------------------------------
BASE_MODEL = os.environ.get("BASE_MODEL", "../falcon-7b")
DATA_PATH  = os.environ.get("DATA_PATH", "dataset/ecommercefaq.jsonl")  # json array oder .jsonl
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "finetunedmodels/falcon7b-lora-out")
USE_QLORA  = True  # 4-bit QLoRA
RANK       = 16    # LoRA-Rank (8..64 gängig)
ALPHA      = 32
DROPOUT    = 0.05
MAX_SEQ_LEN= 2048  # Falcon-7B konfig; ggf. kleiner stellen bei wenig VRAM
BATCH_SIZE = 1
GRAD_ACCUM = 8
EPOCHS     = 2
LR         = 2e-4
WARMUP     = 0.05
LOG_STEPS  = 10

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------------------
# Tokenizer laden
# ------------------------------
print(">> Lade Tokenizer…")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ------------------------------
# 4-bit Quantisierung (QLoRA)
# ------------------------------
bnb_config = None
if USE_QLORA:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

# ------------------------------
# Basismodell laden
# ------------------------------
print(">> Lade Basismodell…")
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    trust_remote_code=True,
    quantization_config=bnb_config if USE_QLORA else None,
    torch_dtype=torch.bfloat16 if not USE_QLORA else None,
    device_map="auto"
)

# Empfohlen für PEFT/QLoRA
if hasattr(model, "gradient_checkpointing_enable"):
    model.gradient_checkpointing_enable()

# ------------------------------
# LoRA/PEFT konfigurieren
# Falcon: target_modules i.d.R. ["query_key_value"]
# ------------------------------
lora_cfg = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=RANK,
    lora_alpha=ALPHA,
    lora_dropout=DROPOUT,
    target_modules=["query_key_value"]
)
model = get_peft_model(model, lora_cfg)

# ------------------------------
# Datensatz laden & normalisieren
# Unterstützt:
# - JSON-Lines (.jsonl): pro Zeile ein Objekt
# - JSON-Array (.json): Liste von Objekten
# Erwartete Felder (irgendeines der Paare):
#   (question, answer) | (instruction, output) | (input, output) | (prompt, response)
# ------------------------------
def _detect_qa_keys(ex: Dict[str, Any]) -> Dict[str, str]:
    candidates = [
        ("question", "answer"),
        ("instruction", "output"),
        ("input", "output"),
        ("prompt", "response"),
        ("src", "tgt"),
    ]
    for qk, ak in candidates:
        if qk in ex and ak in ex and isinstance(ex[qk], str) and isinstance(ex[ak], str):
            return {"q": qk, "a": ak}
    # Fallback: wenn es nur "text" gibt -> Einzeiler-SFT
    if "text" in ex and isinstance(ex["text"], str):
        return {"q": "text", "a": None}
    return {"q": None, "a": None}

def _load_any_json(path: str) -> List[Dict[str, Any]]:
    if path.endswith(".jsonl"):
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line=line.strip()
                if not line: continue
                rows.append(json.loads(line))
        return rows
    else:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            assert isinstance(data, list), "dataset.json muss ein JSON-Array oder .jsonl sein."
            return data

print(">> Lade Daten…")
raw = _load_any_json(DATA_PATH)
if not raw:
    raise RuntimeError("Dataset ist leer.")

keys = _detect_qa_keys(raw[0])
if keys["q"] is None:
    raise RuntimeError("Konnte keine passenden Felder finden. Erwartet z.B. (question/answer) oder (instruction/output).")

def build_prompt(ex: Dict[str, Any]) -> str:
    if keys["a"] is None:
        # Einzeiler (rein SFT): Model lernt von vollständigen Beispielen
        return ex[keys["q"]]
    user = ex[keys["q"]].strip()
    assistant = ex[keys["a"]].strip()
    # simples, modellagnostisches Chat-Format
    return f"<human>: {user}\n<assistant>: {assistant}"

normalized = [{"text": build_prompt(ex)} for ex in raw if isinstance(ex.get(keys["q"], ""), str)]

ds = Dataset.from_list(normalized)
ds = ds.shuffle(seed=42)
splits = ds.train_test_split(test_size=min(0.05, max(1/len(ds), 0.01)), seed=42)
dataset = DatasetDict({"train": splits["train"], "eval": splits["test"]})

# ------------------------------
# TRL SFTTrainer – packt effizient und ist SOTA in HuggingFace-Ökosystem
# ------------------------------
print(">> Konfiguriere Training…")
# Achtung: 'packing=True' concateniert Beispiele bis MAX_SEQ_LEN -> bessere Auslastung
sft_config = SFTConfig(
    max_seq_length=MAX_SEQ_LEN,
    packing=True,
)

training_args = TrainingArguments(
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    num_train_epochs=EPOCHS,
    learning_rate=LR,
    warmup_ratio=WARMUP,
    logging_steps=LOG_STEPS,
    evaluation_strategy="steps",
    eval_steps=max(LOG_STEPS, 50),
    save_strategy="steps",
    save_steps=max(LOG_STEPS, 200),
    save_total_limit=2,
    bf16=True,                         # bfloat16 (auch bei QLoRA ok, compute_dtype bfloat16)
    optim="paged_adamw_8bit",          # Memory-sparend; klassischer QLoRA-Default
    lr_scheduler_type="cosine",
    gradient_checkpointing=True,
    output_dir=OUTPUT_DIR,
    report_to="none"
)

# Optional Speed-Stack (Unsloth) – auskommentiert lassen, wenn nicht installiert
# from unsloth import FastLanguageModel
# model = FastLanguageModel.wrap_peft_model(model)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["eval"],
    args=training_args,
    tokenizer=tokenizer,
    peft_config=lora_cfg,
    dataset_text_field="text",
    sft_config=sft_config,
)

print(">> Starte Training…")
trainer.train()

print(">> Speichere Adapter…")
trainer.model.save_pretrained(os.path.join(OUTPUT_DIR, "adapter"))
tokenizer.save_pretrained(OUTPUT_DIR)

# Optional: Merge Adapter in das Basismodell (für einfaches Deployment ohne PEFT)
# Achtung: erhöht Dateigröße & VRAM-Bedarf zur Inferenz.
try:
    merged_path = os.path.join(OUTPUT_DIR, "merged-model")
    os.makedirs(merged_path, exist_ok=True)
    merged = trainer.model.merge_and_unload()
    merged.save_pretrained(merged_path, safe_serialization=True)
    tokenizer.save_pretrained(merged_path)
    print(f">> Fertig. Gemergtes Modell unter: {merged_path}")
except Exception as e:
    print(f"Merge übersprungen (optional). Grund: {e}")

print(">> Done.")
