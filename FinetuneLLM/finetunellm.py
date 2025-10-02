#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, json
from typing import Dict, Any, List
import torch
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig,
    TrainingArguments, Trainer, DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

# ------------------------------
# Konfiguration
# ------------------------------
BASE_MODEL = os.environ.get("BASE_MODEL", "../falcon-7b")  # lokaler Ordner oder "tiiuae/falcon-7b"
DATA_PATH  = os.environ.get("DATA_PATH", "datasets/ecommercefaq.json")  # .json (Array) oder .jsonl
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "finetunedmodels/falcon7b-lora-out")

USE_QLORA   = True     # 4-bit QLoRA
RANK        = 16
ALPHA       = 32
DROPOUT     = 0.05
MAX_SEQ_LEN = 2048
BATCH_SIZE  = 1
GRAD_ACCUM  = 8
EPOCHS      = 2
LR          = 2e-4
WARMUP      = 0.05
LOG_STEPS   = 10

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------------------
# Tokenizer
# ------------------------------
print(">> Lade Tokenizer…")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
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
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

# ------------------------------
# Modell
# ------------------------------
print(">> Lade Basismodell…")
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config if USE_QLORA else None,
    torch_dtype=torch.bfloat16 if not USE_QLORA else None,
    device_map="auto",
)

# Empfehlenswerte Trainings-Flags
model.config.use_cache = False  # wichtig bei Grad-Checkpointing
if hasattr(model, "gradient_checkpointing_enable"):
    model.gradient_checkpointing_enable()

# Für k-bit Training vorbereiten (fixiert LayerNorm etc.)
if USE_QLORA:
    model = prepare_model_for_kbit_training(model)

# ------------------------------
# PEFT / LoRA
# ------------------------------
lora_cfg = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=RANK,
    lora_alpha=ALPHA,
    lora_dropout=DROPOUT,
    target_modules=["query_key_value"],  # Falcon-7B
)
model = get_peft_model(model, lora_cfg)

# ------------------------------
# Daten laden & normalisieren (QA-Paare & Fallback "text")
# ------------------------------
def _load_any_json(path: str) -> List[Dict[str, Any]]:
    if path.endswith(".jsonl"):
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line: 
                    continue
                # falls die Datei aus Versehen ein Array enthält
                if line.startswith("[") and line.endswith("]"):
                    rows.extend(json.loads(line))
                else:
                    rows.append(json.loads(line))
        return rows
    else:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            assert isinstance(data, list), "dataset.json muss ein JSON-Array sein (oder .jsonl Zeilen)."
            return data

print(">> Lade Daten…")
raw = _load_any_json(DATA_PATH)
if not raw:
    raise RuntimeError("Dataset ist leer.")

# Erlaubte Paare (QA-Stil) ODER 'text'
pairs = [
    ("question", "answer"),
    ("instruction", "output"),
    ("input", "output"),
    ("prompt", "response"),
    ("src", "tgt"),
]

def to_text(ex: Dict[str, Any]) -> Dict[str, str] | None:
    # QA-Paare zuerst
    for qk, ak in pairs:
        q, a = ex.get(qk), ex.get(ak)
        if isinstance(q, str) and isinstance(a, str):
            return {"text": f"<human>: {q.strip()}\n<assistant>: {a.strip()}"}
    # Fallback: reines SFT-Textfeld
    if isinstance(ex.get("text"), str):
        return {"text": ex["text"]}
    return None

normalized = []
for ex in raw:
    item = to_text(ex)
    if item is not None:
        normalized.append(item)

if not normalized:
    raise RuntimeError("Keine passenden QA-Felder oder 'text' gefunden.")

ds = Dataset.from_list(normalized).shuffle(seed=42)
splits = ds.train_test_split(test_size=min(0.05, max(1/len(ds), 0.01)), seed=42)
dataset = DatasetDict({"train": splits["train"], "eval": splits["test"]})

# ------------------------------
# Tokenisierung für Causal LM
# ------------------------------
def _tok(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        max_length=MAX_SEQ_LEN,
        padding="max_length",
        return_attention_mask=True,
    )

print(">> Tokenisiere…")
tokenized = DatasetDict({
    "train": dataset["train"].map(_tok, batched=True, remove_columns=["text"]),
    "eval":  dataset["eval"].map(_tok,  batched=True, remove_columns=["text"]),
})

# DataCollator: erstellt labels = input_ids (mit -100 auf Padding)
dc = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# ------------------------------
# TrainingArguments (kompatibel)
# ------------------------------
print(">> Konfiguriere Training…")
training_args = TrainingArguments(
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    num_train_epochs=EPOCHS,
    learning_rate=LR,
    warmup_ratio=WARMUP,
    logging_steps=LOG_STEPS,
    save_steps=max(LOG_STEPS, 200),
    save_total_limit=2,
    bf16=True,
    optim="adamw_torch",        # robust; falls vorhanden kannst du später "paged_adamw_8bit" nehmen
    lr_scheduler_type="cosine",
    output_dir=OUTPUT_DIR,
    report_to=[],               # vermeidet alte "none"-Inkompatibilitäten
)

# ------------------------------
# Trainer
# ------------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    # eval_dataset=tokenized["eval"],  # bei Bedarf aktivieren, wenn dein HF das stabil unterstützt
    data_collator=dc,
    processing_class=tokenizer, # neu anstelle alt: tokenizer=tokenizer, ist gewollt.
)

# ------------------------------
# Train
# ------------------------------
print(">> Starte Training…")
trainer.train()

# ------------------------------
# Speichern
# ------------------------------
print(">> Speichere Adapter…")
trainer.model.save_pretrained(os.path.join(OUTPUT_DIR, "adapter"))
tokenizer.save_pretrained(OUTPUT_DIR)

# Optional: merge Adapter in Basismodell
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
