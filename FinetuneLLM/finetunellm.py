#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, json
from typing import Dict, Any, List, Optional
import torch
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig,
    TrainingArguments, Trainer
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

# ------------------------------
# Konfiguration
# ------------------------------
BASE_MODEL = os.environ.get("BASE_MODEL", "../Falcon3-1B-Base")
DATA_PATH = os.environ.get("DATA_PATH", "datasets/trainpirate.jsonl")  # unterstützt .json oder .jsonl
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "finetunedmodels/Falcon3-1B-Base-lora-pirate-out")

USE_QLORA = True    # 4-bit QLoRA Aktiviert die QLoRA-Methode (Low-Rank Adapters + 4-Bit Quantisierung) / Wenn du das ausschaltest, trainierst du klassisch auf ganzen Modellgewichten; das braucht deutlich mehr Speicher und Rechenzeit
RANK = 16           # Die Rang-Dimension der LoRA-Adapter — d.h. statt ganze Gewichtsmatri­zen zu trainieren, trainierst du zwei kleinere „low-rank“ Matrizen der Dimension rank. Je höher der Rang, desto mehr Anpassungsfähigkeit haben die Adapter. / hoch = Mehr Kapazität, das Modell kann komplexere Anpassungen lernen / nieder = Weniger Ausdrucksfähigkeit, der Adapter ist beschränkter / Wenn zu hoch, kann es Overfitting begünstigen oder Speicher/RAM-Bedarf wachsen
ALPHA = 32          # Ein Skalierungsfaktor, der bestimmt, wie stark die Anpassungen (Adapter) in das Basis-Modell eingehen. Oft wird alpha = rank * 2 als Heuristik verwendet. / hoch = Adapter wirkt stärker — größere Modifikation des Modells / nieder = Adapter-Effekt wird schwächer, das Fine-Tuning „flüsternder“ / Wenn alpha zu groß im Verhältnis zum rank, kann es zu Instabilität oder Übersteuerung kommen
DROPOUT = 0.05      # Wahrscheinlichkeit, mit der man zufällig Verbindungen (Parameter) im Adapter während des Trainings abschaltet — Werkzeug zur Regularisierung, Verhinderung von Overfitting / hoch = Mehr Regularisierung, das Modell wird robuster gegen Überanpassung / nieder = Weniger Regularisierung, das Modell könnte zu stark auf Trainingsdaten lernen / Wenn zu hoch, kann Lernen gestört werden / Modell erhält zu wenige Signale
MAX_SEQ_LEN = 2048  # Maximale Länge (in Tokens) eines Eingabe- oder Prompt+Antwortkontexts, der in das Modell gegeben wird (Trunkierung, Padding etc.) / hoch = Längere Kontexte erlauben mehr Information, aber benötigen mehr Speicher & Rechenzeit / nieder = Kürzere Kontexte können Informationen abschneiden / das Modell kann weniger „sehen“ / Wenn du zu lang setzt, kann der Speicher überlaufen oder das Training extrem langsam werden
BATCH_SIZE = 1      # Anzahl der Beispiele pro Batch, die gleichzeitig verarbeitet werden (Gradienten werden über den Batch gemittelt) / hoch = Glattere Gradienten, effizientere Nutzung der GPU / nieder = Rauschigere Gradienten, langsameres Konvergieren / Wenn zu groß, OOM (Out-of-Memory); zu klein => instabile Updates, starkes Rauschen
GRAD_ACCUM = 8      # Gradient Accumulation Steps: du sammelst Gradienten über mehrere kleine Batches, bevor du ein Update machst — simuliert einen größeren Batch, wenn GPU nicht genug Platz bietet / hoch = Ermöglicht effektiv größeren Batch bei limitierter Hardware / nieder = Verlängert die Zeit bis zum Update (weniger Updates pro Zeiteinheit) / Wenn zu groß, kann Training ineffizient werden oder Instabilitäten auftreten
EPOCHS = 5.0        # Anzahl der Durchläufe über das komplette Trainingsset / hoch = Mehr Lernen möglich, Modell hat mehr Gelegenheit, Muster zu extrahieren / nieder = Risiko von Overfitting, Training dauert länger / Wenn zu viele Epochen, kann das Modell zu stark memorisieren
LR = 2e-4           # Schrittweite, mit der Modellgewichte in Richtung des Gradienten verschoben werden / hoch = Schnellere Anpassung / schnelleres Lernen (sofern stabil) / nieder = Wenn zu hoch, kann das Training instabil werden, Loss explodieren / Wenn zu niedrig, dauert Lernen sehr lange oder bleibt im Lokalminimum stecken
WARMUP = 0.05       # Anteil (Ratio) der Trainingsschritte, in denen die Lernrate von 0 (oder niedrigem Wert) linear auf die volle LR hochgefahren wird. Dies stabilisiert frühe Updates. / hoch = Zu kurzer Warmup: riskante große Updates zu früh / nieder = Zu langer Warmup: verschwendete Schritte mit zu kleinen Updates / Ein guter Warmup hilft, dass das Modell stabil lernt zu Beginn
LOG_STEPS = 10      # Nach wie vielen Update-Schritten (oder Batches) protokolliert / geloggt wird / hoch = Häufigeres Logging — du siehst den Fortschritt feiner / nieder = Weniger Logging — Übersichtlicher, weniger Overhead / Wenn zu häufig geloggt wird, kann Logging Overhead das Training bremsen

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------------------
# Tokenizer
# ------------------------------
print(">> Lade Tokenizer …")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ------------------------------
# Quantisierung (QLoRA)
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
# Basismodell laden
# ------------------------------
print(">> Lade Basismodell …")
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    trust_remote_code=True,
    quantization_config=bnb_config if USE_QLORA else None,
    torch_dtype=torch.bfloat16 if not USE_QLORA else None,
    device_map="auto",
)
model.config.use_cache = False
if hasattr(model, "gradient_checkpointing_enable"):
    model.gradient_checkpointing_enable()

if USE_QLORA:
    model = prepare_model_for_kbit_training(model)

# ------------------------------
# LoRA / Adapter konfigurieren
# ------------------------------
lora_cfg = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=RANK,
    lora_alpha=ALPHA,
    lora_dropout=DROPOUT,
    target_modules="all-linear",
    bias="none",
)
model = get_peft_model(model, lora_cfg)

# ------------------------------
# Daten laden (JSON oder JSONL)
# ------------------------------
def _load_any_json(path: str) -> List[Dict[str, Any]]:
    if path.endswith(".jsonl"):
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        return rows
    else:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            assert isinstance(data, list), "JSON muss Liste von Objekten sein"
            return data

print(">> Lade Daten …")
raw = _load_any_json(DATA_PATH)
if not raw:
    raise RuntimeError("Dataset ist leer")

# Schlüsselpaare, die möglich sind
pairs = [
    ("question", "answer"),
    ("instruction", "output"),
    ("input", "output"),
    ("prompt", "response"),
    ("src", "tgt"),
]

def to_prompt_text(ex: Dict[str, Any]) -> Optional[str]:
    # 1. Klassisches QA-Format
    for qk, ak in pairs:
        if qk in ex and ak in ex and isinstance(ex[qk], str) and isinstance(ex[ak], str):
            q = ex[qk].strip()
            a = ex[ak].strip()
            return f"<human>: {q}\n<assistant>: {a}"

    # 2. Nachrichten-Format („messages“) erkennen
    msgs = ex.get("messages")
    if isinstance(msgs, list):
        parts = []
        for m in msgs:
            role = m.get("role")
            content = m.get("content")
            if not isinstance(content, str):
                continue
            content = content.strip()
            if role == "user":
                parts.append(f"<human>: {content}")
            elif role == "assistant":
                parts.append(f"<assistant>: {content}")
            else:
                parts.append(f"<system>: {content}")
        return "\n".join(parts)

    # 3. Fallback auf gemeinsames Textfeld
    if "text" in ex and isinstance(ex["text"], str):
        return ex["text"]

    return None

normalized = []
for ex in raw:
    txt = to_prompt_text(ex)
    if txt is not None:
        normalized.append({"text": txt})

if len(normalized) == 0:
    raise RuntimeError("Keine gültigen Prompt-Beispiele gefunden")

from datasets import Dataset
ds = Dataset.from_list(normalized).shuffle(seed=42)
splits = ds.train_test_split(test_size=min(0.05, max(1/len(ds), 0.01)), seed=42)
dataset = DatasetDict({"train": splits["train"], "eval": splits["test"]})

# ------------------------------
# Tokenisierung
# ------------------------------
def tokenize_fn(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=MAX_SEQ_LEN,
    )

print(">> Tokenisiere …")
tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])

from transformers import DataCollatorForLanguageModeling
dc = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# ------------------------------
# TrainingArguments
# ------------------------------
print(">> Konfiguriere Training …")
training_args = TrainingArguments(
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    num_train_epochs=EPOCHS,
    learning_rate=LR,
    warmup_ratio=WARMUP,
    logging_steps=LOG_STEPS,
    save_steps=200,
    save_total_limit=2,
    bf16=True,
    optim="paged_adamw_8bit",
    lr_scheduler_type="cosine",
    output_dir=OUTPUT_DIR,
    report_to=[],
    remove_unused_columns=False,
)

# ------------------------------
# Trainer
# ------------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["eval"],
    data_collator=dc,
    tokenizer=tokenizer,
)

# ------------------------------
# Trainieren
# ------------------------------
print(">> Starte Training …")
trainer.train()

# ------------------------------
# Speichern
# ------------------------------
print(">> Speichere Adapter …")
model.save_pretrained(os.path.join(OUTPUT_DIR, "adapter"))
tokenizer.save_pretrained(OUTPUT_DIR)

print(">> Fertig.")
