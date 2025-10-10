#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Dieses File kann am Ende mit Hilfe unsloth als GGUF konvertiert werden. Siehe: https://docs.unsloth.ai/get-started/beginner-start-here Aber alles was man braucht ist pip install unsloth, das cuda toolkit installiert, torch mit cuda version installiert
# If you have Huggingface - SSL errors install: pip install python-certifi-win32
# https://github.com/ggml-org/llama.cpp bei release schauen dass man die convert.py Dateien hat und die .exe sonst von build/bin in llama.cpp kopieren. Also die einen Files sind in Main die anderen in Release halt zusammen suchen in root kopieren dann
# Nachdem dieses Skript aufgerufen wurde navigiert man in den Ordner merged_model_prepared_gguf und kopiert dessen Link
# clone normales llama.cpp mit: git clone https://github.com/ggml-org/llama.cpp.git     darin befinden sich die Skripte - falls man die .exen will für Server oder Chat einfach das Release runterladen.
# geht dann mit cd in llama.cpp und ruft auf: python convert_hf_to_gguf.py "C:\Users\lhglij1\OneDrive - Liebherr\Desktop\Master\lhgchatbot\FinetuneLLM\finetunedmodels\Falcon3-1B-Base-lora-pirate-origin-out\ggufmerged" --outfile name.gguf
# das kann man direkt benutzen und zum Beispiel mit koboldcpp starten (das File wird im llama.cpp gespeichert)
from unsloth import FastLanguageModel

# Danach erst TRL / SFTTrainer
from trl import SFTTrainer, SFTConfig
import os, json
os.environ["TRITON_CACHE_DIR"] = r"C:\triton_cache" # verhindert Speichererrors bei triton
os.environ["TORCHINDUCTOR_CACHE_DIR"] = r"C:\triton_cache"
from typing import Dict, Any, List, Optional
from datasets import Dataset, DatasetDict

# transformers/trl bleiben für Dataprep & Trainer-Args – das Modell kommt aber aus Unsloth
from transformers import (
    DataCollatorForLanguageModeling,
    TrainingArguments,
)

# [UNSLOTH] Neu: Unsloth-Imports
from unsloth import FastLanguageModel

# ------------------------------
# Konfiguration
# ------------------------------
BASE_MODEL = os.environ.get("BASE_MODEL", "../Falcon3-1B-Base")
DATA_PATH = os.environ.get("DATA_PATH", "datasets/trainpirate.jsonl")  # unterstützt .json oder .jsonl
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "finetunedmodels/Falcon3-1B-Base-lora-unsloth-pirate-out")

USE_QLORA = True    # 4-bit QLoRA Aktiviert die QLoRA-Methode (Low-Rank Adapters + 4-Bit Quantisierung) / Wenn du das ausschaltest, trainierst du klassisch auf ganzen Modellgewichten; das braucht deutlich mehr Speicher und Rechenzeit
RANK = 16           # Die Rang-Dimension der LoRA-Adapter — d.h. statt ganze Gewichtsmatri­zen zu trainieren, trainierst du zwei kleinere „low-rank“ Matrizen der Dimension rank. Je höher der Rang, desto mehr Anpassungsfähigkeit haben die Adapter. / hoch = Mehr Kapazität, das Modell kann komplexere Anpassungen lernen / nieder = Weniger Ausdrucksfähigkeit, der Adapter ist beschränkter / Wenn zu hoch, kann es Overfitting begünstigen oder Speicher/RAM-Bedarf wachsen
ALPHA = 32          # Ein Skalierungsfaktor, der bestimmt, wie stark die Anpassungen (Adapter) in das Basis-Modell eingehen. Oft wird alpha = rank * 2 als Heuristik verwendet. / hoch = Adapter wirkt stärker — größere Modifikation des Modells / nieder = Adapter-Effekt wird schwächer, das Fine-Tuning „flüsternder“ / Wenn alpha zu groß im Verhältnis zum rank, kann es zu Instabilität oder Übersteuerung kommen
DROPOUT = 0.05      # Wahrscheinlichkeit, mit der man zufällig Verbindungen (Parameter) im Adapter während des Trainings abschaltet — Werkzeug zur Regularisierung, Verhinderung von Overfitting / hoch = Mehr Regularisierung, das Modell wird robuster gegen Überanpassung / nieder = Weniger Regularisierung, das Modell könnte zu stark auf Trainingsdaten lernen / Wenn zu hoch, kann Lernen gestört werden / Modell erhält zu wenige Signale
MAX_SEQ_LEN = 2048  # Maximale Länge (in Tokens) eines Eingabe- oder Prompt+Antwortkontexts, der in das Modell gegeben wird (Trunkierung, Padding etc.) / hoch = Längere Kontexte erlauben mehr Information, aber benötigen mehr Speicher & Rechenzeit / nieder = Kürzere Kontexte können Informationen abschneiden / das Modell kann weniger „sehen“ / Wenn du zu lang setzt, kann der Speicher überlaufen oder das Training extrem langsam werden
BATCH_SIZE = 1      # Anzahl der Beispiele pro Batch, die gleichzeitig verarbeitet werden (Gradienten werden über den Batch gemittelt) / hoch = Glattere Gradienten, effizientere Nutzung der GPU / nieder = Rauschigere Gradienten, langsameres Konvergieren / Wenn zu groß, OOM (Out-of-Memory); zu klein => instabile Updates, starkes Rauschen
GRAD_ACCUM = 8      # Gradient Accumulation Steps: du sammelst Gradienten über mehrere kleine Batches, bevor du ein Update machst — simuliert einen größeren Batch, wenn GPU nicht genug Platz bietet / hoch = Ermöglicht effektiv größeren Batch bei limitierter Hardware / nieder = Verlängert die Zeit bis zum Update (weniger Updates pro Zeiteinheit) / Wenn zu groß, kann Training ineffizient werden oder Instabilitäten auftreten
EPOCHS = 3.0        # Anzahl der Durchläufe über das komplette Trainingsset / hoch = Mehr Lernen möglich, Modell hat mehr Gelegenheit, Muster zu extrahieren / nieder = Risiko von Overfitting, Training dauert länger / Wenn zu viele Epochen, kann das Modell zu stark memorisieren
LR = 2e-4           # Schrittweite, mit der Modellgewichte in Richtung des Gradienten verschoben werden / hoch = Schnellere Anpassung / schnelleres Lernen (sofern stabil) / nieder = Wenn zu hoch, kann das Training instabil werden, Loss explodieren / Wenn zu niedrig, dauert Lernen sehr lange oder bleibt im Lokalminimum stecken
WARMUP = 0.05       # Anteil (Ratio) der Trainingsschritte, in denen die Lernrate von 0 (oder niedrigem Wert) linear auf die volle LR hochgefahren wird. Dies stabilisiert frühe Updates. / hoch = Zu kurzer Warmup: riskante große Updates zu früh / nieder = Zu langer Warmup: verschwendete Schritte mit zu kleinen Updates / Ein guter Warmup hilft, dass das Modell stabil lernt zu Beginn
LOG_STEPS = 10      # Nach wie vielen Update-Schritten (oder Batches) protokolliert / geloggt wird / hoch = Häufigeres Logging — du siehst den Fortschritt feiner / nieder = Weniger Logging — Übersichtlicher, weniger Overhead / Wenn zu häufig geloggt wird, kann Logging Overhead das Training bremsen

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------------------
# Modell + Tokenizer via Unsloth laden (VOR Datenerstellung, damit tokenizer.eos_token verfügbar ist)
# ------------------------------
print(">> Lade Basismodell mit Unsloth …")
model, tokenizer = FastLanguageModel.from_pretrained(
    BASE_MODEL,
    max_seq_length=MAX_SEQ_LEN,
    load_in_4bit=USE_QLORA,     # QLoRA an/aus
)

# [UNSLOTH] – spezielle Tokens sicherstellen (du hattest EOS=<|endoftext|>, PAD=<|pad|>)
if tokenizer.pad_token is None:
    # Wenn dein Tokenizer das Pad-Token kennt, setzt Unsloth es korrekt;
    # falls nicht vorhanden, setzen wir es auf eos (oder füge es explizit hinzu)
    tokenizer.pad_token = tokenizer.eos_token

# IDs einmalig hart setzen, damit sie in Adapter + Merge landen
model.config.eos_token_id = tokenizer.eos_token_id
model.config.pad_token_id = tokenizer.pad_token_id

# ------------------------------
# Daten laden & Prompt-Format
# ------------------------------
def _load_any_json(path: str) -> List[Dict[str, Any]]:
    """Liest JSON/JSONL in eine Liste von Dicts."""
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
            return f"<human>: {q}\n<assistant>: {a}{tokenizer.eos_token}"

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
                parts.append(f"<assistant>: {content}{tokenizer.eos_token}")
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

ds = Dataset.from_list(normalized).shuffle(seed=42)
splits = ds.train_test_split(test_size=min(0.05, max(1/len(ds), 0.01)), seed=42)
dataset = DatasetDict({"train": splits["train"], "eval": splits["test"]})

# ------------------------------
# LoRA / Adapter konfigurieren – über Unsloth (wrappt PEFT korrekt)
# ------------------------------
print(">> Rüste LoRA-Adapter aus …")
model = FastLanguageModel.get_peft_model(
    model,
    r=RANK,
    lora_alpha=ALPHA,
    lora_dropout=DROPOUT,
    #target_modules="all-linear", # klappt eigentlich bei allen - falls Fehler gerne in ./helpe/get_target_modules.py die targets holen und die dann benutzen
    target_modules = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "mlp.down_proj",
        "mlp.gate_proj",
        "mlp.up_proj"
    ],
    bias="none",
)

# ------------------------------
# Tokenisierung
# ------------------------------
def tokenize_fn(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        padding=False,          # statt "max_length" da sonst hinten eos abgeschnitten werden kann
        max_length=MAX_SEQ_LEN,
    )

print(">> Tokenisiere …")
tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
dc = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# ------------------------------
# Training (TRL SFTTrainer + Unsloth)
# ------------------------------
print(">> Konfiguriere Training …")
# Du kannst dein altes TrainingArguments behalten; SFTConfig ist bequemer
sft_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    num_train_epochs=EPOCHS,
    learning_rate=LR,
    warmup_ratio=WARMUP,
    logging_steps=LOG_STEPS,
    save_strategy="steps",
    save_steps=200,
    save_total_limit=2,
    bf16=True,                    # bfloat16 wenn möglich
    optim="paged_adamw_8bit",
    lr_scheduler_type="cosine",
    report_to=[],
    remove_unused_columns=False,
    dataset_num_proc=1,
    eos_token=tokenizer.eos_token,  # wichtig!
    pad_token=tokenizer.pad_token,
    # packing/add_eos_token werden hier bewusst nicht geändert
)

trainer = SFTTrainer(
    model=model,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["eval"],
    data_collator=dc,
    args=sft_args,
    processing_class=tokenizer,   # statt tokenizer=tokenizer ist nämlich alt
)

# ------------------------------
# Trainieren
# ------------------------------
print(">> Starte Training …")
trainer.train()

# (Optional, gut für Inferenz-Stabilität/KV-Cache – kann man drinlassen)
from unsloth import FastLanguageModel as _FLM_
_FLM_.for_inference(model)

# ------------------------------
# Speichern: Adapter, Merged, GGUF
# ------------------------------
print(">> Speichere Adapter …")
adapter_dir = os.path.join(OUTPUT_DIR, "adapter")
os.makedirs(adapter_dir, exist_ok=True)
model.save_pretrained(adapter_dir)      # [UNSLOTH] Adapter-Delta
tokenizer.save_pretrained(adapter_dir)

print(">> Merge Adapter und speichere vollständiges Modell (16-bit) …")
merged_dir = os.path.join(OUTPUT_DIR, "merged_model_for_gguf_convert")
os.makedirs(merged_dir, exist_ok=True)
# [UNSLOTH] sauberer Merge – erzeugt FP16/BF16-Gewichte, die sich zuverlässig in GGUF konvertieren lassen
# das gespeicherte Modell kann dann nun mit llama.cpp mittels convert hf to gguf umgewandelt werden und klappt auch. In llama.cpp: python convert_hf_to_gguf.py "C:\XXX\Master\lhgchatbot\FinetuneLLM\finetunedmodels\ModelName\merged_model" --outfile modelname.gguf 
model.save_pretrained_merged(
    merged_dir,
    tokenizer,
    save_method="merged_16bit",  # Alternativ: "merged_4bit" o.ä., aber für GGUF brauchen wir 16-bit als Quelle
)

print("Training + Speichern abgeschlossen.")