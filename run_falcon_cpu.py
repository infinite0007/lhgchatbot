# run_infer.py
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import PeftModel

# ------------------------------
# Initialisierung
# ------------------------------
base_model_name = "./Falcon3-1B-Base"
adapters_path   = "FinetuneLLM/finetunedmodels/Falcon3-1B-Base-lora-ecommerce-out/adapter"

print("Lade Tokenizer ...")
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Lade Modell ...")
model = AutoModelForCausalLM.from_pretrained(
    base_model_name,

    # CPU-----------------------
    # dtype=torch.float32,
    # device_map="cpu",            # erzwingt CPU-Ladevorgang über Accelerate

    # GPU-----------------------
    dtype=torch.bfloat16,  # oder torch.float16 da effizienter bei gpu
    device_map="auto",     # automatisch auf GPU laden bei auto
)

print("Lade Adapter ...")
model = PeftModel.from_pretrained(model, adapters_path)

# Pipeline
gen = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
)

# Template
TEMPLATE = "<human>: {user}\n<assistant>:"

# ------------------------------
# Interaktiv
# ------------------------------
while True:
    user_input = input("\nBitte Prompt eingeben (oder 'exit'): ").strip()
    if user_input.lower() in ("exit", "quit"):
        print("Beende.")
        break

    prompt = TEMPLATE.format(user=user_input)

    # Inferenz
    outputs = gen(
        prompt,
        max_new_tokens=200,
        do_sample=False,
        temperature=0.0,
        top_p=0.7,
        return_full_text=True,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    # Ausgabe extrahieren
    full_text = outputs[0]["generated_text"]
    if "<assistant>:" in full_text:
        answer = full_text.split("<assistant>:")[1].strip()
    else:
        answer = full_text.strip()

    print("\nAntwort:\n", answer)
