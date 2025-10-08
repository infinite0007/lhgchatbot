#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Conv1D
import bitsandbytes as bnb

def get_candidate_modules(model):
    """
    Gibt Namen der Module zurück, die für LoRA / Adapter in Frage kommen.
    Erkennt Module vom Typ:
     - torch.nn.Linear
     - transformers.Conv1D (häufig in GPT / Falcon / LLaMA)
     - bitsandbytes.nn.Linear4bit
    """
    names = set()
    for full_name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) \
           or isinstance(module, Conv1D) \
           or isinstance(module, bnb.nn.Linear4bit):
            names.add(full_name)
    return sorted(names)

if __name__ == "__main__":
    # Beispiel: HF-Modell oder lokaler Pfad
    model_name = r"C:\Users\lhglij1\OneDrive - Liebherr\Desktop\Master\lhgchatbot\Falcon3-1B-Base"
    print("Lade Modell:", model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.float16, low_cpu_mem_usage=True)
    print("Modell geladen.")
    cand = get_candidate_modules(model)
    print("Gefundene Module für LoRA/Adapter (Kandidaten):")
    for n in cand:
        print("  ", n)
