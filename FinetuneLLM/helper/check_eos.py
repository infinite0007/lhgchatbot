from transformers import AutoTokenizer

# Pfad zu deinem HF-Modellordner (kann lokal sein oder aus HuggingFace Hub) End of Sentence (EOS)
model_path = "finetunedmodels/Falcon3-1B-Base-lora-pirate-out/merged_model"

tokenizer = AutoTokenizer.from_pretrained(model_path)

print("EOS Token:", tokenizer.eos_token)
print("EOS Token ID:", tokenizer.eos_token_id)
print("Pad Token:", tokenizer.pad_token)
print("Pad Token ID:", tokenizer.pad_token_id)

# Test: Wie wird der EOS-Token tokenisiert?
encoded = tokenizer(tokenizer.eos_token, add_special_tokens=False)
print("Tokenized EOS:", encoded["input_ids"])