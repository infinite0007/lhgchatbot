from transformers import AutoTokenizer

# Pfad zu deinem HF-Modellordner (kann lokal sein oder aus HuggingFace Hub) End of Sentence (EOS)
# bei gguf Files funktioniert das hier nicht dann einfach das GGUF mit koboldcpp laden und dann erscheinen diese in der Console
model_path = r"C:\Users\lhglij1\OneDrive - Liebherr\Desktop\Master\lhgchatbot\FinetuneLLM\finetunedmodels\Falcon3-1B-Base-lora-unsloth-pirate-out\merged_model_for_gguf_convert"

tokenizer = AutoTokenizer.from_pretrained(model_path)

print("EOS Token:", tokenizer.eos_token)
print("EOS Token ID:", tokenizer.eos_token_id)
print("Pad Token:", tokenizer.pad_token)
print("Pad Token ID:", tokenizer.pad_token_id)

# Test: Wie wird der EOS-Token tokenisiert?
encoded = tokenizer(tokenizer.eos_token, add_special_tokens=False)
print("Tokenized EOS:", encoded["input_ids"])