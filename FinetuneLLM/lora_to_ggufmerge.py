# If you have SSL errors install: pip install python-certifi-win32
# https://github.com/ggml-org/llama.cpp bei release schauen dass man die convert.py Dateien hat und die .exe sonst von build/bin in llama.cpp kopieren. Also die einen Files sind in Main die anderen in Release halt zusammen suchen in root kopieren dann
# Nachdem dieses Skript mit python lora_to_ggufmerge.py aufgerufen wurde navigiert man in den Ordner und kopiert dessen Pfad (Mit normalem merged-model hätte es nicht funktioniert da etwas Verloren geht für llama deswegen übernimmt das nun unsloth)
# geht dann mit cd in llama.cpp und ruft auf: python convert_hf_to_gguf.py "C:\Users\lhglij1\OneDrive - Liebherr\Desktop\Master\lhgchatbot\FinetuneLLM\finetunedmodels\Falcon3-1B-Base-lora-pirate-origin-out\ggufmerged" --outfile name.gguf
# das kann man direkt benutzen und zum Beispiel mit koboldcpp starten

from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    "finetunedmodels/Falcon3-1B-Base-lora-pirate-origin-out/adapter",  # dein Adapter-Ordner oder gemergtes Modell
    load_in_4bit=True       # falls dein Modell mit QLoRA trainiert wurde
)
model.save_pretrained_merged(
    "finetunedmodels/Falcon3-1B-Base-lora-pirate-origin-out/gguf",
    tokenizer,
)