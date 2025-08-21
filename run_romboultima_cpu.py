# https://huggingface.co/FINGU-AI/RomboUltima-32B
import torch
from transformers import (
    modeling_utils,
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline
)

def main():
    # ────────────────────────────────────────────────────────────────────────────
    # A) Workaround: ALL_PARALLEL_STYLES initialisieren (für Qwen2-/Rombo-Modelle)
    # ────────────────────────────────────────────────────────────────────────────
    if not hasattr(modeling_utils, "ALL_PARALLEL_STYLES") or modeling_utils.ALL_PARALLEL_STYLES is None:
        modeling_utils.ALL_PARALLEL_STYLES = ["tp", "none", "colwise", "rowwise"]

    # ────────────────────────────────────────────────────────────────────────────
    # B) Lokaler Klon-Pfad: RomboUltima-32B
    # ────────────────────────────────────────────────────────────────────────────
    model_path = "./RomboUltima-32B"  # Passe an, falls dein Ordner anders heißt

    # ────────────────────────────────────────────────────────────────────────────
    # C) 1. Config laden & nur CPU-Offload-Feld setzen
    # ────────────────────────────────────────────────────────────────────────────
    print("1/4: Lade Config und setze llm_int8_enable_fp32_cpu_offload …")
    config = AutoConfig.from_pretrained(model_path)
    if hasattr(config, "quantization_config") and config.quantization_config is not None:
        # Die bereits vorhandene 4-Bit-Einstellung bleibt; hier fügen wir CPU-Offload hinzu.
        config.quantization_config["llm_int8_enable_fp32_cpu_offload"] = True

    # ────────────────────────────────────────────────────────────────────────────
    # D) 2. Tokenizer laden (rein lokal)
    # ────────────────────────────────────────────────────────────────────────────
    print("2/4: Lade Tokenizer …")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # ────────────────────────────────────────────────────────────────────────────
    # E) 3. Modell laden: vorquantisierte 4-Bit-Shards + Auto-Sharding + Offload
    # ────────────────────────────────────────────────────────────────────────────
    print("3/4: Lade vorquantisiertes 4-Bit-Modell mit Auto-Sharding …")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            config=config,               # Config mit angepasstem CPU-Offload-Feld
            device_map="cuda:0",         # Accelerate verteilt automatisch GPU/CPU cpu war davor drin für gpu cuda eingeben
            offload_folder="offload",    # Ordner für ausgelagerte Tensoren
            offload_state_dict=True      # Schreibt große Tensors ins Dateisystem
        )
    except RuntimeError as e:
        print("Fehler beim Laden des Modells:", e)
        print("Mögliche Gründe:")
        print(" • Nicht genug GPU-VRAM (12 GB) oder")
        print(" • Nicht genug RAM (≥ 16 GB), um Offload zu puffern")
        print("Lösungsvorschläge:")
        print(" • Stelle sicher, dass GPU >11 GB frei hat, ggf. mit `max_memory` nachsteuern")
        print(" • Falls das weiter scheitert, wechsele zu Int8 (`load_in_8bit=True`) oder CPU-Modus")
        return

    # ────────────────────────────────────────────────────────────────────────────
    # F) 4. Pipeline für Text-Generierung einrichten
    # ────────────────────────────────────────────────────────────────────────────
    print("4/4: Richte Text-Generation-Pipeline ein …")
    gen = pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer
    )

    # ────────────────────────────────────────────────────────────────────────────
    # G) Interaktive Schleife: Prompt eingeben → Antwort ausgeben → bis "exit"
    # ────────────────────────────────────────────────────────────────────────────
    print("Modell ist bereit. Gib deinen Prompt ein (oder 'exit', um abzubrechen):")
    while True:
        prompt = input(">> ").strip()
        if prompt.lower() in ("exit", "quit"):
            print("Beende Programm.")
            break

        print("Generiere Antwort …")
        try:
            output = gen(
                prompt,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.7,
                top_k=50,
                top_p=0.9,
                return_full_text=False,
                eos_token_id=tokenizer.eos_token_id,
                no_repeat_ngram_size=3,
                early_stopping=True
            )
            print("\nAntwort:\n" + output[0]["generated_text"])
        except RuntimeError as gen_error:
            print("Fehler während der Generierung:", gen_error)
            print("Evtl. zu wenig GPU-VRAM (12 GB) oder RAM (16 GB).")
            print("Überlege, Prompt zu verkürzen, `max_memory` zu definieren oder auf Int8/CPU-Modus zu wechseln.")
            break

if __name__ == "__main__":
    main()
