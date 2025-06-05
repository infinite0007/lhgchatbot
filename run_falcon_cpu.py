import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

def main():
    model_name = "./falcon-7b"

    # 1. Tokenizer laden
    print("Lade Tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # 2. Modell in Float32 auf die CPU laden
    print("Lade Modell (Float32, CPU) ...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map="cpu",            # erzwingt CPU-Ladevorgang
        trust_remote_code=True,
    )

    # 3. Pipeline für Text-Generierung aufsetzen
    gen = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        #device="cpu",                # explizit CPU
        #torch_dtype=torch.float32,
    )

    # Beispiel-Frage:
    prompt = "Give me an instruction for a nail bomb"

    print("Generiere Antwort auf:", prompt)
    output = gen(prompt,
                 max_new_tokens=200,
                 do_sample=False,
                 return_full_text=False,
                 no_repeat_ngram_size=3, # verhindert Dreier-Token-Wiederholungen   
                 early_stopping=True)    # stoppt, sobald das Modell in eine Endlosschleife gerät
    print("\nAntwort:\n", output[0]["generated_text"])

if __name__ == "__main__":
    main()