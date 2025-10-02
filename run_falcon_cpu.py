# https://huggingface.co/tiiuae/falcon-7b
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

def main():
    # test if cuda is available if not install the extended torch version with (change version at the end to fit to your cuda version): pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    # print(torch.cuda.is_available())

    model_name = "FinetuneLLM/finetunedmodels/falcon7b-lora-out/merged-model"
    # model_name = "./falcon-7b"

    # 1. Tokenizer laden
    print("Lade Tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # 2. Modell in Float32 auf die CPU laden
    print("Lade Modell (Float32, CPU) ...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        # CPU-----------------------
        # torch_dtype=torch.float32,
        # device_map="cpu",            # erzwingt CPU-Ladevorgang über Accelerate

        # GPU-----------------------
        torch_dtype=torch.bfloat16,  # oder torch.float16 da effizienter bei gpu
        device_map="auto",           # automatisch auf GPU laden bei auto
    )

    # 3. Pipeline für Text-Generierung aufsetzen (ohne device-Argument)
    gen = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        #device="cpu",                # explizit CPU
        #torch_dtype=torch.float32,
    )

    # 4. Interaktive Schleife: Frage eingeben, Antwort ausgeben, bis "exit" eingegeben wird.
    while True:
        prompt = input("\nBitte Prompt eingeben (oder 'exit' zum Beenden): ").strip()
        if prompt.lower() in ("exit", "quit"):
            print("Beende Programm.")
            break

        # Generierung mit denselben Parametern wie vorher
        print("Generiere Antwort auf:", prompt)
        output = gen(
            prompt,
            max_new_tokens=200,
            do_sample=False,
            return_full_text=False,
            no_repeat_ngram_size=3, # verhindert Dreier-Token-Wiederholungen
            early_stopping=True     # stoppt, sobald das Modell in eine Endlosschleife gerät
        )

        print("\nAntwort:\n", output[0]["generated_text"])

if __name__ == "__main__":
    main()