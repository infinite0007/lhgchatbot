# lhgchatbot

Masterarbeit:  
**„Finetuning, RAG und CAG: Techniken zur Einbindung interner Wissensdaten in LLM-basierte Chatbots“**


Alle Informationen, Methoden und ergänzende Materialien zu meiner Masterarbeit sind auf **ResearchGate** verfügbar.  
Hier geht’s direkt zur Publikation:  
[Finetuning RAG und CAG Techniken zur Einbindung interner Wissensdaten in LLM-basierte Chatbots](https://www.researchgate.net/publication/399530401_Finetuning_RAG_und_CAG_Techniken_zur_Einbindung_interner_Wissensdaten_in_LLM-basierte_Chatbots)


Dieses Repository enthält alle Skripte, um lokale LLM-Chatbots mit internen Wissensdaten aufzubauen und zu evaluieren:

- **Finetuning** eines Basismodells auf Liebherr-spezifische QA-Daten
- **RAG (Retrieval-Augmented Generation)** mit Vektordatenbank (Confluence-Inhalte)
- **CAG (Cache-Augmented Generation)** mit lokalem KV-/Embedding-Cache
- **WikiExtraction**: Pipeline zum Extrahieren und Aufbereiten der Confluence-Daten

Obsolete/alte Skripte sind im Repo belassen, können aber ignoriert werden, wenn nicht explizit erwähnt.

---

## 1. Voraussetzungen

- Python (getestet mit 3.10/3.11)
- `pip` für Python-Pakete
- (Optional, empfohlen) CUDA-fähige GPU + passender NVIDIA-Treiber

### Nvidia Cuda (GPU Fähigkeit)
Wichtig zudem ist es, für die Projekt eine Nvidia GPU zu besitzen. Nachfolgendes Nvidia Toolkit ist deshalb absolut erforderlich:

https://developer.nvidia.com/cuda-12-8-0-download-archive

Es wird empfohlen immer die aktuelle Version herunterzuladen. 

noch vor nachfolgender requirementsinstallation sollte die torch Version: torch==2.8.0+cu128 die der Cuda Version des Toolkits angepasst werden. In diesem Fall entspricht cu128 der Cuda Version 12.8.

### Installation der Python-Abhängigkeiten

Im Verzeichnis `lhgchatbot/`:

```bash
pip install -r requirements.txt
```

Einige Teilmodule haben eigene `requirements.txt`, z.B.:

- [FinetuneLLM/requirements.txt](FinetuneLLM/requirements.txt)
- [RAG/requirements.txt](RAG/requirements.txt)
- [WikiExtraction/requirement.txt](WikiExtraction/requirement.txt)

Diese sorgen dafür, dass das Projekt die richtigen Python pakete installier um volle funktionsfähigkeit zu gewährleisten. Pakete die Python mit pip installieren sollte mittels:

pip install -r requirements.txt

Ebenso wird ein Modell in unserem Fall der Sieger: Qwen2.5-1.5B-Instruct benötigt.

Dieser wird im Hauptordner neben den anderen RAG/CAG/FinetuneLLM geclont. Dazu nachfolgendes ausführen:

Im Verzeichnis `lhgchatbot/`:

```bash
git clone https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct
```

sollte dies zu Unternehmensnetzblockierungen kommen sollte SSL kurzzeitig deaktiviert werden mittels:

```bash
git -c http.sslVerify=false clone https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct
```

Danach kann das Modell für Finetuning, Rag & Cag genutzt werden.

### WikiExtraction
Um die Confluence Datenextraktion zu gewährleisten müssen die Nutzerdaten mit E-Mail & Passwort in eine gesicherte .env Datei abgelegt werden. Zu diesem Zeitpunkt der Erstellung hatte es nur mit dieser - nicht aber mit der Tokenmethode funktioniert.

Diese .env wird von gitignore ignoriert und sollte deswegen auch nicht online landen.

Diese befindet sich im Verzeichnis `WikiExtraction/.env`:



---

## 2. Datenpipeline: Confluence → „canonical“ JSONL → Fine-tune/RAG/CAG

Nachfolgend werden die Firmeninternen Daten angereichert und verarbeitet. Alles findet in Confluence statt - die Seiten und Unterseiten eines Bereichs werden dabei komplett in der Baumhierarchie durchsucht und gespeichert. Danach bearbeiten weitere Skripte die gefundenen Dateien z. B. Pdfs oder Bilder.


Alle Methoden (Finetuning, RAG, CAG) basieren auf denselben aufbereiteten Confluence-Daten. Das wichtigste ist zunächst einen Zugang also Account für Confluence für die Datensammlung zu deklarieren. Dazu bitte in Order: WikiExtraction/.env die Nutzerdaten anpassen. Achtung - Diese werden automatisch im gitignore für späteren Upload ignoriert und sollen nie online committed & gepushed werden.

### 2.1 Confluence-Daten abrufen (WikiExtraction)

Verzeichnis: [WikiExtraction](WikiExtraction/)

1. **Confluence-Export (kanonisch)**  
   Skript: [`WikiExtraction/canonical_extractor.py`](WikiExtraction/canonical_extractor.py)  
   Ziel: `data/raw/confluence.jsonl` erzeugen

   Beispielaufruf:

   ```bash
   cd WikiExtraction
   python canonical_extractor.py \
     --space SWEIG \
     --with-attachments \
     --since 2024-01-01 \
     --out data/raw/confluence.jsonl
   ```

2. **Attachments parsen (optional, wenn benötigt)**  
   Beispiele:
   - PDFs (Docling, OCR)
   - Draw.io
   - Bilder-OCR

   Relevante Skripte (nicht alle müssen genutzt werden, je nach Setup):

   - [`attachments_drawio_extract.py`](WikiExtraction/attachments_drawio_extract.py)  
   - [`attachments_pdf_docling_prepareA_figures.py`](WikiExtraction/attachments_pdf_docling_prepareA_figures.py)  
   - [`attachments_pdf_docling_prepareB_figuresocr.py`](WikiExtraction/attachments_pdf_docling_prepareB_figuresocr.py)  
   - [`attachments_ocr.py`](WikiExtraction/attachments_ocr.py)

   Diese schreiben abgeleitete JSONL-Dateien nach `data/derivatives/`, z. B.:

   - `data/derivatives/pdf_docling_prepared.jsonl`
   - `data/derivatives/ocr.jsonl`
   - `data/derivatives/prepared_figures_ocr.jsonl`
   - `data/derivatives/drawio_text.jsonl`

3. **Alles zu „joined_pages_full.jsonl“ zusammenführen**

   Skript: [`WikiExtraction/join_all_pages.py`](WikiExtraction/join_all_pages.py)  

   ```bash
   cd WikiExtraction

   python join_all_pages.py \
     --pages data/raw/confluence.jsonl \
     --ocr-images data/derivatives/ocr.jsonl \
     --drawio data/derivatives/drawio_text.jsonl \
     --pdf-docling data/derivatives/pdf_docling_prepared.jsonl \
     --pdf-fig-ocr data/derivatives/prepared_figures_ocr.jsonl \
     --out data/derivatives/joined_pages_full.jsonl \
     --tables-as-text
   ```

   Ergebnis: **`data/derivatives/joined_pages_full.jsonl`**  
   → Basis für **Finetuning**, **RAG** und **CAG**.

---
Nachfolgend wird 3. Finetuning und 5. CAG erklärt aber da durch die Arbeit bewiesen wurde das für uns 4. RAG die beste Methode für Chatbots ist - kann direkt zu RAG gesprungen werden. 

## 3. Finetuning

Verzeichnis: [FinetuneLLM](FinetuneLLM/)

Ziel: Ein vorhandenes Foundation-Modell (z. B. Falcon) mit Liebherr-spezifischen QA-Daten weitertrainieren.

### 3.1 Q&A-Datensatz aus Confluence erzeugen

Skript: [`WikiExtraction/to_finetune.py`](WikiExtraction/to_finetune.py)  
Dieses Skript erzeugt **extraktive** Q&A-Paare (ohne Online-LLM, keine Halluzinationen) aus `joined_pages_full.jsonl`.

```bash
cd WikiExtraction

python to_finetune.py \
  --in  data/derivatives/joined_pages_full.jsonl \
  --out data/derivatives/qa_dataset.jsonl \
  --chunk-chars 1800 \
  --max-items-per-chunk 3
```

Ergebnis: `data/derivatives/qa_dataset.jsonl`

### 3.2 LLM finetunen (Unsloth + GGUF)

Hauptskript: [`FinetuneLLM/finetunellm_unsloth_gguf.py`](FinetuneLLM/finetunellm_unsloth_gguf.py)

Dieses Skript nutzt:

- ein Basis-HF-Modell (z. B. Falcon 3 1B)
- QLoRA (Low-Rank + 4-bit Quantisierung)
- `unsloth` + `trl.SFTTrainer`, um LoRA-Adapter zu trainieren
- optional Konvertierung des gemergten Modells nach GGUF (für `llama.cpp`)

Typische Schritte:

1. **Konfiguration im Skript anpassen**
   - Pfad zum Basis-Modell (`BASE_MODEL`)
   - `DATA_PATH` → auf `WikiExtraction/data/derivatives/qa_dataset.jsonl`
   - `OUTPUT_DIR` → z. B. `FinetuneLLM/finetunedmodels/Falcon3-1B-Base-lora-unsloth-liebherrqa-out`

2. **Training starten**

   ```bash
   cd FinetuneLLM
   python finetunellm_unsloth_gguf.py
   ```

3. **(Optional) GGUF-Konvertierung**

   Nach dem Mergen der LoRA-Gewichte:  
   Im Skript ist beschrieben, wie man mit `llama.cpp`/`convert_hf_to_gguf.py` ein GGUF-Modell erzeugt.

   Beispiel (auskommentierte Anleitung im Skript):

   ```bash
   cd ..  # in den Ordner mit llama.cpp wechseln, dort:
   python convert_hf_to_gguf.py \
     "Pfad/zum/merged_model_prepared_gguf" \
     --outfile name.gguf
   ```

### 3.3 Finetuned Modell lokal testen

Einfacher CLI-Runner: [`run_model.py`](run_model.py)

- Lädt ein (ggf. finetuned) HF-Modell
- Unterstützt CPU oder GPU (über `device_map` und `dtype` anpassbar)

Konfiguration im Skript:

- `model_name` auf den Ordner deines finetuned/merged Modells setzen, z. B.:

  ```python
  model_name = "FinetuneLLM/finetunedmodels/Falcon3-1B-Base-lora-pirate-out/merged_model"
  ```

Ausführen:

```bash
cd lhgchatbot
python run_model.py
```

Dann interaktiv Prompts eingeben (`exit` zum Beenden).  
`run_romboultima_cpu.py` ist ein Spezial-Runner für das RomboUltima-32B-Modell auf CPU.

---

## 4. RAG (Retrieval-Augmented Generation)

Verzeichnis: [RAG](RAG/)

Ziel: LLM-Antworten mit aktuellen Confluence-Inhalten anreichern, indem relevante Chunks per Vektorsuche gesucht werden.

### 4.1 Index-Build (Chroma-DB)

Skript: [`RAG/rag_indexdb.py`](RAG/rag_indexdb.py)

- Nimmt `joined_pages_full.jsonl`
- Chunkt den Inhalt
- Erstellt Embeddings (`sentence-transformers/all-MiniLM-L6-v2`)
- Schreibt alles in eine Chroma-Vektordatenbank

Standardpfade:

- Input: `../WikiExtraction/data/derivatives/joined_pages_full.jsonl`
- Chroma-DB: `chroma_db`
- Collection: `rag_data`

Beispiel:

```bash
cd RAG

python rag_indexdb.py \
  --dataset-path      ../WikiExtraction/data/derivatives/joined_pages_full.jsonl \
  --database-location chroma_db \
  --collection-name   rag_data \
  --embedding-model   sentence-transformers/all-MiniLM-L6-v2 \
  --chunk-size        1000 \
  --chunk-overlap     200
```

(Alle Parameter sind optional; Standardwerte sind im Skript definiert.)

### 4.2 RAG-Chat-UI (Streamlit)

Skript: [`RAG/rag_chat.py`](RAG/rag_chat.py)

Features:

- UI über Streamlit
- Retrieval via Chroma (Top-K oder MMR)
- Inline-Zitationsformat `[1]`, `[2]` etc.
- Optional:
  - LLM über HF (lokaler Ordner) oder
  - quantifiziertes GGUF-Modell via `LlamaCpp`

Wichtige Konfig im Skript:

- `DATABASE_LOCATION = "chroma_db"`
- `COLLECTION_NAME   = "rag_data"`
- `EMBEDDING_MODEL   = "sentence-transformers/all-MiniLM-L6-v2"`
- `HF_MODEL_PATH     = "../gemma-3-4b-Instruct"`
- `USE_GGUF` / `GGUF_MODEL_PATH` / `N_CTX` / `N_GPU_LAYERS`

Start:

```bash
cd RAG
streamlit run rag_chat.py
```

Voraussetzung:

- Chroma-DB wurde vorher über [`rag_indexdb.py`](RAG/rag_indexdb.py) aufgebaut.
- Das gewählte Modell (HF-Ordner oder GGUF) liegt lokal vor.

### 4.3 RAG-Evaluation (Ähnlichkeit zwischen Modellantworten)

Skript: [`RAG/rag_similaritymodelanswers_eval.py`](RAG/rag_similaritymodelanswers_eval.py)

- Verwendet dieselbe Chroma-DB
- Fragt mehrere lokale Modelle mit denselben Wissensfragen ab
- Nutzt Sentence-Embeddings, um Ähnlichkeit der Antworten zu vergleichen

Konfiguration:

- `MODELS`: Mapping Modellname → lokaler HF-Pfad (z. B. `../Qwen2.5-3B-Instruct`)
- `QUESTIONS`: Fragenkatalog

Start:

```bash
cd RAG
python rag_similaritymodelanswers_eval.py
```

---

## 5. CAG (Cache-Augmented Generation)

Verzeichnis: [CAG](CAG/)

Ziel: Embeddings + LLM-KV-Caches für häufig verwendete Wissenschunks vorab berechnen und in einem Cache speichern, der dann zur Laufzeit (Hot-Set) schnell geladen wird.

### 5.1 CAG-Index + KV-Cache bauen

Skript: [`CAG/cag_build_index.py`](CAG/cag_build_index.py)

Aufgaben:

- Liest `joined_pages_full.jsonl`
- Chunkt die Texte
- Erzeugt Embeddings mit `SentenceTransformer` (Standard: `all-MiniLM-L6-v2`)
- Berechnet für jeden Chunk vorberechnete KV-Caches (Key/Value) eines HF-LLMs (z. B. Qwen2.5-0.5B-Instruct)
- Speichert alles in `cag_cache/` (inkl. „Hotset“-Liste)

Standardkonfiguration (im Skript):

- `DEFAULT_DATASET_PATH   = "../WikiExtraction/data/derivatives/joined_pages_full.jsonl"`
- `DEFAULT_OUTDIR         = "cag_cache"`
- `DEFAULT_EMBED_MODEL    = "sentence-transformers/all-MiniLM-L6-v2"`
- `DEFAULT_CHUNK_SIZE     = 1000`
- `DEFAULT_CHUNK_OVERLAP  = 200`
- `DEFAULT_HF_MODEL       = "../Qwen2.5-0.5B-Instruct"`
- `DEFAULT_HOTSET_N       = 1024`

Beispiel:

```bash
cd CAG

python cag_build_index.py \
  --dataset-path  ../WikiExtraction/data/derivatives/joined_pages_full.jsonl \
  --outdir        cag_cache \
  --embed-model   sentence-transformers/all-MiniLM-L6-v2 \
  --hf-model      ../Qwen2.5-0.5B-Instruct \
  --chunk-size    1000 \
  --chunk-overlap 200 \
  --hotset-n      1024
```

### 5.2 CAG-Chat-UI (Streamlit, mit Hot-Set)

Skript: [`CAG/cag_chat.py`](CAG/cag_chat.py)

Features:

- UI über Streamlit
- Query-Encoding via `SentenceTransformer`
- Ähnlichkeitssuche (FAISS optional)
- Für die Top-K Chunks werden vormals berechnete KV-Caches geladen und mit auf die GPU gemappt → schnellere, kontextangereicherte Generierung

Wichtige Konfiguration:

- `CACHE_DIR        = "cag_cache"`
- `HF_MODEL_PATH    = "../Qwen2.5-0.5B-Instruct"`
- `TOP_K`, `TEMPERATURE`, `MAX_TOKENS`, `SNIPPET_CHARS`
- `USE_FAISS = True`

Start:

```bash
cd CAG
streamlit run cag_chat.py
```

Voraussetzung:

- CAG-Cache wurde vorher mit [`cag_build_index.py`](CAG/cag_build_index.py) erzeugt.
- Modellpfad `HF_MODEL_PATH` zeigt auf ein lokales HF-Modell, das zu den KV-Caches passt.

---

## 6. Sonstige Skripte und Hinweise

- [`run_model.py`](run_model.py)  
  Einfache CLI zum Laden und Testen beliebiger (auch finetuneder) HF-Modelle auf CPU oder GPU.

- [`run_romboultima_cpu.py`](run_romboultima_cpu.py)  
  Beispiel zum Laden des `RomboUltima-32B`-Modells auf CPU (mit Workarounds für Qwen-/Rombo-Parallelisierungs-Settings).

- `*_obsolete.py`  
  Historische Skripte (z. B. ältere Finetune- oder QA-Generatorvarianten).  
  Können für die Hauptpipelines ignoriert werden.

- [`llama.cpp infos to run.txt`](llama.cpp%20infos%20to%20run.txt)  
  Interne Notizen zur Nutzung von `llama.cpp` und GGUF-Modellen (Serverstart, CUDA-Konfiguration etc.).

---

## 7. Typische End-to-End-Pipelines (Kurzüberblick)

### A) Finetuning-Pipeline

1. Confluence exportieren → [`canonical_extractor.py`](WikiExtraction/canonical_extractor.py)  
2. Attachments (optional) → `attachments_*.py`  
3. Join → [`join_all_pages.py`](WikiExtraction/join_all_pages.py) → `joined_pages_full.jsonl`  
4. QA-Dataset → [`to_finetune.py`](WikiExtraction/to_finetune.py) → `qa_dataset.jsonl`  
5. Finetuning → [`finetunellm_unsloth_gguf.py`](FinetuneLLM/finetunellm_unsloth_gguf.py)  
6. Testen → [`run_model.py`](run_model.py)

### B) RAG-Pipeline

1. Datenpipeline bis `joined_pages_full.jsonl`  
2. Chroma-Index → [`rag_indexdb.py`](RAG/rag_indexdb.py)  
3. RAG-Chat-UI → [`rag_chat.py`](RAG/rag_chat.py)  
4. Optional: Evaluation → [`rag_similaritymodelanswers_eval.py`](RAG/rag_similaritymodelanswers_eval.py)

### C) CAG-Pipeline

1. Datenpipeline bis `joined_pages_full.jsonl`  
2. CAG-Cache + Embeddings + KV → [`cag_build_index.py`](CAG/cag_build_index.py)  
3. CAG-Chat-UI → [`cag_chat.py`](CAG/cag_chat.py)

---

## 8. Lizenz / Nutzung

Dieses Projekt wurde im Rahmen einer Masterarbeit entwickelt und ist auf interne Liebherr-Daten fokussiert.  
Bitte nur mit geeigneten Zugangsdaten zu Confluence und im Rahmen der internen Richtlinien verwenden.