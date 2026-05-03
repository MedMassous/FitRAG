# 🏋️ FitRAG — AI Fitness Coach (RAG System)

A production-ready **Retrieval-Augmented Generation** system that turns your training PDFs into a personal fitness coach, powered by **Groq (LLaMA 3)** + **ChromaDB** + **Streamlit**.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                        app.py                           │
│                  (Streamlit Frontend)                   │
└───────────────────────┬─────────────────────────────────┘
                        │
          ┌─────────────▼──────────────┐
          │      rag_pipeline.py        │
          │  UserProfile | MemoryStore  │
          │  Intent Detection | Groq    │
          └──────┬──────────┬──────────┘
                 │          │
    ┌────────────▼───┐  ┌───▼────────────────────┐
    │  retriever.py   │  │     ingestion.py        │
    │  ChromaDB Search│  │  PDF → Chunks → Chroma  │
    │  BM25 Search    │  │  SHA256 dedup           │
    │  RRF Fusion     │  │  SentenceTransformer    │
    │  Re-ranking     │  └─────────────────────────┘
    └────────────────┘
            │
    ┌───────▼────────┐
    │   data/         │
    │  chunks.json    │  ← chunk metadata + text
    │  metadata.json  │  ← per-file info
    │  memory.json    │  ← conversation history
    │  user_profile.  │  ← athlete profile
    └────────────────┘
    ┌───────▼────────┐
    │   chroma_db/    │  ← persisted ChromaDB vector store
    └────────────────┘
```

## 📁 File Structure

```
fitness_rag/
├── app.py              # Streamlit UI (chat, profile, history tabs)
├── ingestion.py        # PDF → chunks → ChromaDB pipeline
├── retriever.py        # Hybrid search (ChromaDB + BM25 + RRF + rerank)
├── rag_pipeline.py     # Groq LLM, memory, user profile, intent detection
├── requirements.txt
├── .env
├── README.md
├── chroma_db/          # Auto-created — ChromaDB persistent storage
└── data/               # Auto-created on first run
    ├── chunks.json              # All chunk texts + metadata
    ├── metadata.json            # Per-file ingestion info
    ├── memory.json              # Conversation history + feedback
    ├── processed_hashes.json    # SHA256 hashes (dedup)
    └── user_profile.json        # Athlete profile
```

---

## 🚀 Quick Start

### 1. Clone / copy the project

```bash
git clone https://github.com/MedMassous/FitRAG.git
cd FitRAG
```

### 2. Create virtual environment

```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> ⚠️ First run downloads the `all-MiniLM-L6-v2` model (~90 MB). This is cached locally afterward.

### 4. Configure environment

```bash
# Edit .env and add your GROQ_API_KEY
```

Get a **free** Groq API key at [console.groq.com](https://console.groq.com).

### 5. Run the app

```bash
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

---

## 📖 Usage Guide

### Step 1: Upload PDFs
- Click **"Browse files"** in the sidebar
- Select one or more training PDF files
- Click **"⚡ Index PDFs"** — takes ~10–30 seconds per PDF

### Step 2: Set your profile
- Go to the **👤 Profile** tab
- Fill in: level, goal, days/week, injuries
- This is automatically injected into every query

### Step 3: Ask questions
Example prompts:
- `"What should I train today?"`
- `"Build me a 4-day upper/lower split based on my PDFs"`
- `"What mistakes might I be making based on my programs?"`
- `"When should I take a deload week?"`
- `"Suggest a progressive overload schedule for bench press"`

### Step 4: Review sources
- Every answer shows **source badges** (which PDFs were used)
- Click **"Show N retrieved chunks"** to see the exact text passages
- Give 👍/👎 feedback to improve future responses

### Step 5: Add more PDFs
- Upload additional PDFs anytime
- Already-indexed files are **automatically skipped** (SHA256 dedup)
- Use **"🔄 Re-index"** to force a full rebuild

---

## 🧠 Technical Features

### Hybrid Search
| Method | Description |
|--------|-------------|
| **ChromaDB (Semantic)** | Cosine similarity search on 384-dim normalized embeddings, persisted via `PersistentClient` |
| **BM25 (Keyword)** | Okapi BM25 over tokenized chunk corpus |
| **RRF Fusion** | Reciprocal Rank Fusion with α=0.6 semantic weight |
| **Re-ranker** | Token overlap + fitness keyword boost + length norm |

### RAG Pipeline
- **Model**: LLaMA 3 70B via Groq (fastest inference available)
- **Context window**: Up to 8192 tokens
- **Anti-hallucination**: System prompt enforces document-grounded answers
- **Intent detection**: Classifies queries → expands for better retrieval
- **Memory**: Rolling 50-entry Q&A history injected as context

### Ingestion Pipeline
- **PDF parsing**: PyMuPDF (handles tables, multi-column layouts)
- **Chunking**: RecursiveCharacterTextSplitter (512 chars, 64 overlap)
- **Embeddings**: `all-MiniLM-L6-v2` via SentenceTransformers (384-dim, normalized)
- **Storage**: ChromaDB collection `fitness_chunks` — stores ids, embeddings, documents, and metadata (`source`, `page`, `chunk_index`) atomically via `upsert`
- **Deduplication**: SHA256 hash per file — re-uploads are instant skips
- **Incremental indexing**: New PDFs appended without full reindex

---

## ⚙️ Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `GROQ_API_KEY` | *(required)* | Your Groq API key |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | Groq model name |
| `EMBED_MODEL` | `all-MiniLM-L6-v2` | SentenceTransformer model |
| `CHROMA_DB_PATH` | `./chroma_db` | ChromaDB persistent storage directory |
| `CHUNK_SIZE` | `512` | Characters per chunk |
| `TOP_K` | `6` | Retrieved chunks per query |

---

## 🔧 Advanced: Python API

Use the pipeline programmatically:

```python
from ingestion import IngestionPipeline
from rag_pipeline import FitnessRAGPipeline

# Ingest PDFs
pipeline_in = IngestionPipeline()
result = pipeline_in.ingest(["program_a.pdf", "workout_b.pdf"])
print(f"Indexed: {result}")

# Ask questions
rag = FitnessRAGPipeline()
response = rag.ask("What should I train today?")
print(response["answer"])
print("Sources:", response["sources"])

# Update user profile
rag.update_profile(
    fitness_level="advanced",
    goal="muscle gain",
    days_per_week=5,
)

# Submit feedback
rag.submit_feedback("What should I train today?", feedback=1)

# Streaming response
for token in rag.ask_stream("Build me a push-pull-legs split"):
    print(token, end="", flush=True)
```

---

## 🛠️ Troubleshooting

**`NameError: name 'nn' is not defined`**

`transformers` ≥ 4.52 has a bug where `accelerate.py` uses `nn.Module` in a type
annotation without importing `torch.nn`. Pinned in `requirements.txt` — reinstall to fix:
```bash
pip install "transformers>=4.41.0,<4.52.0"
```

**`RuntimeError: Numpy is not available`**

PyTorch's `.numpy()` bridge was compiled against the numpy 1.x C ABI. numpy 2.x
broke that ABI silently. Pinned in `requirements.txt` — reinstall to fix:
```bash
pip install "numpy>=1.24.0,<2.0"
```

**`Model download stuck`**

The embedding model (~90 MB) downloads on first run. Check your internet connection.
Once cached it loads instantly on subsequent runs.

**`Groq rate limit`**

Free tier: 30 req/min on LLaMA 3 70B. Reduce `MAX_TOKENS` in `rag_pipeline.py`
or switch to `llama3-8b-8192`.

**`Empty answers`**

Your PDFs may contain scanned images (not text). Use an OCR tool
(e.g., Adobe Acrobat, `ocrmypdf`) to make them text-searchable first.

**`Streamlit port conflict`**
```bash
streamlit run app.py --server.port 8502
```

---

## 🗺️ Roadmap / Bonus Features

- [ ] Cross-encoder re-ranker (`cross-encoder/ms-marco-MiniLM-L-6-v2`)
- [ ] HyDE (Hypothetical Document Embeddings) query expansion
- [ ] Workout session logger with chart visualisation
- [ ] Export workout plan as PDF
- [ ] Multi-user support (per-user ChromaDB collections)
- [ ] WhatsApp / Telegram bot integration

---

## 📄 License

MIT — free for personal and commercial use.
