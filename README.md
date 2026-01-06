# ⚡ Private-RAG: Local-First Document Intelligence (Tables + Figures + PDFs)

![Status](https://img.shields.io/badge/Status-Beta-brightgreen)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![UI](https://img.shields.io/badge/UI-Streamlit-ff4b4b)
![LLM](https://img.shields.io/badge/LLM-Ollama%20(qwen3%3A4b)-orange)
![VectorDB](https://img.shields.io/badge/VectorDB-Chroma-6f42c1)

> A production-leaning RAG assistant that runs **fully on your machine**: local LLM (Ollama), persistent vector DB (Chroma), hybrid retrieval (BM25 + embeddings), and **table/figure-aware PDF ingestion**.

---

## Why this repo is different

Most RAG demos are “PDF → chunk → embed → ask”. This project adds the stuff to improve the accuracy:

- **Hybrid Retrieval (Semantic + Keyword)**  
  Combines a semantic retriever (Chroma embeddings) with a BM25 retriever and ensembles them for stronger recall on technical PDFs.

- **Cross-Encoder Reranking (High-signal context)**  
  Reranks retrieved candidates using a HuggingFace cross-encoder reranker so the LLM sees the most relevant chunks.

- **Table + Figure Awareness (not just “OCR everything”)**  
  Uses PyMuPDF native `find_tables()` when possible, and falls back to **GOT-OCR2** for image-based tables/formulas. Tables are serialized into structured metadata and also converted into searchable plain text for BM25.

- **Query Router + Follow-up Condensing (multi-turn done right)**  
  Classifies queries into Standalone / Follow-up / Clarification / Chitchat and only condenses follow-ups when needed, reducing retrieval drift.

- **Streaming UI + Source Grounding**  
  Streamlit chat UI streams tokens, shows retrieval metrics, and renders table sources as interactive DataFrames + CSV download.

---

## Architecture (high-level)

1. **Ingestion**
   - PDF parsing with structural blocks (text/table/figure)
   - Native table extraction via PyMuPDF + OCR fallback via GOT-OCR2
   - Structured blocks formatted with `[TABLE:] ... [/TABLE]` and `[FIGURE] ... [/FIGURE]` markers

2. **Indexing**
   - Persistent vector DB: **Chroma** (on disk)
   - Keyword index: **BM25** over all stored chunks (rebuilt from Chroma corpus)

3. **Retrieval**
   - Ensemble retriever: embeddings + BM25
   - Cross-encoder reranking to select top-k context

4. **Answering**
   - Local LLM via **Ollama**
   - (Optional) retrieval grading + query transformation loop (LangGraph mode toggle)

---

## Features

### ✅ What works well
- Ask questions across multiple PDFs / text / markdown uploads (local-only).
- Retrieves relevant text, tables, and formula/figure OCR snippets with visible sources.
- Tables render as interactive DataFrames + CSV export (when metadata is present).
- Persistent caching:
  - OCR/text extraction cached to disk
  - Chroma persisted to disk for reuse across runs

### 🔁 Optional “self-correct” mode
There is a LangGraph flow that can grade retrieved docs and (optionally) transform the query once if retrieval is weak. This is controlled via config flags.

---

## Tech stack

- **UI**: Streamlit  
- **LLM**: Ollama (default model in config: `qwen3:4b-instruct`)  
- **Embeddings**: `Alibaba-NLP/gte-multilingual-base` (HuggingFace)  
- **Vector DB**: Chroma persistent store  
- **Keyword Retrieval**: BM25  
- **Reranker**: `cross-encoder/ms-marco-MiniLM-L12-v2`  
- **PDF parsing**: PyMuPDF (`fitz`) + GOT-OCR2 for image OCR  

---

## Quickstart

### 1) Install Ollama + pull the model
```bash
ollama pull qwen3:4b-instruct
```

### 2) Create a venv + install deps
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Mac/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

### 3) Run the app
```bash
streamlit run app.py
```

Upload PDFs and start asking questions like:
- “What does Table 3.1 contain?”
- “Compare the values in row 3 vs row 5”
- “Summarize the assumptions used in the derivation on page 12”

---

## Configuration

All tuning is centralized in `config.py`.

### Retrieval/Chunking
- `CHUNK_SIZE`, `CHUNK_OVERLAP` — affects recall & latency
- `N_SEMANTIC_RESULTS`, `N_BM25_RESULTS` — how many candidates each retriever returns
- `N_CONTEXT_RESULTS` — final context size used after reranking

### Chat behavior
- `ENABLE_QUERY_ROUTER` — classify/condense follow-ups
- `GRADING_MODE` — enable optional grading + query transform loop

### Paths
- `VECTOR_DB_DIR` — Chroma persistence location
- `DATA_DIR/ocr_cache` — OCR extraction cache

---

## Table & Figure handling (how it works)

This project treats tables as first-class content:

- **Native tables** are extracted with PyMuPDF and converted to markdown + structured dicts.
- **Image-based tables/formulas** go through GOT-OCR2; if a markdown table is detected, it’s parsed into `StructuredTable`.
- Tables are indexed in two complementary ways:
  1) The markdown table itself for semantic retrieval
  2) A “searchable text” rendering (headers + row=value pairs) to boost BM25 hits

In the UI, table sources display as interactive tables + allow CSV download.

---

## Repo layout

- `app.py` — Streamlit UI, streaming responses, source rendering (tables/figures/text)
- `chatbot.py` — LangGraph workflow, query router/condense, retrieval + generation
- `data_ingestor.py` — chunking, Chroma persistence, BM25 build, ensemble retrieval, reranking
- `pdf_loader.py` — PDF parsing, native table extraction, GOT-OCR2 OCR, caching
- `table_intelligence.py` — markdown table detection + structured parsing utilities
- `config.py` — all tunables and paths

---

## Notes / Known limitations:

- **GPU expectation**: embeddings + reranker are configured for CUDA in ingestion (can be made CPU-fallback if needed).
- **Re-indexing updated PDFs**: if a file changes, re-indexing occurs (production upgrade: delete old chunks for that source before re-adding).
- **OCR cost**: GOT-OCR2 is heavy; caching is enabled to avoid repeated runs on the same file.

---

## Roadmap (high-signal upgrades)

- CPU fallback for embeddings/reranker (auto device selection)
- “Delete stale chunks by source” on file change (true incremental indexing)
- Offline evaluation harness (retrieval hit-rate + latency + answer faithfulness)
- Docker Compose: app + ollama + persisted volumes

---
