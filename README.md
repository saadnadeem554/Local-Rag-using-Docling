# 📚 Local RAG System using Docling

A fully local, CPU-based multimodal Retrieval-Augmented Generation (RAG) system that processes PDF documents with text, tables, and images.

## ✨ Features

- **🔍 PDF Parsing** - Extract text, tables and images using Docling
- **🖼️ Multimodal Support** - Image analysis using Llama 4 Scout vision model
- **🧠 Semantic Search** - BGE embeddings with ChromaDB vector store
- **⚡ Fast Reranking** - FlashRank for improved retrieval accuracy
- **💬 Interactive Chat** - CLI-based chat interface with streaming responses
- **🆓 Free to Run** - Uses Groq API (free tier available) for LLM inference

## 🏗️ Architecture

```
PDF Document
     │
     ├─→ Text Chunks ──→ Embeddings ──→ ChromaDB
     ├─→ Tables ──────→ Markdown ────→ ChromaDB
     └─→ Images ──────→ Vision LLM ──→ ChromaDB
                              │
                              ▼
                    Query → Retrieve → Rerank → Generate
```

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| PDF Parsing | [Docling](https://github.com/DS4SD/docling) |
| Embeddings | [sentence-transformers](https://www.sbert.net/) (BGE-small) |
| Vector Store | [ChromaDB](https://www.trychroma.com/) |
| Reranking | [FlashRank](https://github.com/PrithivirajDamodaran/FlashRank) |
| LLM | [Groq](https://groq.com/) (Llama 3.3 70B) |
| Vision | Llama 4 Scout 17B |
| CLI | [Typer](https://typer.tiangolo.com/) + [Rich](https://rich.readthedocs.io/) |

## 📦 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/saadnadeem554/Local-Rag-using-Docling.git
   cd Local-Rag-using-Docling
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # Linux/Mac
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   cp .env.example .env
   ```
   
   Edit `.env` and add your Groq API key:
   ```
   GROQ_API_KEY=your_groq_api_key_here
   ```
   
   Get your free API key from [console.groq.com/keys](https://console.groq.com/keys)

## 🚀 Usage

### Interactive Chat Mode
```bash
python main.py chat
```

In chat mode, you can:
- `/ingest <path>` - Ingest a PDF document
- `/stats` - View vector store statistics
- `/clear` - Clear all documents
- `/quit` - Exit the chat

Or just type your question to query the documents!

### CLI Commands

**Ingest a PDF:**
```bash
python main.py ingest path/to/document.pdf
```

**Ask a question:**
```bash
python main.py ask "What is the main topic of the document?"
```

**Stream the response:**
```bash
python main.py ask "Summarize the key points" --stream
```

**View statistics:**
```bash
python main.py stats
```

**Clear vector store:**
```bash
python main.py clear
```

## ⚙️ Configuration

Edit `rag_system/config.py` to customize:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `CHUNK_SIZE` | 500 | Token size for text chunks |
| `CHUNK_OVERLAP` | 50 | Overlap between chunks |
| `TOP_K_RETRIEVAL` | 15 | Documents to retrieve |
| `TOP_K_RERANK` | 5 | Documents after reranking |
| `EMBEDDING_MODEL` | BGE-small-en | Embedding model |
| `LLM_MODEL` | llama-3.3-70b | Generation model |

## 📁 Project Structure

```
├── main.py                 # CLI entry point
├── requirements.txt        # Dependencies
├── .env.example           # Environment template
├── rag_system/
│   ├── config.py          # Configuration
│   ├── parser.py          # PDF parsing with Docling
│   ├── chunker.py         # Text chunking
│   ├── embedder.py        # Embedding generation
│   ├── vector_store.py    # ChromaDB operations
│   ├── reranker.py        # FlashRank reranking
│   ├── image_describer.py # Vision model integration
│   ├── generator.py       # LLM response generation
│   └── pipeline.py        # Main RAG pipeline
└── data/                  # Generated data (gitignored)
    ├── pdfs/              # Stored PDFs
    ├── images/            # Extracted images
    └── chroma_db/         # Vector database
```

## 📄 License

MIT License

## 🤝 Contributing

Contributions are welcome! Feel free to open an issue or submit a pull request.
