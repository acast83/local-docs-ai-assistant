# Local Docs AI Assistant 🤖

A local Retrieval-Augmented Generation (RAG) chatbot that lets you chat with any documentation using LangChain,
ChromaDB, and Ollama LLMs. Get instant AI-powered answers with source citations, all running completely locally.

![Python](https://img.shields.io/badge/python-3.12+-blue.svg)
![LangChain](https://img.shields.io/badge/LangChain-0.3+-green.svg)

## ✨ Features

- 📚 **Load & Process**: Automatically load and chunk your markdown documentation
- 🔍 **Smart Search**: Vector-based similarity search using ChromaDB
- 🤖 **Local AI**: Powered by Ollama LLMs (no API keys required)
- 💬 **Interactive Chat**: Conversational interface with context memory
- 📖 **Source Citations**: Always shows which documents the answers came from
- ⚡ **Fast Responses**: Optimized for quick responses using efficient models
- 🔧 **Easy Setup**: Simple configuration via environment variables
- 📁 **Any Docs**: Works with any markdown documentation (Unity, Django, React, etc.)

## 🚀 Quick Start

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager
- [Ollama](https://ollama.ai/) installed and running

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/acast83/local-docs-ai-assistant.git
   cd local-docs-ai-assistant
   ```

2. **Set up environment with uv**
   ```bash
   # Create virtual environment and install dependencies
   uv venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   uv sync
   ```

3. **Install Ollama models**
   ```bash
   # Install a fast LLM model
   ollama pull llama3.2
   
   # Install embedding model
   ollama pull nomic-embed-text
   ```

4. **Configure environment**
   ```bash
   cp .env.example .env
   # Edit .env with your settings
   ```

### Basic Usage

1. **Load your documentation**
   ```bash
   python load_documents.py
   ```

2. **Start chatting**
   ```bash
   python rag_chatbot.py
   ```

3. **Ask questions about your documentation!**
   ```
   👤 You: How do I implement authentication?
   🤖 Answer: Based on your documentation...
   ```

## ⚙️ Configuration

Create a `.env` file in the project root:

```env
# Required: Path to your documentation folder
DOCS_DIRECTORY_PATH=/path/to/your/docs

# Optional: Customize models and paths
CHROMA_PATH=./chroma_db
EMBEDDING_MODEL=nomic-embed-text
LLM_MODEL=llama3.2
OLLAMA_BASE_URL=http://localhost:11434
```

## 📁 Project Structure

```
local-docs-ai-assistant/
├── load_documents.py      # Script to load and process documentation
├── rag_chatbot.py         # Main chatbot interface
├── pyproject.toml         # Python dependencies
├── .python-version        # Python version for uv
├── .env.sample            # Environment variables template
├── .env                   # Your environment configuration (create this)
├── chroma_db/             # ChromaDB vector store (auto-created)
└── README.md              # This file
```

## 💡 Use Cases

This assistant works great with any markdown documentation:

- **🎮 Game Development**: Unity, Unreal Engine docs
- **🌐 Web Development**: React, Django, Next.js docs
- **📱 Mobile Development**: Flutter, React Native docs
- **🔧 DevOps**: Kubernetes, Docker, Terraform docs
- **📊 Data Science**: Pandas, TensorFlow, PyTorch docs
- **🏢 Internal Docs**: Company wikis, API documentation, project guides

## 🔧 Advanced Usage

### Single Question Mode

```bash
python rag_chatbot.py "How do I set up authentication in this framework?"
```

### Custom Models

```bash
# Try different models for better quality or speed
ollama pull mistral        # Alternative LLM
ollama pull mxbai-embed-large  # Higher quality embeddings
```

### Chunk Size Optimization

Edit `load_documents.py` to adjust chunk sizes for your content:

```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,  # Increase for longer code examples
    chunk_overlap=250,  # Adjust context preservation
)
```

## 🛠️ Dependencies

Core dependencies managed via `uv`:

- **langchain** - LLM application framework
- **langchain-ollama** - Ollama integration
- **langchain-chroma** - ChromaDB vector store
- **python-dotenv** - Environment variable management

For a complete list, see `pyproject.toml`.

## 📊 Model Recommendations

| Use Case            | LLM Model    | Embedding Model   | Speed | Quality |
|---------------------|--------------|-------------------|-------|---------|
| **Fast responses**  | llama3.2     | nomic-embed-text  | ⚡⚡⚡   | ⭐⭐⭐     |
| **Balanced**        | mistral      | nomic-embed-text  | ⚡⚡    | ⭐⭐⭐⭐    |
| **High quality**    | llama3.2:8b  | mxbai-embed-large | ⚡     | ⭐⭐⭐⭐⭐   |
| **Code-heavy docs** | codellama:7b | nomic-embed-text  | ⚡⚡    | ⭐⭐⭐⭐    |

## 🐛 Troubleshooting

### Common Issues

**Ollama connection errors:**

```bash
# Make sure Ollama is running
ollama serve

# Check if models are installed
ollama list
```

**No documents found:**

- Verify `DOCS_DIRECTORY_PATH` in your `.env` file
- Ensure your directory contains `.md` files
- Check file permissions

**Slow responses:**

- Try a smaller/faster model like `llama3.2`
- Reduce `chunk_size` in document loading
- Use fewer retrieval chunks (`k=2` instead of `k=4`)

**Memory issues:**

- Reduce `chunk_size` and `chunk_overlap`
- Use smaller embedding models
- Clear and rebuild ChromaDB if corrupted

## 🙏 Acknowledgments

- [LangChain](https://python.langchain.com/) - Amazing framework for LLM apps
- [Ollama](https://ollama.ai/) - Local LLM inference made easy
- [ChromaDB](https://www.trychroma.com/) - Excellent vector database
- [uv](https://docs.astral.sh/uv/) - Fast Python package manager

---

**Happy coding! 🚀**