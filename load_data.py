from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Load documents
DOCS_DIRECTORY_PATH = os.getenv("DOCS_DIRECTORY_PATH")

if not DOCS_DIRECTORY_PATH:
    print("❌ DOCS_DIRECTORY_PATH not found in environment variables!")
    print("Please set DOCS_DIRECTORY_PATH in your .env file")
    exit(1)

if not os.path.exists(DOCS_DIRECTORY_PATH):
    print(f"❌ Directory not found: {DOCS_DIRECTORY_PATH}")
    print("Please check your DOCS_DIRECTORY_PATH in the .env file")
    exit(1)

print(f"📁 Loading documents from: {DOCS_DIRECTORY_PATH}")
loader = DirectoryLoader(DOCS_DIRECTORY_PATH, glob="**/*.md")
documents = loader.load()

print(f"📄 Number of documents loaded: {len(documents)}")

if len(documents) == 0:
    print("⚠️  No .md files found in the specified directory!")
    print("Please check that your directory contains markdown files.")
    exit(1)

# Split documents into chunks - optimized for software documentation
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,  # Larger to keep code examples intact
    chunk_overlap=250,  # More overlap to preserve context
    length_function=len,
    separators=[  # Markdown-aware splitting
        "\n\n",  # Paragraph breaks
        "\n",  # Line breaks
        "```",  # Code block boundaries
        " ",  # Words
        ""  # Characters
    ]
)

# Split the documents
texts = text_splitter.split_documents(documents)
print(f"✂️  Number of chunks created: {len(texts)}")

# Set up Ollama embeddings
# Make sure you have an embedding model installed in Ollama
# Popular options: 'nomic-embed-text', 'mxbai-embed-large', 'all-minilm'
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

print(f"🔗 Using embedding model: {EMBEDDING_MODEL}")
embeddings = OllamaEmbeddings(
    model=EMBEDDING_MODEL,
    base_url=OLLAMA_BASE_URL
)

# Create or load ChromaDB vector store
CHROMA_PATH = os.getenv("CHROMA_PATH", './chroma_db')

# Check if database already exists
if os.path.exists(CHROMA_PATH) and os.listdir(CHROMA_PATH):
    print("🔄 Loading existing ChromaDB...")
    vectorstore = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings
    )

    # Optional: Add new documents (checks for duplicates by default)
    print("🔍 Checking for new documents to add...")
    existing_sources = set()

    # Get existing document sources
    try:
        existing_docs = vectorstore.get()
        if existing_docs['metadatas']:
            existing_sources = {meta.get('source', '') for meta in existing_docs['metadatas']}

        print(f"📊 Found {len(existing_sources)} existing document sources in database")
    except Exception as e:
        print(f"⚠️  Warning: Could not retrieve existing documents: {e}")
        existing_sources = set()

    # Filter out documents that are already in the database
    new_texts = [doc for doc in texts if doc.metadata.get('source', '') not in existing_sources]

    if new_texts:
        print(f"➕ Adding {len(new_texts)} new document chunks...")
        try:
            vectorstore.add_documents(new_texts)
            print("✅ New documents added successfully!")
        except Exception as e:
            print(f"❌ Error adding new documents: {e}")
    else:
        print("ℹ️  No new documents to add.")

else:
    print("🆕 Creating new ChromaDB...")
    try:
        vectorstore = Chroma.from_documents(
            documents=texts,
            embedding=embeddings,
            persist_directory=CHROMA_PATH
        )
        print("✅ Documents successfully loaded into ChromaDB!")
    except Exception as e:
        print(f"❌ Error creating ChromaDB: {e}")
        print("Make sure Ollama is running and your embedding model is installed!")
        exit(1)

# Optional: Test the vector store with a sample query
print("\n🧪 Testing the vector store...")
try:
    test_query = "documentation"
    test_results = vectorstore.similarity_search(test_query, k=2)
    print(f"✅ Vector store test successful! Found {len(test_results)} results for '{test_query}'")

    if test_results:
        print("\n📝 Sample result:")
        sample_doc = test_results[0]
        source_file = os.path.basename(sample_doc.metadata.get('source', 'Unknown'))
        content_preview = sample_doc.page_content.strip()[:100]
        print(f"   📄 {source_file}")
        print(f"   💬 {content_preview}...")

except Exception as e:
    print(f"⚠️  Vector store test failed: {e}")

print(f"\n🎉 Setup complete! ChromaDB saved at: {CHROMA_PATH}")
print("You can now run your RAG chatbot script!")
