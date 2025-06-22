from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_chroma import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
import os
import sys
import dotenv

# Load environment variables
dotenv.load_dotenv()


class RAGChatbot:
    def __init__(self, chroma_path='./chroma_db', embedding_model='nomic-embed-text', llm_model='llama3.2'):
        """Initialize the RAG chatbot with ChromaDB and LLama 3.2"""

        print("🔧 Initializing RAG Chatbot...")

        # Check if ChromaDB exists
        if not os.path.exists(chroma_path):
            print(f"❌ ChromaDB not found at {chroma_path}")
            print("Please run the document loading script first!")
            sys.exit(1)

        # Initialize embeddings (same as used for creating the DB)
        print("📊 Loading embeddings...")
        self.embeddings = OllamaEmbeddings(
            model=embedding_model,
            base_url="http://localhost:11434"
        )

        # Load the vector store
        print("📚 Loading ChromaDB...")
        self.vectorstore = Chroma(
            persist_directory=chroma_path,
            embedding_function=self.embeddings
        )

        # Initialize the LLM
        print(f"🤖 Loading {llm_model}...")
        self.llm = OllamaLLM(
            model=llm_model,
            base_url="http://localhost:11434",
            temperature=0.1  # Lower temperature for more focused responses
        )

        # Set up conversation history (modern approach)
        self.chat_history = []

        # Create custom prompt template
        self.prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""You are a helpful assistant that answers questions based on the provided documentation context.

Context from documentation:
{context}

Question: {question}

Instructions:
- Answer based primarily on the provided context
- If the context doesn't contain enough information, say so clearly
- Be specific and reference relevant parts of the documentation
- For code examples, provide practical implementation details
- Keep answers concise but comprehensive

Answer:"""
        )

        # Create the conversational retrieval chain
        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 4}  # Retrieve top 4 most relevant chunks
            ),
            return_source_documents=True,
            combine_docs_chain_kwargs={"prompt": self.prompt_template}
        )

        print("✅ RAG Chatbot ready!")

    def ask_question(self, question):
        """Ask a question and get a response with sources"""

        print(f"\n🔍 Searching for relevant information...")

        # Get response from the chain using the modern invoke method
        response = self.qa_chain.invoke({
            "question": question,
            "chat_history": self.chat_history
        })

        answer = response["answer"]
        source_docs = response["source_documents"]

        # Update chat history for context in future questions
        self.chat_history.extend([
            ("Human", question),
            ("Assistant", answer)
        ])

        return answer, source_docs

    def print_response(self, question, answer, sources):
        """Pretty print the response with sources"""

        print(f"\n" + "=" * 80)
        print(f"❓ Question: {question}")
        print(f"=" * 80)
        print(f"🤖 Answer:\n{answer}")
        print(f"\n" + "-" * 80)
        print(f"📖 Sources:")

        for i, doc in enumerate(sources, 1):
            source_file = doc.metadata.get('source', 'Unknown')
            # Clean up the file path for display
            source_file = os.path.basename(source_file)

            print(f"\n{i}. 📄 {source_file}")
            # Show first 150 characters of the relevant content
            content_preview = doc.page_content.strip()[:150]
            print(f"   Preview: {content_preview}...")

        print(f"\n" + "=" * 80)

    def interactive_chat(self):
        """Start an interactive chat session"""

        print("\n🎯 RAG Chatbot Interactive Mode")
        print("Ask questions about your documentation!")
        print("Type 'quit', 'exit', or 'q' to stop\n")

        while True:
            try:
                question = input("👤 You: ").strip()
                if question.lower() in ['quit', 'exit', 'q', '']:
                    print("👋 Goodbye!")
                    break

                answer, sources = self.ask_question(question)
                self.print_response(question, answer, sources)

            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                continue


def main():
    """Main function to run the chatbot"""

    # You can customize these parameters
    CHROMA_PATH = os.getenv("CHROMA_PATH", './chroma_db')  # Path to your ChromaDB
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", 'nomic-embed-text')  # Embedding model for RAG
    LLM_MODEL = os.getenv("LLM_MODEL", 'llama3.2')  # LLM model for RAG

    try:
        # Initialize the chatbot
        chatbot = RAGChatbot(
            chroma_path=CHROMA_PATH,
            embedding_model=EMBEDDING_MODEL,
            llm_model=LLM_MODEL
        )

        # Check if running with a specific question
        if len(sys.argv) > 1:
            # Single question mode
            question = " ".join(sys.argv[1:])
            answer, sources = chatbot.ask_question(question)
            chatbot.print_response(question, answer, sources)
        else:
            # Interactive mode
            chatbot.interactive_chat()

    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Error initializing chatbot: {e}")
        print("Make sure Ollama is running and your models are installed!")


if __name__ == "__main__":
    main()
