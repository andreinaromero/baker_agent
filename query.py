import os
import sys
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
try:
    from langchain.chains import create_retrieval_chain
    from langchain.chains.combine_documents import create_stuff_documents_chain
except ImportError:
    from langchain_classic.chains import create_retrieval_chain
    from langchain_classic.chains.combine_documents import create_stuff_documents_chain

# Import configuration
from config import (
    CHROMA_DB_PATH,
    EMBEDDING_PROVIDER,
    EMBEDDING_MODEL_NAME,
    OPENAI_API_KEY,
    OPENAI_MODEL_NAME,
    OLLAMA_MODEL_NAME,
    OLLAMA_BASE_URL
)

def get_embedding_model():
    """Returns the embedding model according to configuration."""
    if EMBEDDING_PROVIDER == 'openai':
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY must be set in .env to use openai embeddings.")
        return OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    else:
        # Default to local HuggingFace embeddings
        return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

def get_llm():
    """Returns the LLM instance according to configuration."""
    if EMBEDDING_PROVIDER == 'openai':
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY must be set in .env to use OpenAI LLM.")
        return ChatOpenAI(model=OPENAI_MODEL_NAME, openai_api_key=OPENAI_API_KEY)
    else:
        # Default to local Ollama LLM
        print(f"Connecting to local Ollama service at {OLLAMA_BASE_URL} (model: {OLLAMA_MODEL_NAME})...")
        return ChatOllama(model=OLLAMA_MODEL_NAME, base_url=OLLAMA_BASE_URL)

def main():
    if not os.path.exists(CHROMA_DB_PATH):
        print(f"Error: Vector database not found at '{CHROMA_DB_PATH}'. Please run 'python ingest.py' first.")
        sys.exit(1)

    # Load vector store
    embeddings = get_embedding_model()
    vectorstore = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    # Initialize LLM
    try:
        llm = get_llm()
    except Exception as e:
        print(f"Error initializing LLM: {e}")
        sys.exit(1)

    # Define prompt template for the baking assistant
    system_prompt = (
        "You are an expert, professional pastry chef and baker helper. "
        "Use the following pieces of retrieved context to answer the user question. "
        "If you don't know the answer, say that you don't know. "
        "Keep your answer concise, precise, and helpful. "
        "Reply in Spanish, as the user is a Spanish speaker.\n\n"
        "Context:\n{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    # Create RAG chain
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    print("\n--- Baking Assistant Initialized ---")
    print("Type your questions about baking and recipes below. Type 'exit' or 'quit' to end.\n")

    while True:
        try:
            query = input("Question: ")
            if query.strip().lower() in ['exit', 'quit']:
                print("Goodbye!")
                break
            
            if not query.strip():
                continue

            print("Thinking...")
            response = rag_chain.invoke({"input": query})

            print("\nAnswer:")
            print(response["answer"])
            print("\nSources used:")
            sources = set()
            for doc in response.get("context", []):
                filename = doc.metadata.get("filename", "Unknown")
                source_type = doc.metadata.get("source_type", "Unknown")
                sources.add(f"- {filename} ({source_type})")
            
            for source in sources:
                print(source)
            print("-" * 50 + "\n")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\nAn error occurred: {e}\n")

if __name__ == "__main__":
    main()
