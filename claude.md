# Project Guidelines

- **Language Specifications**:
  - **Source Code**: All code (variables, functions, classes, comments, database schemas, etc.) must be written in English.
  - **User Interface**: The Streamlit web interface (`app.py`) must be completely in Spanish for a friendly user experience.
  - **Project Documentation**: All main user-facing documentation (e.g., `README.md`) must be written in Spanish.
  - **LLM Output**: The AI assistant model responses must always be in Spanish (configured via the system prompt).
- **Directory Scanning**: Do not perform wide, recursive, or system-wide directory scans (e.g. `find ~` or `find /`). Only work inside the project folder: `/Users/aromero/Projects/agente_repostero`.
- **Git workflow**: Work on a branch (currently `feature/baking-agent`), not directly on `main`/`master`. Only merge to `main`/`master` once everything is fully verified.

# Project Architecture & Goals

- **Goal**: Build a local AI Agent for a baking assistant using LangChain and a local vector database.
- **Data Sources**:
  - Google Drive (via GoogleDriveLoader or official Google API) to fetch baking PDFs.
  - Dropbox (via DropboxLoader or official SDK) to fetch baking PDFs.
  - Local PDFs in `temp_downloads/` (optional manual input).
- **Core Functionality**:
  1. **Ingest (`ingest.py`)**: Download PDFs, parse them using `pypdf`, extract metadata (e.g. category based on content/filename, like 'cakes', 'fillings', 'breads', etc.), split text into chunks with overlap, and store them in a local `Chroma` database.
  2. **Query/AI Agent (`query.py` / `app.py`)**:
     - Retrieve relevant context based on queries.
     - Answer user/subscriber questions in the persona of an expert pastry chef.
     - Filter and browse documents/recipes by categories (e.g., 'cakes', 'fillings', 'keto', etc.).
- **Tech Stack**:
  - Python 3.10+
  - LangChain & LangChain-Ollama (using local Ollama models like `llama3` or `phi3`, configured in `.env`).
  - ChromaDB (persisted locally under `./chroma_db`).
  - Streamlit (for a lightweight, interactive web UI to browse recipes and chat with the AI).
  - pypdf (for PDF text extraction).

## Roadmap

1. [x] Setup Project Structure, `.env.example`, `.env` and `requirements.txt`.
2. [x] Implement Google Drive & Dropbox integration to download PDFs to a temporary directory.
3. [x] Implement PDF extraction, metadata tagging (categorization), and embedding into a local `ChromaDB` instance.
4. [x] Build a local CLI query interface (`query.py`) using a local Ollama model to answer questions as an expert baker.
5. [x] Create a web dashboard using Streamlit (`app.py`) to search/filter recipes by categories and interact with the AI assistant.
