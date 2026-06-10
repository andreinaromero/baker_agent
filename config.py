import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent

# Vector database configuration
CHROMA_DB_DIR = os.getenv("CHROMA_DB_DIR", "./chroma_db")
CHROMA_DB_PATH = str(BASE_DIR / CHROMA_DB_DIR)

# Embeddings and LLM Configuration
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "huggingface")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL_NAME = os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini")
OLLAMA_MODEL_NAME = os.getenv("OLLAMA_MODEL_NAME", "llama3")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Google Drive Configuration
GOOGLE_CREDENTIALS_FILE = os.getenv("GOOGLE_CREDENTIALS_FILE", "credentials.json")
GOOGLE_CREDENTIALS_PATH = str(BASE_DIR / GOOGLE_CREDENTIALS_FILE)
GOOGLE_TOKEN_FILE = os.getenv("GOOGLE_TOKEN_FILE", "token.json")
GOOGLE_TOKEN_PATH = str(BASE_DIR / GOOGLE_TOKEN_FILE)
GOOGLE_DRIVE_FOLDER_ID = os.getenv("GOOGLE_DRIVE_FOLDER_ID")

# Dropbox Configuration
DROPBOX_ACCESS_TOKEN = os.getenv("DROPBOX_ACCESS_TOKEN")
DROPBOX_APP_KEY = os.getenv("DROPBOX_APP_KEY")
DROPBOX_APP_SECRET = os.getenv("DROPBOX_APP_SECRET")
DROPBOX_REFRESH_TOKEN = os.getenv("DROPBOX_REFRESH_TOKEN")
DROPBOX_FOLDER_PATH = os.getenv("DROPBOX_FOLDER_PATH", "/recipes")

# Temporary Data Directory for downloads
TEMP_DATA_DIR = str(BASE_DIR / "temp_downloads")

def update_env(updates):
    """Updates key-value pairs in the local .env file, preserving comments."""
    env_path = BASE_DIR / ".env"
    if not env_path.exists():
        env_path.touch()
    
    lines = env_path.read_text().splitlines()
    new_lines = []
    keys_handled = set()
    
    for line in lines:
        line_strip = line.strip()
        if line_strip and not line_strip.startswith("#") and "=" in line_strip:
            key = line_strip.split("=", 1)[0].strip()
            if key in updates:
                new_lines.append(f"{key}={updates[key]}")
                keys_handled.add(key)
                continue
        new_lines.append(line)
        
    for key, val in updates.items():
        if key not in keys_handled:
            new_lines.append(f"{key}={val}")
            
    env_path.write_text("\n".join(new_lines) + "\n")
    # Reload environment variables override
    load_dotenv(override=True)
