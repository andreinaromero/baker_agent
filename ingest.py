import os
import io
import sys
from pathlib import Path
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
import dropbox
from dropbox.exceptions import AuthError

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

# Import configuration
from config import (
    CHROMA_DB_PATH,
    EMBEDDING_PROVIDER,
    EMBEDDING_MODEL_NAME,
    OPENAI_API_KEY,
    GOOGLE_CREDENTIALS_PATH,
    GOOGLE_TOKEN_PATH,
    GOOGLE_DRIVE_FOLDER_ID,
    DROPBOX_ACCESS_TOKEN,
    DROPBOX_APP_KEY,
    DROPBOX_APP_SECRET,
    DROPBOX_REFRESH_TOKEN,
    DROPBOX_FOLDER_PATH,
    TEMP_DATA_DIR
)

# Scopes required for Google Drive API
GOOGLE_SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

def get_google_drive_service():
    """Initializes and returns a Google Drive service client."""
    if not os.path.exists(GOOGLE_CREDENTIALS_PATH):
        print(f"Warning: Google credentials file not found at '{GOOGLE_CREDENTIALS_PATH}'. Skipping Google Drive...")
        return None

    creds = None
    if os.path.exists(GOOGLE_TOKEN_PATH):
        creds = Credentials.from_authorized_user_file(GOOGLE_TOKEN_PATH, GOOGLE_SCOPES)
    
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
            except Exception as e:
                print(f"Error refreshing Google token: {e}")
                creds = None
        
        if not creds:
            try:
                flow = InstalledAppFlow.from_client_secrets_file(GOOGLE_CREDENTIALS_PATH, GOOGLE_SCOPES)
                creds = flow.run_local_server(port=0)
            except Exception as e:
                print(f"Failed to run Google local auth server: {e}")
                return None
            
        with open(GOOGLE_TOKEN_PATH, 'w') as token_file:
            token_file.write(creds.to_json())
            
    return build('drive', 'v3', credentials=creds)

def download_from_google_drive():
    """Downloads PDF files from the specified Google Drive folder."""
    if not GOOGLE_DRIVE_FOLDER_ID:
        print("Google Drive Folder ID not configured. Skipping Google Drive...")
        return []

    service = get_google_drive_service()
    if not service:
        return []

    try:
        query = f"'{GOOGLE_DRIVE_FOLDER_ID}' in parents and mimeType='application/pdf' and trashed=false"
        results = service.files().list(q=query, fields="files(id, name)").execute()
        items = results.get('files', [])

        if not items:
            print("No PDF files found in the specified Google Drive folder.")
            return []

        print(f"Found {len(items)} PDF files in Google Drive. Downloading...")
        os.makedirs(TEMP_DATA_DIR, exist_ok=True)
        downloaded_files = []

        for item in items:
            file_id = item['id']
            file_name = item['name']
            file_path = os.path.join(TEMP_DATA_DIR, file_name)
            
            request = service.files().get_media(fileId=file_id)
            fh = io.BytesIO()
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            while not done:
                status, done = downloader.next_chunk()
                
            with open(file_path, 'wb') as f:
                f.write(fh.getvalue())
            
            downloaded_files.append((file_path, "Google Drive"))
            print(f"Downloaded: {file_name} from Google Drive")
            
        return downloaded_files
    except Exception as e:
        print(f"Error downloading from Google Drive: {e}")
        return []

def download_from_dropbox():
    """Downloads PDF files from the specified Dropbox folder."""
    if not DROPBOX_ACCESS_TOKEN and not (DROPBOX_APP_KEY and DROPBOX_APP_SECRET and DROPBOX_REFRESH_TOKEN):
        print("Dropbox credentials not configured. Skipping Dropbox...")
        return []

    try:
        if DROPBOX_APP_KEY and DROPBOX_APP_SECRET and DROPBOX_REFRESH_TOKEN:
            dbx = dropbox.Dropbox(
                oauth2_refresh_token=DROPBOX_REFRESH_TOKEN,
                app_key=DROPBOX_APP_KEY,
                app_secret=DROPBOX_APP_SECRET
            )
        else:
            dbx = dropbox.Dropbox(DROPBOX_ACCESS_TOKEN)

        folder = DROPBOX_FOLDER_PATH if DROPBOX_FOLDER_PATH != '/' else ''
        res = dbx.files_list_folder(folder)
        files = [entry for entry in res.entries if isinstance(entry, dropbox.files.FileMetadata) and entry.name.lower().endswith('.pdf')]
        
        while res.has_more:
            res = dbx.files_list_folder_continue(res.cursor)
            files.extend([entry for entry in res.entries if isinstance(entry, dropbox.files.FileMetadata) and entry.name.lower().endswith('.pdf')])
            
        if not files:
            print("No PDF files found in the specified Dropbox folder.")
            return []
            
        print(f"Found {len(files)} PDF files in Dropbox. Downloading...")
        os.makedirs(TEMP_DATA_DIR, exist_ok=True)
        downloaded_files = []
        
        for entry in files:
            file_path = os.path.join(TEMP_DATA_DIR, entry.name)
            dbx.files_download_to_file(file_path, entry.path_lower)
            downloaded_files.append((file_path, "Dropbox"))
            print(f"Downloaded from Dropbox: {entry.name}")
            
        return downloaded_files
    except Exception as e:
        print(f"Error downloading from Dropbox: {e}")
        return []

def categorize_filename(filename):
    """Categorizes a file based on its name."""
    name_lower = filename.lower()
    if any(keyword in name_lower for keyword in ['cake', 'torta', 'pastel', 'bizcocho']):
        return 'cakes'
    elif any(keyword in name_lower for keyword in ['filling', 'relleno', 'crema', 'frosting', 'sauce']):
        return 'fillings'
    elif any(keyword in name_lower for keyword in ['bread', 'pan', 'masa', 'dough']):
        return 'breads'
    elif any(keyword in name_lower for keyword in ['keto', 'low carb', 'sin azucar']):
        return 'keto'
    elif any(keyword in name_lower for keyword in ['breakfast', 'desayuno', 'brunch']):
        return 'breakfasts'
    elif any(keyword in name_lower for keyword in ['cupcake', 'muffin', 'magdalena']):
        return 'cupcakes'
    elif any(keyword in name_lower for keyword in ['cookie', 'galleta']):
        return 'cookies'
    else:
        return 'others'

def get_embedding_model():
    """Returns the embedding model according to configuration."""
    if EMBEDDING_PROVIDER == 'openai':
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY must be set in .env to use openai embeddings.")
        return OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    else:
        print(f"Initializing local embeddings model: {EMBEDDING_MODEL_NAME}...")
        return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

def main():
    all_files = []
    
    os.makedirs(TEMP_DATA_DIR, exist_ok=True)
    
    # Try downloading from cloud
    all_files.extend(download_from_google_drive())
    all_files.extend(download_from_dropbox())
    
    # Also check if there are any pre-existing/manually placed PDFs in the directory
    local_files = [os.path.join(TEMP_DATA_DIR, f) for f in os.listdir(TEMP_DATA_DIR) if f.lower().endswith('.pdf')]
    existing_paths = {f[0] for f in all_files}
    for lf in local_files:
        if lf not in existing_paths:
            all_files.append((lf, "Local"))
            
    if not all_files:
        print(f"No PDF files found in '{TEMP_DATA_DIR}' or cloud storage. Please place some PDFs there or check configuration.")
        return

    # Extract Text and Load Documents
    print(f"Processing {len(all_files)} PDF files...")
    documents = []
    for filepath, source in all_files:
        try:
            loader = PyPDFLoader(filepath)
            loaded_docs = loader.load()
            filename = os.path.basename(filepath)
            category = categorize_filename(filename)
            for doc in loaded_docs:
                doc.metadata['source_type'] = source
                doc.metadata['filename'] = filename
                doc.metadata['category'] = category
            documents.extend(loaded_docs)
            print(f"Successfully loaded: {filename} under category: {category}")
        except Exception as e:
            print(f"Error loading {filepath}: {e}")

    if not documents:
        print("No document content loaded.")
        return

    # Split Text into Chunks
    print("Splitting text into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks.")

    # Initialize Embeddings
    embeddings = get_embedding_model()

    # Save to Local Vector Database
    print(f"Saving to local Vector DB at: {CHROMA_DB_PATH}...")
    db = Chroma.from_documents(chunks, embeddings, persist_directory=CHROMA_DB_PATH)
    print("Database built successfully!")

if __name__ == "__main__":
    main()
