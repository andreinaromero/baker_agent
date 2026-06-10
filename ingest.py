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
    TEMP_DATA_DIR,
    OLLAMA_MODEL_NAME,
    OLLAMA_BASE_URL
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
    """Downloads PDF files from the specified Google Drive folder, preserving subfolder names."""
    if not GOOGLE_DRIVE_FOLDER_ID:
        print("Google Drive Folder ID not configured. Skipping Google Drive...")
        return []

    service = get_google_drive_service()
    if not service:
        return []

    try:
        downloaded_files = []
        os.makedirs(TEMP_DATA_DIR, exist_ok=True)

        # 1. Download PDFs directly inside the root folder
        query_root = f"'{GOOGLE_DRIVE_FOLDER_ID}' in parents and mimeType='application/pdf' and trashed=false"
        results_root = service.files().list(q=query_root, fields="files(id, name)").execute()
        root_items = results_root.get('files', [])
        
        if root_items:
            print(f"Found {len(root_items)} PDF files in Google Drive root folder. Downloading...")
            for item in root_items:
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
                print(f"Downloaded: {file_name} from Google Drive root")

        # 2. Search for subfolders inside the root folder
        query_folders = f"'{GOOGLE_DRIVE_FOLDER_ID}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false"
        results_folders = service.files().list(q=query_folders, fields="files(id, name)").execute()
        folders = results_folders.get('files', [])

        for folder in folders:
            folder_id = folder['id']
            folder_name = folder['name']
            folder_local_dir = os.path.join(TEMP_DATA_DIR, folder_name)
            
            # Find PDFs inside this subfolder
            query_pdf = f"'{folder_id}' in parents and mimeType='application/pdf' and trashed=false"
            results_pdf = service.files().list(q=query_pdf, fields="files(id, name)").execute()
            pdf_items = results_pdf.get('files', [])

            if pdf_items:
                print(f"Found {len(pdf_items)} PDF files in Google Drive folder '{folder_name}'. Downloading...")
                os.makedirs(folder_local_dir, exist_ok=True)
                for item in pdf_items:
                    file_id = item['id']
                    file_name = item['name']
                    file_path = os.path.join(folder_local_dir, file_name)
                    
                    request = service.files().get_media(fileId=file_id)
                    fh = io.BytesIO()
                    downloader = MediaIoBaseDownload(fh, request)
                    done = False
                    while not done:
                        status, done = downloader.next_chunk()
                        
                    with open(file_path, 'wb') as f:
                        f.write(fh.getvalue())
                    
                    downloaded_files.append((file_path, "Google Drive"))
                    print(f"Downloaded: {folder_name}/{file_name} from Google Drive")

        return downloaded_files
    except Exception as e:
        print(f"Error downloading from Google Drive: {e}")
        return []

def download_from_dropbox():
    """Downloads PDF files from the specified Dropbox folder, preserving subfolder structure."""
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
        res = dbx.files_list_folder(folder, recursive=True)
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
            # Get path relative to the root Dropbox folder path
            rel_path = entry.path_display
            if folder:
                if rel_path.lower().startswith(folder.lower()):
                    rel_path = rel_path[len(folder):]
            
            rel_path = rel_path.lstrip('/')
            file_path = os.path.join(TEMP_DATA_DIR, rel_path)
            
            # Recreate subfolders locally
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            dbx.files_download_to_file(file_path, entry.path_lower)
            downloaded_files.append((file_path, "Dropbox"))
            print(f"Downloaded from Dropbox: {rel_path}")
            
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

def extract_recipe_metadata(filepath, text_content, parent_folder_name=None):
    """Extracts recipe title and creator using local Ollama model with subfolder/filename fallbacks."""
    filename = os.path.basename(filepath)
    # Default fallback values
    title = filename.replace('.pdf', '').replace('_', ' ').replace('-', ' ').strip().title()
    creator = parent_folder_name if parent_folder_name else "Desconocido"

    # Use local Ollama to get highly accurate metadata from the text
    try:
        # We only send the first 1500 chars to save time and tokens
        sample_text = text_content[:1500]
        
        # Call Ollama API directly (this is fast and avoids dependency issues)
        import urllib.request
        import json
        
        model = OLLAMA_MODEL_NAME
        url = f"{OLLAMA_BASE_URL}/api/generate"
        
        prompt = (
            f"Identify the recipe title and the author/creator/source of this baking document.\n"
            f"Filename: '{filename}'\n"
            f"Folder: '{parent_folder_name or 'None'}'\n"
            f"Document start text:\n\"\"\"\n{sample_text}\n\"\"\"\n\n"
            f"Respond ONLY with a JSON object containing keys 'title' and 'creator'. Do not include markdown code block formatting or explanations.\n"
            f"If the author/creator is not mentioned, use '{creator}' as the creator.\n"
            f"Ensure both values are in Spanish if possible (e.g. clean recipe name).\n"
            f"Example response: {{\"title\": \"Torta Húmeda de Chocolate\", \"creator\": \"Juliana Postres\"}}"
        )
        
        req_data = json.dumps({
            "model": model,
            "prompt": prompt,
            "stream": False,
            "format": "json"
        }).encode('utf-8')
        
        req = urllib.request.Request(url, data=req_data, headers={'Content-Type': 'application/json'}, method='POST')
        
        # Wait max 5 seconds so we don't hang if Ollama is slow
        with urllib.request.urlopen(req, timeout=5.0) as response:
            res_data = json.loads(response.read().decode())
            res_text = res_data.get("response", "").strip()
            res_json = json.loads(res_text)
            
            extracted_title = res_json.get("title", "").strip()
            extracted_creator = res_json.get("creator", "").strip()
            
            if extracted_title:
                title = extracted_title
            if extracted_creator:
                creator = extracted_creator
    except Exception as e:
        # Fallback if Ollama query fails
        print(f"Ollama metadata extraction skipped/failed for {filename}: {e}. Using fallback values.")
        
    return title, creator

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
    
    # Check if there are any pre-existing/manually placed PDFs recursively
    local_files = []
    for root, dirs, files in os.walk(TEMP_DATA_DIR):
        for f in files:
            if f.lower().endswith('.pdf'):
                local_files.append(os.path.join(root, f))
                
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
            
            # Find the parent folder name relative to temp_downloads/
            parent_dir = os.path.basename(os.path.dirname(filepath))
            parent_folder_name = parent_dir if parent_dir.lower() != os.path.basename(TEMP_DATA_DIR).lower() else None
            
            # Combine the text of the first page to pass to the metadata extractor
            first_page_text = loaded_docs[0].page_content if loaded_docs else ""
            
            # Extract metadata
            recipe_title, recipe_creator = extract_recipe_metadata(filepath, first_page_text, parent_folder_name)
            
            for doc in loaded_docs:
                doc.metadata['source_type'] = source
                doc.metadata['filename'] = filename
                doc.metadata['category'] = category
                doc.metadata['recipe_title'] = recipe_title
                doc.metadata['recipe_creator'] = recipe_creator
                
            documents.extend(loaded_docs)
            print(f"Successfully loaded: '{recipe_title}' by '{recipe_creator}' [Category: {category}]")
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
    
    # Delete existing DB directory to avoid merging deleted file metadata
    import shutil
    if os.path.exists(CHROMA_DB_PATH):
        print("Clearing existing vector database...")
        try:
            shutil.rmtree(CHROMA_DB_PATH)
        except Exception as e:
            print(f"Warning: Could not clear existing database directory: {e}")
            
    db = Chroma.from_documents(chunks, embeddings, persist_directory=CHROMA_DB_PATH)
    print("Database built successfully!")

if __name__ == "__main__":
    main()
