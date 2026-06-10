import os
import sys
import streamlit as st
from pathlib import Path

# Set Streamlit Page Configuration (Must be the very first Streamlit command)
st.set_page_config(
    page_title="Agente Repostero - Asistente de Pastelería AI",
    page_icon="🍰",
    layout="wide",
    initial_sidebar_state="expanded"
)

from langchain_chroma import Chroma
from config import (
    CHROMA_DB_PATH,
    OLLAMA_MODEL_NAME,
    OLLAMA_BASE_URL,
    TEMP_DATA_DIR,
    GOOGLE_CREDENTIALS_PATH,
    GOOGLE_TOKEN_PATH,
    GOOGLE_DRIVE_FOLDER_ID,
    DROPBOX_APP_KEY,
    DROPBOX_APP_SECRET,
    DROPBOX_REFRESH_TOKEN,
    DROPBOX_FOLDER_PATH,
    update_env
)
from query import get_embedding_model, get_llm
from langchain_core.prompts import ChatPromptTemplate
try:
    from langchain.chains import create_retrieval_chain
    from langchain.chains.combine_documents import create_stuff_documents_chain
except ImportError:
    from langchain_classic.chains import create_retrieval_chain
    from langchain_classic.chains.combine_documents import create_stuff_documents_chain

# Import ingestion main function to trigger it from the UI
try:
    from ingest import main as run_ingestion
except ImportError:
    run_ingestion = None

# Custom styling for a warm, premium baking aesthetic
st.markdown("""
    <style>
        /* Base page background and font colors */
        .stApp {
            background-color: #fcf8f2;
            color: #43281c;
            font-family: 'Outfit', 'Inter', sans-serif;
        }
        
        /* Main title styling */
        .main-title {
            font-size: 2.8rem;
            font-weight: 800;
            color: #7f5539;
            text-align: center;
            margin-bottom: 5px;
            background: linear-gradient(90deg, #9c6644, #ddb892);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        
        .subtitle {
            font-size: 1.1rem;
            color: #b08968;
            text-align: center;
            margin-bottom: 30px;
            font-style: italic;
        }

        /* Sidebar custom styles */
        [data-testid="stSidebar"] {
            background-color: #ede0d4;
            border-right: 1px solid #ddb892;
        }
        
        /* Custom tabs styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 12px;
            background-color: #e6ccb2;
            padding: 8px;
            border-radius: 12px;
        }
        
        .stTabs [data-baseweb="tab"] {
            height: 48px;
            white-space: pre-wrap;
            background-color: transparent;
            border-radius: 8px;
            padding: 10px 24px;
            color: #7f5539;
            font-weight: 600;
            border: none;
            transition: all 0.3s ease;
        }
        
        .stTabs [aria-selected="true"] {
            background-color: #7f5539 !important;
            color: #ffffff !important;
            box-shadow: 0 4px 10px rgba(127, 85, 57, 0.2);
        }

        /* Recipe card layout styling */
        .recipe-card {
            background-color: #ffffff;
            padding: 22px 22px 10px 22px;
            border-radius: 16px 16px 0 0;
            box-shadow: 0 4px 15px rgba(127, 85, 57, 0.04);
            border-left: 6px solid #b08968;
            margin-bottom: 0px;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }
        
        .recipe-card h3 {
            margin: 0 0 4px 0;
            color: #7f5539;
            font-size: 1.35rem;
        }

        /* Category badges */
        .badge {
            display: inline-block;
            padding: 4px 10px;
            border-radius: 20px;
            font-size: 0.75rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            color: #ffffff;
            margin-right: 6px;
        }
        
        .badge-cakes { background-color: #e5989b; }
        .badge-fillings { background-color: #b5828c; }
        .badge-breads { background-color: #e07a5f; }
        .badge-keto { background-color: #81b29a; }
        .badge-breakfasts { background-color: #f2cc8f; }
        .badge-cupcakes { background-color: #cdb4db; }
        .badge-cookies { background-color: #a8dadc; }
        .badge-others { background-color: #6d597a; }

        .badge-source {
            background-color: #e6ccb2;
            color: #7f5539;
        }

        /* Chat speech bubbles */
        .chat-bubble {
            padding: 16px 20px;
            border-radius: 18px;
            margin-bottom: 12px;
            max-width: 85%;
            line-height: 1.5;
            font-size: 0.95rem;
        }
        
        .user-bubble {
            background-color: #ede0d4;
            color: #43281c;
            margin-left: auto;
            border-bottom-right-radius: 4px;
            border-left: 4px solid #b08968;
        }
        
        .assistant-bubble {
            background-color: #ffffff;
            color: #43281c;
            margin-right: auto;
            border-bottom-left-radius: 4px;
            border-left: 4px solid #7f5539;
            box-shadow: 0 3px 10px rgba(0,0,0,0.03);
        }
        
        .source-tag {
            font-size: 0.75rem;
            color: #9c6644;
            margin-top: 8px;
            font-weight: 500;
        }
        
        /* Expander custom border mapping for card bottoms */
        .stDetailsExpander {
            border-radius: 0 0 16px 16px !important;
            border: 1px solid #e6ccb2 !important;
            border-top: none !important;
            margin-bottom: 20px;
        }

        /* Custom legible button styles using Pink, Turquoise, Brown, and White */
        .stButton > button {
            background-color: #ffccd5 !important; /* Pastel pink */
            color: #43281c !important; /* Dark brown text for high readability */
            border: 2px solid #7f5539 !important; /* Brown border */
            border-radius: 10px !important;
            padding: 8px 16px !important;
            font-weight: 700 !important;
            font-size: 0.95rem !important;
            transition: all 0.3s ease !important;
            box-shadow: 0 4px 6px rgba(127, 85, 57, 0.05) !important;
            width: 100% !important;
        }
        
        .stButton > button:hover {
            background-color: #2ec4b6 !important; /* Turquoise on hover */
            color: #ffffff !important; /* White text on hover */
            border-color: #00a896 !important; /* Deep turquoise border */
            box-shadow: 0 6px 14px rgba(46, 196, 182, 0.25) !important;
            transform: translateY(-1px) !important;
        }

        .stButton > button:active {
            transform: translateY(1px) !important;
            box-shadow: 0 2px 4px rgba(46, 196, 182, 0.1) !important;
        }

        /* Sidebar buttons override (reindexing button) */
        [data-testid="stSidebar"] .stButton > button {
            background-color: #2ec4b6 !important; /* Turquoise */
            color: #ffffff !important; /* White text */
            border: 2px solid #00a896 !important; /* Deep turquoise border */
        }
        
        [data-testid="stSidebar"] .stButton > button:hover {
            background-color: #ffccd5 !important; /* Pink on hover */
            color: #43281c !important; /* Brown text on hover */
            border-color: #7f5539 !important; /* Brown border */
            box-shadow: 0 6px 14px rgba(255, 204, 213, 0.35) !important;
        }
    </style>
""", unsafe_allow_html=True)

# Helper function to get correct badge class
def get_category_badge_class(category):
    category = category.lower()
    valid_categories = ['cakes', 'fillings', 'breads', 'keto', 'breakfasts', 'cupcakes', 'cookies', 'others']
    if category in valid_categories:
        return f"badge-{category}"
    return "badge-others"

# Load the vector store
def get_vectorstore():
    if not os.path.exists(CHROMA_DB_PATH):
        return None
    try:
        embeddings = get_embedding_model()
        return Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embeddings)
    except Exception as e:
        st.error(f"Error al cargar la base de datos de vectores: {e}")
        return None

# Fetch unique documents from database
def get_indexed_documents(vectorstore):
    if not vectorstore:
        return []
    try:
        data = vectorstore.get(include=["metadatas"])
        metadatas = data.get("metadatas", [])
        
        unique_docs = {}
        for meta in metadatas:
            if not meta:
                continue
            filename = meta.get("filename")
            if filename:
                unique_docs[filename] = {
                    "category": meta.get("category", "others"),
                    "source_type": meta.get("source_type", "Local"),
                    "recipe_title": meta.get("recipe_title", filename.replace('.pdf', '').replace('_', ' ').replace('-', ' ').title()),
                    "recipe_creator": meta.get("recipe_creator", "Desconocido")
                }
        
        return [{"filename": name, **info} for name, info in unique_docs.items()]
    except Exception as e:
        st.error(f"Error al consultar metadatos: {e}")
        return []

# Sidebar UI
st.sidebar.markdown("<h2 style='text-align: center; color: #7f5539; margin-top: 0;'>🍰 Panel de Control</h2>", unsafe_allow_html=True)
st.sidebar.markdown("---")

# Database status info
db_exists = os.path.exists(CHROMA_DB_PATH)
st.sidebar.markdown(f"**Modelo LLM:** `{OLLAMA_MODEL_NAME}`")
st.sidebar.markdown(f"**Dirección Ollama:** `{OLLAMA_BASE_URL}`")

if db_exists:
    st.sidebar.success("✅ Base de Datos: Conectada")
else:
    st.sidebar.warning("⚠️ Base de Datos: No Encontrada")

st.sidebar.markdown("---")
st.sidebar.markdown("### 🔄 Sincronizar Documentos")
st.sidebar.info("Sincroniza e indexa los archivos PDF en tu Google Drive, Dropbox o en la carpeta local `temp_downloads/`.")

if st.sidebar.button("Sincronizar y Reindexar PDFs", use_container_width=True):
    if run_ingestion:
        with st.spinner("Indexando documentos, por favor espera..."):
            try:
                # Capture standard output to show feedback
                import contextlib
                from io import StringIO
                
                f = StringIO()
                with contextlib.redirect_stdout(f):
                    run_ingestion()
                output = f.getvalue()
                
                st.sidebar.success("¡Reindexación completada con éxito!")
                with st.sidebar.expander("Mostrar Registro de Ingesta"):
                    st.code(output)
                
                # Force refresh
                st.rerun()
            except Exception as e:
                st.sidebar.error(f"Error durante la reindexación: {e}")
    else:
        st.sidebar.error("No se pudo cargar el módulo de ingesta.")

# Sidebar Quick Presets
st.sidebar.markdown("---")
st.sidebar.markdown("### 💡 Preguntas Frecuentes")
presets = [
    "¿Cómo puedo hacer un relleno firme que soporte el peso de una torta de 3 pisos?",
    "Dame una opción de pan keto rápido para el desayuno.",
    "¿Qué recomendaciones hay para hornear cupcakes y evitar que queden secos?",
    "¿Cuáles son los pasos clave para un merengue perfecto?"
]

# Title and main description
st.markdown("<h1 class='main-title'>🍰 Agente Repostero AI</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Tu base de conocimiento local para repostería y pastelería profesional</p>", unsafe_allow_html=True)

# Main container
vectorstore = get_vectorstore()
documents = get_indexed_documents(vectorstore)

# Create layout tabs
tab_chat, tab_catalog, tab_settings = st.tabs(["💬 Chat y Ayuda de YouTube", "📚 Catálogo de Recetas", "⚙️ Configuración"])

# TAB 1: Chat and YouTube Helper
with tab_chat:
    st.markdown("### Consulta a tu Asistente Repostero")
    st.markdown("Copia y pega las preguntas de tus seguidores de YouTube o escribe tus propias consultas. El asistente responderá basándose en tus recetas y ebooks indexados.")
    
    # Quick preset buttons
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        if st.button(presets[0], key="preset_0", use_container_width=True):
            st.session_state.chat_input = presets[0]
    with col_p2:
        if st.button(presets[1], key="preset_1", use_container_width=True):
            st.session_state.chat_input = presets[1]
            
    col_p3, col_p4 = st.columns(2)
    with col_p3:
        if st.button(presets[2], key="preset_2", use_container_width=True):
            st.session_state.chat_input = presets[2]
    with col_p4:
        if st.button(presets[3], key="preset_3", use_container_width=True):
            st.session_state.chat_input = presets[3]

    # Category Filter for Search
    categories_list = ["Todas"]
    if documents:
        categories_list.extend(sorted(list(set(doc["category"] for doc in documents))))
    
    category_filter = st.selectbox(
        "Filtrar la búsqueda por categoría:",
        options=categories_list,
        help="Elige una categoría específica para buscar solo en esas recetas."
    )

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        role_class = "user-bubble" if message["role"] == "user" else "assistant-bubble"
        content_html = f"<div class='chat-bubble {role_class}'><b>{message['author']}:</b><br>{message['content']}"
        if "sources" in message and message["sources"]:
            sources_str = ", ".join(message["sources"])
            content_html += f"<div class='source-tag'>📚 Fuentes: {sources_str}</div>"
        content_html += "</div>"
        st.markdown(content_html, unsafe_allow_html=True)

    # Chat input box
    if "chat_input" not in st.session_state:
        st.session_state.chat_input = ""

    def process_chat():
        user_query = st.session_state.chat_input_val.strip()
        if not user_query:
            return
        
        # Add user message to state
        st.session_state.messages.append({
            "role": "user",
            "author": "Tú",
            "content": user_query
        })
        
        # Check if database is initialized
        if not vectorstore:
            st.session_state.messages.append({
                "role": "assistant",
                "author": "Asistente Pastelero",
                "content": "Lo siento, la base de datos de recetas no está creada. Por favor, coloca algunos archivos PDF en la carpeta `temp_downloads/` y pulsa 'Sincronizar y Reindexar PDFs' en el panel lateral."
            })
            return

        # Query the RAG
        try:
            # Build retriever search filters if set
            search_kwargs = {"k": 4}
            if category_filter and category_filter != "Todas":
                search_kwargs["filter"] = {"category": category_filter.lower()}
            
            retriever = vectorstore.as_retriever(search_kwargs=search_kwargs)
            llm = get_llm()
            
            # Setup Prompt
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
            
            question_answer_chain = create_stuff_documents_chain(llm, prompt)
            rag_chain = create_retrieval_chain(retriever, question_answer_chain)
            
            # Call model
            response = rag_chain.invoke({"input": user_query})
            
            # Gather sources
            sources = set()
            for doc in response.get("context", []):
                recipe_title = doc.metadata.get("recipe_title")
                recipe_creator = doc.metadata.get("recipe_creator")
                if recipe_title and recipe_creator:
                    sources.add(f"{recipe_title} (por {recipe_creator})")
                else:
                    sources.add(doc.metadata.get("filename", "Unknown"))
                
            st.session_state.messages.append({
                "role": "assistant",
                "author": "Asistente Pastelero",
                "content": response["answer"],
                "sources": list(sources)
            })
        except Exception as e:
            st.session_state.messages.append({
                "role": "assistant",
                "author": "Asistente Pastelero",
                "content": f"Ocurrió un error al procesar tu pregunta: {e}"
            })

    # Display a text input that uses a callback
    st.text_input(
        "Escribe tu pregunta o pega el mensaje de tu suscriptor de YouTube aquí:",
        key="chat_input_val",
        value=st.session_state.chat_input,
        on_change=process_chat,
        placeholder="¿Cómo evito que se baje el bizcocho?"
    )
    
    # Clear Chat Button
    if st.button("Limpiar Conversación"):
        st.session_state.messages = []
        st.rerun()

# TAB 2: Library Catalog
with tab_catalog:
    st.markdown("### 📚 Recetas y Ebooks Indexados")
    st.markdown("A continuación verás el catálogo de todos los documentos indexados en tu base de datos local. Puedes filtrar por categoría o buscar por nombre de archivo o título de receta.")
    
    if not documents:
        st.warning("Aún no se han indexado documentos. Sube tus PDFs a Dropbox/Drive, o cópialos directamente a la carpeta `temp_downloads/` y haz clic en 'Sincronizar y Reindexar PDFs' en la barra lateral.")
    else:
        # Search and Category filter for the list
        col_search, col_cat = st.columns([2, 1])
        with col_search:
            search_query = st.text_input("🔍 Buscar recetas por nombre o creador:", placeholder="ej. Torta de Vainilla, Juliana Postres...")
        with col_cat:
            catalog_categories = ["Todas"] + sorted(list(set(doc["category"] for doc in documents)))
            selected_cat = st.selectbox("📂 Filtrar catálogo por categoría:", options=catalog_categories)
            
        # Filter documents based on inputs
        filtered_docs = documents
        if search_query:
            filtered_docs = [
                d for d in filtered_docs 
                if search_query.lower() in d["filename"].lower() or 
                   search_query.lower() in d["recipe_title"].lower() or 
                   search_query.lower() in d["recipe_creator"].lower()
            ]
        if selected_cat != "Todas":
            filtered_docs = [d for d in filtered_docs if d["category"].lower() == selected_cat.lower()]
            
        # Display document list
        if not filtered_docs:
            st.info("Ningún documento coincide con los criterios de búsqueda.")
        else:
            st.markdown(f"Mostrando **{len(filtered_docs)}** recetas:")
            
            # Display documents in grid format
            cols_per_row = 3
            for i in range(0, len(filtered_docs), cols_per_row):
                cols = st.columns(cols_per_row)
                for j in range(cols_per_row):
                    if i + j < len(filtered_docs):
                        doc = filtered_docs[i + j]
                        badge_class = get_category_badge_class(doc['category'])
                        
                        card_html = f"""
                        <div class="recipe-card">
                            <h3>{doc['recipe_title']}</h3>
                            <div style="font-size: 0.95rem; color: #7f5539; margin-bottom: 12px; font-weight: 500;">
                                🧑‍🍳 Creador: {doc['recipe_creator']}
                            </div>
                            <div>
                                <span class="badge {badge_class}">{doc['category']}</span>
                                <span class="badge badge-source">{doc['source_type']}</span>
                            </div>
                        </div>
                        """
                        with cols[j]:
                            st.markdown(card_html, unsafe_allow_html=True)
                            with st.container():
                                with st.expander("🔍 Ver detalles del archivo PDF", expanded=False):
                                    st.markdown(f"**Archivo PDF:** `{doc['filename']}`")
                                    st.markdown(f"**Fuente de Origen:** {doc['source_type']}")

# TAB 3: Settings Configuration
with tab_settings:
    st.markdown("### ⚙️ Configuración del Agente y Fuentes Nube")
    st.markdown("Configura las credenciales de tus fuentes de nube (Google Drive y Dropbox) y selecciona el modelo de Ollama que deseas utilizar.")
    
    col_settings_left, col_settings_right = st.columns(2)
    
    with col_settings_left:
        st.markdown("#### 1. Conexión a Google Drive")
        
        # Check Google credentials status
        google_creds_exist = os.path.exists(GOOGLE_CREDENTIALS_PATH)
        google_token_exist = os.path.exists(GOOGLE_TOKEN_PATH)
        
        if google_creds_exist:
            st.success("✅ Archivo `credentials.json` cargado correctamente.")
        else:
            st.warning("⚠️ No se ha detectado el archivo `credentials.json`. Súbelo a continuación.")
            
        if google_token_exist:
            st.info("ℹ️ Sesión OAuth autorizada activa (archivo `token.json` presente).")
            
        # File uploader for credentials.json
        credentials_file = st.file_uploader(
            "Cargar archivo credentials.json de Google Cloud:", 
            type=["json"],
            help="Sube el archivo JSON que descargaste de la Google Cloud Console para activar la API de Drive."
        )
        
        drive_folder_id = st.text_input(
            "ID de la Carpeta de Google Drive con Recetas:",
            value=GOOGLE_DRIVE_FOLDER_ID or "",
            placeholder="Introduce el ID largo de la carpeta de Drive..."
        )
        
        st.markdown("---")
        st.markdown("#### 2. Conexión a Dropbox")
        
        dbx_app_key = st.text_input(
            "Dropbox App Key:",
            value=DROPBOX_APP_KEY or "",
            placeholder="App Key de tu aplicación Dropbox..."
        )
        
        dbx_app_secret = st.text_input(
            "Dropbox App Secret:",
            value=DROPBOX_APP_SECRET or "",
            type="password",
            placeholder="App Secret de tu aplicación Dropbox..."
        )
        
        dbx_refresh_token = st.text_input(
            "Dropbox Refresh Token:",
            value=DROPBOX_REFRESH_TOKEN or "",
            type="password",
            placeholder="Refresh Token para tokens de larga duración..."
        )
        
        dbx_folder_path = st.text_input(
            "Ruta de la Carpeta en Dropbox:",
            value=DROPBOX_FOLDER_PATH or "/recetas",
            placeholder="ej. /recetas o /"
        )
        
    with col_settings_right:
        st.markdown("#### 3. Cerebro del Asistente (Ollama)")
        
        # Fetch local Ollama models list
        ollama_models = ["llama3", "llama3.2", "phi3"]
        try:
            import urllib.request
            import json
            with urllib.request.urlopen(f"{OLLAMA_BASE_URL}/api/tags", timeout=1.5) as response:
                tags_data = json.loads(response.read().decode())
                local_names = [m["name"] for m in tags_data.get("models", [])]
                if local_names:
                    ollama_models = local_names
        except Exception:
            st.caption("⚠️ No se pudo conectar a Ollama para listar los modelos. Asegúrate de que Ollama está abierto.")
            
        selected_model = st.selectbox(
            "Seleccionar Modelo Local (Ollama):",
            options=ollama_models,
            index=ollama_models.index(OLLAMA_MODEL_NAME) if OLLAMA_MODEL_NAME in ollama_models else 0,
            help="El modelo que usará el chat para responderte."
        )
        
        ollama_base_url = st.text_input(
            "Dirección URL de Ollama:",
            value=OLLAMA_BASE_URL,
            placeholder="Normalmente http://localhost:11434"
        )
        
        st.markdown("<br><br>", unsafe_allow_html=True)
        
        # Save Configuration Button
        if st.button("💾 Guardar Configuración", use_container_width=True):
            config_updates = {
                "GOOGLE_DRIVE_FOLDER_ID": drive_folder_id.strip(),
                "DROPBOX_APP_KEY": dbx_app_key.strip(),
                "DROPBOX_APP_SECRET": dbx_app_secret.strip(),
                "DROPBOX_REFRESH_TOKEN": dbx_refresh_token.strip(),
                "DROPBOX_FOLDER_PATH": dbx_folder_path.strip(),
                "OLLAMA_MODEL_NAME": selected_model,
                "OLLAMA_BASE_URL": ollama_base_url.strip()
            }
            
            # Save credentials file if uploaded
            credentials_saved = False
            if credentials_file is not None:
                try:
                    creds_path = Path(GOOGLE_CREDENTIALS_PATH)
                    creds_path.parent.mkdir(parents=True, exist_ok=True)
                    creds_path.write_bytes(credentials_file.read())
                    credentials_saved = True
                except Exception as e:
                    st.error(f"Error al guardar credentials.json: {e}")
            
            try:
                update_env(config_updates)
                
                success_msg = "¡Configuración de variables guardada con éxito en el archivo `.env`!"
                if credentials_saved:
                    success_msg += " Además, se guardó el archivo `credentials.json` correctamente."
                
                st.success(success_msg)
                
                # Delay for a moment to let the user see the success, then rerun
                import time
                time.sleep(1.5)
                st.rerun()
            except Exception as e:
                st.error(f"Error al guardar configuración: {e}")
