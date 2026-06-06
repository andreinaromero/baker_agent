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
    TEMP_DATA_DIR
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
            padding: 22px;
            border-radius: 16px;
            box-shadow: 0 4px 15px rgba(127, 85, 57, 0.06);
            border-left: 6px solid #b08968;
            margin-bottom: 20px;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }
        
        .recipe-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(127, 85, 57, 0.12);
        }
        
        .recipe-card h3 {
            margin: 0 0 10px 0;
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
                    "source_type": meta.get("source_type", "Local")
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
tab_chat, tab_catalog = st.tabs(["💬 Chat y Ayuda de YouTube", "📚 Catálogo de Recetas"])

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
                filename = doc.metadata.get("filename", "Unknown")
                sources.add(filename)
                
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
    st.markdown("A continuación verás el catálogo de todos los documentos indexados en tu base de datos local. Puedes filtrar por categoría o buscar por nombre de archivo.")
    
    if not documents:
        st.warning("Aún no se han indexado documentos. Sube tus PDFs a Dropbox/Drive, o cópialos directamente a la carpeta `temp_downloads/` y haz clic en 'Sincronizar y Reindexar PDFs' en la barra lateral.")
    else:
        # Search and Category filter for the list
        col_search, col_cat = st.columns([2, 1])
        with col_search:
            search_query = st.text_input("🔍 Buscar documentos por nombre:", placeholder="ej. bizcocho, keto...")
        with col_cat:
            catalog_categories = ["Todas"] + sorted(list(set(doc["category"] for doc in documents)))
            selected_cat = st.selectbox("📂 Filtrar catálogo por categoría:", options=catalog_categories)
            
        # Filter documents based on inputs
        filtered_docs = documents
        if search_query:
            filtered_docs = [d for d in filtered_docs if search_query.lower() in d["filename"].lower()]
        if selected_cat != "Todas":
            filtered_docs = [d for d in filtered_docs if d["category"].lower() == selected_cat.lower()]
            
        # Display document list
        if not filtered_docs:
            st.info("Ningún documento coincide con los criterios de búsqueda.")
        else:
            st.markdown(f"Mostrando **{len(filtered_docs)}** documentos:")
            
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
                            <h3>{doc['filename']}</h3>
                            <div>
                                <span class="badge {badge_class}">{doc['category']}</span>
                                <span class="badge badge-source">{doc['source_type']}</span>
                            </div>
                        </div>
                        """
                        cols[j].markdown(card_html, unsafe_allow_html=True)
