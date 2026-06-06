# Agente Repostero AI - Asistente Local de Pastelería (RAG)

Este es un asistente inteligente local basado en Recuperación de Información Aumentada por Generación (RAG). Está diseñado para ayudarte a organizar tus recetas, ebooks y apuntes técnicos en PDF provenientes de tus cursos y clases, y para ayudarte a responder las preguntas de tu canal de YouTube de manera profesional utilizando tu propia base de conocimiento privada.

---

## Características Principales

- **Privacidad y Gratuidad 100% Local**: Utiliza **Ollama** con el modelo `llama3` para procesar consultas y **HuggingFace** (`all-MiniLM-L6-v2`) para generar embeddings vectoriales de forma local y gratuita, sin necesidad de enviar tus datos a servicios externos de pago.
- **Ingesta Multi-fuente**: Descarga y procesa PDFs directamente desde carpetas de **Google Drive**, **Dropbox** o una carpeta local (`temp_downloads/`).
- **Clasificación y Metadatos Dinámicos**: Clasifica automáticamente tus documentos en categorías (como *cakes*, *fillings*, *keto*, *breads*, etc.) analizando el nombre de los archivos.
- **Interfaz Web Interactiva (Streamlit)**: Un panel visual moderno adaptado a una estética repostera que te permite:
  1. Explorar de forma gráfica tu biblioteca de archivos indexados por categoría.
  2. Hacer consultas al asistente con un historial interactivo en tiempo real y atajos de preguntas comunes.
  3. Filtrar las respuestas para que busquen información solo dentro de una categoría específica (por ejemplo, buscar solo recetas *keto*).
  4. Sincronizar y reindexar todos tus archivos con un solo clic.

---

## Estructura del Proyecto

El proyecto está diseñado con una estructura plana y organizada en el directorio raíz:

- `app.py`: Código de la aplicación de Streamlit (interfaz gráfica del usuario).
- `ingest.py`: Script de ingesta que descarga PDFs de Drive/Dropbox/Local, los procesa, genera los fragmentos de texto (chunks) y construye la base de datos de vectores.
- `query.py`: Interfaz de consola rápida para chatear con el asistente directamente desde la terminal.
- `config.py`: Gestor centralizado de variables de configuración cargadas del archivo `.env`.
- `.env`: Archivo de configuración local donde almacenas las credenciales y rutas de tus bases de datos (no se sube al control de versiones).
- `requirements.txt`: Lista de dependencias de Python requeridas para ejecutar el proyecto.
- `claude.md`: Archivo con las pautas técnicas de desarrollo del proyecto.

---

## Requisitos Previos

1. **Python 3.10 o superior** instalado en tu Mac.
2. **Ollama** instalado y ejecutándose en segundo plano.

---

## Instalación y Configuración

1. **Instalar Dependencias de Python**:
   Abre tu terminal en la carpeta del proyecto y ejecuta:
   ```bash
   pip install -r requirements.txt
   ```

2. **Descargar el Modelo en Ollama**:
   Asegúrate de que la aplicación Ollama esté abierta en tu Mac y ejecuta el siguiente comando en la terminal para descargar el cerebro de tu asistente (`llama3`):
   ```bash
   ollama pull llama3
   ```

3. **Configurar las Variables de Entorno (`.env`)**:
   - Duplica el archivo `.env.example` y renómbralo como `.env`.
   - Si deseas usar Google Drive y Dropbox, completa los campos correspondientes con tus claves API. Si prefieres trabajar de forma puramente local, puedes dejar los valores por defecto y colocar tus PDFs en la carpeta `temp_downloads/`.

---

## Cómo Utilizar el Proyecto

### 1. Ingesta Inicial de Recetas
Si deseas procesar tus archivos PDF para guardarlos en la base de datos vectorial local, ejecuta:
```bash
python3 ingest.py
```
*Nota: Este comando creará la base de datos local en la carpeta `./chroma_db/`.*

### 2. Iniciar el Dashboard Visual (Recomendado)
Para arrancar el panel interactivo web en tu navegador:
```bash
streamlit run app.py
```
Una vez ejecutado, se abrirá automáticamente una pestaña en tu navegador en:
[http://localhost:8501](http://localhost:8501)

### 3. Consultas Rápidas por Consola
Si deseas hacer una pregunta rápida directamente desde la terminal sin abrir el navegador:
```bash
python3 query.py
```

---

## ¿Para qué sirve el botón "Sincronizar y Reindexar PDFs"?

En la barra lateral izquierda del panel web verás el botón **Sincronizar y Reindexar PDFs** (Sync). Debes utilizar este botón en los siguientes casos:

1. **Cuando añadas nuevos archivos PDF** a la carpeta `temp_downloads/` o realices cambios en los archivos existentes.
2. **Cuando subas o elimines recetas en tus carpetas vinculadas de Google Drive o Dropbox** y quieras que el asistente incorpore o elimine esa información de su memoria.
3. **Al configurar por primera vez tus credenciales de Drive/Dropbox** en el archivo `.env` para descargar todos los archivos por primera vez.

*¿Qué hace internamente?* Borra el índice de búsqueda antiguo, descarga los nuevos archivos de tus nubes (si están configuradas), extrae el texto de todos tus PDFs locales y de la nube, y regenera la base de datos de búsqueda para que el asistente conozca tus recetas más recientes.
