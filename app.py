import os
import fitz  # PyMuPDF
import streamlit as st
import tempfile
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import tiktoken
import requests
from deep_translator import GoogleTranslator
from gtts import gTTS
import time

st.set_page_config(
    page_title="RAG Document Assistant",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

def sidebar_profiles():
    st.sidebar.markdown("""<hr>""", unsafe_allow_html=True)
    st.sidebar.markdown("### 🎉Author: Maria Nadeem🌟")
    st.sidebar.markdown("### 🔗 Connect With Me")
    st.sidebar.markdown("""
    <hr>
    <div class="profile-links">
        <a href="https://github.com/marianadeem755" target="_blank">
            <img src="https://cdn-icons-png.flaticon.com/512/25/25231.png" width="20px"> GitHub
        </a><br><br>
        <a href="https://www.kaggle.com/marianadeem755" target="_blank">
            <img src="https://cdn4.iconfinder.com/data/icons/logos-and-brands/512/189_Kaggle_logo_logos-512.png" width="20px"> Kaggle
        </a><br><br>
        <a href="mailto:marianadeem755@gmail.com">
            <img src="https://cdn-icons-png.flaticon.com/512/561/561127.png" width="20px"> Email
        </a><br><br>
        <a href="https://huggingface.co/maria355" target="_blank">
            <img src="https://huggingface.co/front/assets/huggingface_logo-noborder.svg" width="20px"> Hugging Face
        </a>
    </div>
    <hr>
    """, unsafe_allow_html=True)
# Add the profile section
sidebar_profiles()
def get_api_key():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        st.error("GROQ_API_KEY environment variable is not set. Please set it before running the application.")
    return api_key

# Session state initialization
for key, default in {
    "chunks": [],
    "chunk_sources": [],
    "debug_mode": False,
    "last_query_time": None,
    "last_response": None
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedder = load_embedder()
embedding_dim = 384
index = faiss.IndexFlatL2(embedding_dim)
tokenizer = tiktoken.get_encoding("cl100k_base")

def num_tokens_from_string(string: str) -> int:
    return len(tokenizer.encode(string))

def chunk_text(text, max_tokens=250):
    sentences = text.split(". ")
    current_chunk = []
    total_tokens = 0
    result_chunks = []
    for sentence in sentences:
        if not sentence.strip():
            continue
        token_len = num_tokens_from_string(sentence)
        if total_tokens + token_len > max_tokens:
            if current_chunk:
                result_chunks.append(". ".join(current_chunk) + ("." if not current_chunk[-1].endswith(".") else ""))
            current_chunk = [sentence]
            total_tokens = token_len
        else:
            current_chunk.append(sentence)
            total_tokens += token_len
    if current_chunk:
        result_chunks.append(". ".join(current_chunk) + ("." if not current_chunk[-1].endswith(".") else ""))
    return result_chunks

def extract_text_from_pdf(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def index_uploaded_text(text):
    global index
    index = faiss.IndexFlatL2(embedding_dim)
    st.session_state.chunks = []
    st.session_state.chunk_sources = []

    chunks_list = chunk_text(text)
    st.session_state.chunks = chunks_list

    for i, chunk in enumerate(chunks_list):
        st.session_state.chunk_sources.append(f"Chunk {i+1}: {chunk[:50]}...")
        vector = embedder.encode([chunk])[0]
        index.add(np.array([vector]).astype('float32'))

    return len(chunks_list)

def retrieve_chunks(query, top_k=5):
    if index.ntotal == 0:
        return []
    q_vector = embedder.encode([query])
    D, I = index.search(np.array(q_vector).astype('float32'), k=min(top_k, index.ntotal))
    return [st.session_state.chunks[i] for i in I[0] if i < len(st.session_state.chunks)]

def build_prompt(system_prompt, context_chunks, question):
    context = "\n\n".join(context_chunks)
    return f"""{system_prompt}
Context:
{context}
Question:
{question}
Answer: Please provide a comprehensive answer based only on the context provided."""

def generate_answer(prompt):
    api_key = get_api_key()
    if not api_key:
        return "API key is missing. Please set the GROQ_API_KEY environment variable or enter it in the sidebar."
    headers = {
        "Authorization": f"Bearer {api_key.strip()}",
        "Content-Type": "application/json"
    }
    selected_model = st.session_state.get("MODEL_CHOICE", "llama3-8b-8192")
    payload = {
        "model": selected_model,
        "messages": [
            {"role": "system", "content": "You are a helpful document assistant that answers questions only using the provided context."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.3,
        "max_tokens": 1024
    }
    try:
        start_time = time.time()
        with st.spinner("Sending request to Groq API..."):
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                json=payload,
                headers=headers,
                timeout=30
            )
        query_time = time.time() - start_time
        st.session_state.last_query_time = f"{query_time:.2f} seconds"

        if response.status_code == 401:
            return "Authentication failed: Invalid or expired API key."
        if response.status_code == 400:
            error_info = response.json().get("error", {})
            error_message = error_info.get("message", "Unknown error")
            if "model not found" in error_message.lower():
                st.warning("Trying with alternate model...")
                payload["model"] = "llama3-8b-8192"
                response = requests.post("https://api.groq.com/openai/v1/chat/completions", json=payload, headers=headers)
                if response.status_code != 200:
                    return f"Both model attempts failed. Error: {error_message}"
            else:
                return f"API Error: {error_message}"
        response.raise_for_status()
        response_json = response.json()
        if "choices" not in response_json or not response_json["choices"]:
            return "No answer was generated."
        answer = response_json["choices"][0]["message"]["content"]
        st.session_state.last_response = answer
        return answer
    except requests.exceptions.RequestException as e:
        return f"API request failed: {str(e)}"
    except Exception as e:
        return f"Unexpected error: {str(e)}"

def translate_text(text, target_language):
    try:
        with st.spinner(f"Translating to {target_language}..."):
            return GoogleTranslator(source='auto', target=target_language).translate(text)
    except Exception as e:
        st.error(f"Translation failed: {str(e)}")
        return text

def text_to_speech(text, lang_code):
    try:
        with st.spinner("Generating audio..."):
            tts = gTTS(text=text, lang=lang_code)
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
            tts.save(temp_file.name)
            return temp_file.name
    except Exception as e:
        st.error(f"Text-to-speech failed: {str(e)}")
        return None
# Streamlit UI
st.title("📄 RAG Explorer:  AI-Powered Document Assistant")
st.markdown("Upload a document and ask questions to get AI-powered answers with translation capabilities.")

# Add API key input in sidebar
with st.sidebar:
    st.header("API Configuration")
    # Remove the API key input field
    st.info("Using Groq API key from environment variables")
    
    # Add model selection
    st.subheader("Model Selection")
    model_choice = st.selectbox(
        "Select LLM Model",
        [
            "llama3-8b-8192",  # Changed default to a model known to work
            "llama3-70b-8192"
        ],
        help="Choose the Groq model to use for answering questions"
    )
    
    st.session_state["MODEL_CHOICE"] = model_choice
    
    # Debug mode toggle
    st.subheader("Debug Settings")
    st.session_state.debug_mode = st.checkbox("Show Debug Information", value=st.session_state.debug_mode)
    
    if st.session_state.last_query_time:
        st.subheader("Performance")
        # Main UI for file upload
        uploaded_file = st.file_uploader("Upload a PDF document", type="pdf")
        
        if uploaded_file:
            text = extract_text_from_pdf(uploaded_file)
            chunk_count = index_uploaded_text(text)
            st.success(f"Indexed {chunk_count} text chunks from the uploaded PDF.")
        
            # User query input
            question = st.text_input("Ask a question based on the document")
        
            if question:
                context_chunks = retrieve_chunks(question)
                system_prompt = "You are a helpful assistant answering only from the document context provided."
                prompt = build_prompt(system_prompt, context_chunks, question)
                answer = generate_answer(prompt)
                
                # Show response
                st.markdown("### 💬 AI Answer")
                st.write(answer)
                
                # Optional: Translate
                translate_option = st.selectbox("Translate the answer to", ["None", "Urdu", "Hindi", "Spanish", "French"])
                language_codes = {
                    "Urdu": "ur",
                    "Hindi": "hi",
                    "Spanish": "es",
                    "French": "fr"
                }
        
                if translate_option != "None":
                    translated_text = GoogleTranslator(source='auto', target='fr').translate(text)
                    st.markdown(f"### 🌐 Translated Answer ({translate_option})")
                    st.write(translated_text)
        
                    # Audio playback
                    audio_file_path = text_to_speech(translated_text, language_codes.get(translate_option, "en"))
                    if audio_file_path:
                        st.audio(audio_file_path, format="audio/mp3")

            elif uploaded_file is None:
                st.info("Please upload a PDF document to begin.")
            
                
                st.subheader("About")
                st.markdown("""
                This app uses Retrieval-Augmented Generation (RAG) to answer questions about uploaded documents.
                1. Upload a document
                2. Ask a question
                3. Optionally translate responses to other languages
                """)

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    uploaded_file = st.file_uploader("Upload a PDF or TXT file", type=["pdf", "txt"])
    if uploaded_file:
        with st.spinner("Reading and indexing document..."):
            raw_text = ""
            if uploaded_file.type == "application/pdf":
                raw_text = extract_text_from_pdf(uploaded_file)
            elif uploaded_file.type == "text/plain":
                raw_text = uploaded_file.read().decode("utf-8")
                
            total_chunks = index_uploaded_text(raw_text)
            st.success(f"Document indexed successfully! Created {total_chunks} chunks.")
            
            # Display document preview
            with st.expander("Document Preview"):          
                # Extract and display key points
                st.subheader("Key Points")
                
                # Simple algorithm to extract potential key points (sentences that might be important)
                sentences = raw_text.split('. ')
                key_points = []
                
                # Look for sentences that might be key points (contains keywords, not too long/short)
                for sentence in sentences[:50]:  # Check first 50 sentences
                    sentence = sentence.strip()
                    if len(sentence) > 15 and len(sentence) < 200:  # Reasonable length for a key point
                        # Keywords that might indicate important information
                        important_keywords = ["important", "key", "significant", "main", "primary", "essential", 
                                             "critical", "crucial", "fundamental", "major", "summary", "conclusion"]
                        
                        if any(keyword in sentence.lower() for keyword in important_keywords) or sentence.endswith(':'):
                            key_points.append(sentence)
                
                # If we didn't find obvious key points, just take some representative sentences
                if len(key_points) < 3:
                    key_points = [s.strip() for s in sentences[:50:10] if len(s.strip()) > 15][:5]  # Every 10th sentence from first 50
                
                # Display the key points as bullets
                for point in key_points[:5]:  # Show up to 5 key points
                    st.markdown(f"• {point}")
                
                if not key_points:
                    st.info("No clear key points detected. Try exploring the full document.")

with col2:
    if st.session_state.chunks:
        st.info(f"Document chunks: {len(st.session_state.chunks)}")

# Query and answer section
# Query and answer section
st.divider()
query = st.text_input("Ask a question about the document")

col1, col2 = st.columns([1, 1])

with col1:
    enable_translation = st.checkbox("Translate answer", value=False)
    use_local = st.checkbox("Use local processing (no API call)", value=False,
                          help="Use this if you're having API issues")

with col2:
    language = st.selectbox("Language", ["English", "Urdu", "Hindi", "French", "Chinese", "Spanish", "German", "Arabic", "Russian"])
    language_codes = {
        "English": "en", "Urdu": "ur", "Hindi": "hi", "French": "fr", "Chinese": "zh-CN",
        "Spanish": "es", "German": "de", "Arabic": "ar", "Russian": "ru"
    }
    lang_code = language_codes[language]

# Add a submit button
submit_button = st.button("Get Answer", type="primary", key="submit_query")

# Only process when the button is clicked and there's a query
if submit_button and query:
    if index.ntotal == 0:
        st.warning("Please upload and index a document first.")
    else:
        with st.spinner("Generating answer..."):
            top_chunks = retrieve_chunks(query)
            if not top_chunks:
                st.error("No relevant content found.")
            else:
                system_prompt = "You are a document assistant. Use only the context to answer accurately."
                prompt = build_prompt(system_prompt, top_chunks, query)
                
                # Check API key before making call
                if not get_api_key() and not use_local:
                    st.error("API key is not set. Please add it in the sidebar.")
                else:
                    if use_local:
                        # Simple local processing that summarizes the chunks without API call
                        st.warning("Using local processing - limited functionality!")
                        answer = f"Local processing summary (no LLM used):\n\n"
                        answer += f"Question: {query}\n\n"
                        answer += "Here are the most relevant passages found:\n\n"
                        for i, chunk in enumerate(top_chunks[:3], 1):
                            answer += f"{i}. {chunk[:200]}...\n\n"
                    else:
                        answer = generate_answer(prompt)
                    
                    # Display query and context if debug mode is on
                    if st.session_state.debug_mode:
                        with st.expander("Query Context", expanded=False):
                            st.write("Query:", query)
                            st.write("Top chunks used:")
                            for i, chunk in enumerate(top_chunks, 1):
                                st.write(f"{i}. {chunk[:100]}...")
                    
                    # Create tabs for original and translated answers
                    tab1, tab2 = st.tabs(["Original Answer", f"Translated ({language})" if enable_translation else "Translation (disabled)"])
                    
                    with tab1:
                        st.markdown("### Answer:")
                        st.write(answer)
                    
                    with tab2:
                        if enable_translation and answer:
                            translated = translate_text(answer, lang_code)
                            st.markdown(f"### Answer ({language}):")
                            st.write(translated)
                            
                            # Audio generation
                            audio_path = text_to_speech(translated, lang_code)
                            if audio_path:
                                st.audio(audio_path, format="audio/mp3")
                        else:
                            st.info("Enable translation to see the answer in your selected language.")

# Add footer
st.divider()
st.caption("RAG Document Assistant - Powered by Groq & Sentence Transformers")
