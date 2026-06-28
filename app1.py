import os
import shutil
import streamlit as st
from langchain_unstructured import UnstructuredLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# ==========================================
# 1. INITIALIZATION & CONFIG
# ==========================================
TEMP_DIR = "./uploaded_txt_files"
DB_DIR = "./chroma_db_streamlit"
LLM_MODEL = "llama3"
EMBED_MODEL = "nomic-embed-text" 

os.makedirs(TEMP_DIR, exist_ok=True)

# Centralized State Router
if "current_page" not in st.session_state:
    st.session_state["current_page"] = "Home"

# ==========================================
# 2. UI STYLING & NAVIGATION HELPERS
# ==========================================
st.set_page_config(page_title="Nexus AI", page_icon="⚡", layout="wide")

# Custom CSS for a professional developer/employer aesthetic
st.markdown("""
    <style>
    .gradient-title {
        font-size: 3rem !important;
        font-weight: 800;
        background: linear-gradient(45deg, #00ffcc, #0077ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .tech-card {
        background-color: #1a1f2c;
        border: 1px solid #2d3748;
        border-radius: 8px;
        padding: 20px;
        margin-bottom: 15px;
    }
    div[data-testid="stSidebarNav"] {display: none;} /* Custom override */
    </style>
""", unsafe_allow_html=True)

def render_sidebar():
    """Renders persistent AI status dashboard and page routing navigation links."""
    with st.sidebar:
        st.header("⚡ NEXUS MATRIX MENU")
        
        # Sidebar Page Navigation Links
        if st.button("🌐 System Home Dashboard", key="sb_home", use_container_width=True):
            st.session_state["current_page"] = "Home"
            st.rerun()
        if st.button("🚀 RAG Query Console Hub", key="sb_search", use_container_width=True):
            st.session_state["current_page"] = "Search"
            st.rerun()
        if st.button("📊 Tensor Analytics Metrics", key="sb_analytics", use_container_width=True):
            st.session_state["current_page"] = "Analytics"
            st.rerun()
        if st.button("🛡️ Cyber Security Isolation", key="sb_security", use_container_width=True):
            st.session_state["current_page"] = "Security"
            st.rerun()
            
        st.markdown("---")
        st.header("⚙️ HARDWARE DESCRIPTOR")
        st.text(f"LLM Core: {LLM_MODEL}")
        st.text(f"Embedding Engine: {EMBED_MODEL}")
        st.text("Store Topology: ChromaDB")
        
        st.markdown("---")
        st.header("📡 NETWORK SYNC")
        if st.session_state.get("vector_db_ready", False):
            st.success("🟢 VECTOR DATA ROUTE LIVE")
        else:
            st.warning("🟡 AWAITING DATA ARCHIVE")

def render_navigation_header():
    """Renders top horizontal header links bar."""
    col_nav1, col_home, col_search, col_analytics, col_security = st.columns([3, 1.2, 1.2, 1.2, 1.2])
    with col_nav1:
        st.caption("⚡ NEXUS // MULTI-AGENT LAYER v2.6")
    with col_home:
        if st.button("🌐 Home Core", key="hdr_home", use_container_width=True):
            st.session_state["current_page"] = "Home"
            st.rerun()
    with col_search:
        if st.button("🚀 RAG Probe", key="hdr_search", use_container_width=True):
            st.session_state["current_page"] = "Search"
            st.rerun()
    with col_analytics:
        if st.button("📊 Analytics", key="hdr_analytics", use_container_width=True):
            st.session_state["current_page"] = "Analytics"
            st.rerun()
    with col_security:
        if st.button("🛡️ Security Node", key="hdr_security", use_container_width=True):
            st.session_state["current_page"] = "Security"
            st.rerun()
    st.markdown("---")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Execute Common Navigation Framework
render_sidebar()
render_navigation_header()

# ==========================================
# 3. APP PAGE ROUTING ENGINE
# ==========================================

# --- PAGE 1: SYSTEM HOME ---
if st.session_state["current_page"] == "Home":
    st.markdown("<h1 class='gradient-title'>Nexus Neural Matrix</h1>", unsafe_allow_html=True)
    st.subheader("Autonomous RAG Pipeline & Multi-Agent Context Retrieval Engine")
    
    st.write(
        "Welcome to an enterprise-grade document intelligence platform. This node transforms "
        "raw unstructured data into structured vector embeddings, enabling isolated local "
        "semantic discovery without cloud telemetry exposure."
    )
    
    st.markdown("### ⚡ System Specifications")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="tech-card">
            <h4 style="color:#00ffcc; margin-top:0;">🤖 Cognitive Matrix</h4>
            <p>Utilizes local Llama 3Instruct parameter weights for non-linear, zero-telemetry conversational inference.</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="tech-card">
            <h4 style="color:#0077ff; margin-top:0;">📐 High-Dim Embeddings</h4>
            <p>Maps files into a 768-dimensional topological vector space via the nomic-embed-text protocol.</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="tech-card">
            <h4 style="color:#ff007f; margin-top:0;">🛡️ Persistent Cluster</h4>
            <p>Employs ChromaDB architecture for fast sub-millisecond document retrieval context queries.</p>
        </div>
        """, unsafe_allow_html=True)

    if st.button("Initialize Console Pipeline ⚡", type="primary"):
        st.session_state["current_page"] = "Search"
        st.rerun()

# --- PAGE 2: OPERATIONAL SEARCH ---
elif st.session_state["current_page"] == "Search":
    st.subheader("🚀 High-Density Data Ingestion & Search Console")
    col_upload, col_query = st.columns(2)
    
    with col_upload:
        st.markdown("### 📥 Payload Ingestion")
        uploaded_files = st.file_uploader("Drop text files:", type=["txt"], accept_multiple_files=True)
        
        if st.button("🔥 Execute Tensor Indexing", disabled=not uploaded_files, use_container_width=True):
            with st.spinner("Processing unstructured shards..."):
                if os.path.exists(TEMP_DIR):
                    shutil.rmtree(TEMP_DIR)
                os.makedirs(TEMP_DIR, exist_ok=True)
                
                file_paths = []
                for uploaded_file in uploaded_files:
                    file_path = os.path.join(TEMP_DIR, uploaded_file.name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    file_paths.append(file_path)
                
                loader = UnstructuredLoader(file_paths)
                docs = loader.load()
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                chunks = text_splitter.split_documents(docs)
                
                embeddings = OllamaEmbeddings(model=EMBED_MODEL)
                Chroma.from_documents(chunks, embeddings, persist_directory=DB_DIR)
                
                st.session_state["vector_db_ready"] = True
                st.success("Indexing successful!")
                st.rerun()
                
    with col_query:
        st.markdown("### 🔍 Semantic Matrix Probe")
        if st.session_state.get("vector_db_ready", False):
            query = st.text_input("Enter natural language request:")
            
            if query:
                with st.spinner("Executing Llama 3 lookup..."):
                    try:
                        embeddings = OllamaEmbeddings(model=EMBED_MODEL)
                        vector_store = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
                        retriever = vector_store.as_retriever(search_kwargs={"k": 3})
                        llm = OllamaLLM(model=LLM_MODEL)
                        
                        system_prompt = (
                            "You are a helpful assistant. Use the following context to answer the question.\n\n"
                            "Context:\n{context}\n\nQuestion: {question}"
                        )
                        prompt = ChatPromptTemplate.from_template(system_prompt)
                        
                        rag_chain = (
                            {"context": retriever | format_docs, "question": RunnablePassthrough()}

                            | prompt | llm | StrOutputParser()
                        )
                        
                        retrieved_docs = retriever.invoke(query)
                        answer = rag_chain.invoke(query)
                        
                        st.markdown("#### 💬 Output Synthesis")
                        st.info(answer)
                        
                        st.markdown("#### 🎯 Linear Node Track")
                        for i, doc in enumerate(retrieved_docs):
                            fname = os.path.basename(doc.metadata.get('filename', 'Unknown'))
                            with st.expander(f"Chunk {i+1} From: {fname}"):
                                st.caption(doc.page_content)
                                
                    except Exception as e:
                        st.error(f"Inference error: {e}")
        else:
            st.warning("⚠️ Engine Offline. Please ingest training documents first to activate search routing.")

