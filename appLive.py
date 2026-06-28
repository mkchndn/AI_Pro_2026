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
# 1. INITIALIZATION & CORE CONFIG
# ==========================================
TEMP_DIR = "./uploaded_txt_files"
DB_DIR = "./chroma_db_streamlit"
LLM_MODEL = "llama3"
EMBED_MODEL = "nomic-embed-text" 
RESUME_FILE = "MITHILESH_15Years_Exp_Dotnet_SQL_AI.pdf" # Place your PDF resume in the same directory

os.makedirs(TEMP_DIR, exist_ok=True)

if "current_page" not in st.session_state:
    st.session_state["current_page"] = "Home"

# ==========================================
# 2. PREMIUM DEVELOPER STYLING ENGINE
# ==========================================
st.set_page_config(page_title="Mithilesh Upadhyay | AI Solutions Architect", page_icon="⚡", layout="wide")

st.markdown("""
    <style>
    /* Neon Gradient Title Branding */
    .gradient-title {
        font-size: 3.8rem !important;
        font-weight: 800;
        background: linear-gradient(45deg, #00ffcc, #0077ff, #ff007f);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
        line-height: 1.1;
    }
    .subtitle-spec {
        color: #a0aec0;
        font-size: 1.4rem;
        font-weight: 400;
        margin-top: 5px;
        margin-bottom: 25px;
    }
    /* Technical Project & Resume Cards */
    .tech-card {
        background-color: #1a1f2c;
        border: 1px solid #2d3748;
        border-radius: 8px;
        padding: 22px;
        margin-bottom: 18px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: border-color 0.2s;
    }
    .tech-card:hover {
        border-color: #00ffcc;
    }
    /* Resume Visual Section Anchors */
    .resume-header {
        font-size: 1.4rem;
        color: #00ffcc;
        background: rgba(0, 255, 204, 0.05);
        padding: 8px 15px;
        border-left: 4px solid #00ffcc;
        margin-top: 30px;
        margin-bottom: 15px;
        font-family: 'Courier New', monospace;
        font-weight: bold;
    }
    .badge {
        background-color: #2d3748;
        color: #00ffcc;
        padding: 4px 10px;
        border-radius: 4px;
        font-size: 0.85rem;
        font-family: monospace;
        margin-right: 5px;
        display: inline-block;
        margin-bottom: 5px;
    }
    div[data-testid="stSidebarNav"] {display: none;}
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 3. INTERACTIVE HUD SIDEBAR ENVIRONMENT
# ==========================================
def render_sidebar():
    with st.sidebar:

        # st.markdown("---")
        st.markdown("### 📞 DIRECT CHANNELS")
        st.markdown("📍 **Ghaziabad, UP, India**")
        st.markdown("📱 **+91 9716372870**")
        st.markdown("✉️ [mithilesh25@gmail.com](mailto:mithilesh25@gmail.com)")
        st.markdown("[🔗 LinkedIn Profile](https://linkedin.com)")
        
        # Native PDF Resume File Downloader
        # st.markdown("---")
        if os.path.exists(RESUME_FILE):
            with open(RESUME_FILE, "rb") as file:
                st.download_button(
                    label="📥 Download Resume PDF",
                    data=file,
                    file_name="MITHILESH_15Years_Exp_Dotnet_SQL_AI.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
        else:
            st.caption("ℹ️ Add 'MITHILESH_15Years_Exp_Dotnet_SQL_AI.pdf' to your folder to unlock download button.")
          
        st.markdown("### 🗺️ PORTFOLIO NAVIGATION")
        if st.button("🌐 Executive Portfolio & CV", key="sb_home", use_container_width=True):
            st.session_state["current_page"] = "Home"
            st.rerun()
        if st.button("🚀 Live RAG Engine Sandbox", key="sb_search", use_container_width=True):
            st.session_state["current_page"] = "Search"
            st.rerun()
        if st.button("📊 Architecture Analytics", key="sb_analytics", use_container_width=True):
            st.session_state["current_page"] = "Analytics"
            st.rerun()
            
         
        st.markdown("---")
        st.markdown("### ⚙️ LOCAL RUNTIME HUD")
        st.caption(f"Orchestration: LangChain / LCEL")
        st.caption(f"Reasoning Core: {LLM_MODEL}")
        st.caption(f"Tensor Matrix: {EMBED_MODEL}")
        
        st.markdown("---")
        if st.session_state.get("vector_db_ready", False):
            st.success("🟢 SANDBOX DEPLOYMENT LIVE")
        else:
            st.warning("🟡 AWAITING PAYLOAD INDEX")

def render_navigation_header():
    col_nav, col_h, col_s, col_a = st.columns([4.5, 1.5, 1.5, 1.5])
    with col_nav:
        st.caption("MITHILESH // AI PORTFOLIO INTERFACE")
    with col_h:
        if st.button("🌐 Portfolio Matrix", key="hdr_home", use_container_width=True):
            st.session_state["current_page"] = "Home"
            st.rerun()
    with col_s:
        if st.button("🚀 Live RAG Sandbox", key="hdr_search", use_container_width=True):
            st.session_state["current_page"] = "Search"
            st.rerun()
    with col_a:
        if st.button("📊 Engine Stats", key="hdr_analytics", use_container_width=True):
            st.session_state["current_page"] = "Analytics"
            st.rerun()
    st.markdown("---")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

render_sidebar()
render_navigation_header()

# ==========================================
# 4. APP EXECUTIVE PAGES ROUTING
# ==========================================

# --- PAGE 1: EXECUTIVE PORTFOLIO & CV ---
if st.session_state["current_page"] == "Home":
    
    # Hero Title Segment
    st.markdown("<h1 class='gradient-title'>Mithilesh Kumar Upadhyay</h1>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle-spec'>AI Powered Solutions Expert & Tech Manager — IT / Software Operations</div>", unsafe_allow_html=True)
    
    st.markdown("""
    I am a dynamic IT Leader and AI Solutions Architect backed by **over 15 years of robust enterprise software industry experience**, 
    headlined by a stellar **12-year progressive leadership career at Samsung India Electronics** (advancing through 4 rapid promotions from 
    Engineer to Manager). I capitalize on a deep foundation in high-scale corporate architectures (.NET, C#, SQL Server) to architect, deploy, 
    and scale advanced **Generative AI workflows, Autonomous Multi-Agent Systems (LangGraph), and secure, privacy-first Local RAG infrastructures**.
    """)
    
    # Interactive Call To Action to test live app
    col_cta1, col_cta2 = st.columns(2)
    with col_cta1:
        st.info("💡 **Employer Spotlight:** You can verify my local AI implementation directly inside this portfolio! Switch tabs or click the button to launch my live search engine.")
    with col_cta2:
        if st.button("Launch Live RAG Sandbox Module ⚡", type="primary", use_container_width=True):
            st.session_state["current_page"] = "Search"
            st.rerun()

    # SECTION: AI PRODUCTION SHIPPED HIGHLIGHTS
    st.markdown("<div class='resume-header'>🚀 SHIPPED ENTERPRISE AI PRODUCTION PROJECT HIGHLIGHTS</div>", unsafe_allow_html=True)
    
    col_p1, col_p2, col_p3 = st.columns(3)
    with col_p1:
        st.markdown("""
        <div class="tech-card">
            <h4 style="color:#00ffcc; margin-top:0;">🤖 1. Enterprise Multi-Agent Analyst</h4>
            <p style="font-size:0.9rem; color:#cbd5e0;">Orchestrated via <b>LangGraph and Groq Cloud</b> to fully automate intricate backend reporting.</p>
            <p style="font-size:0.85rem; color:#a0aec0;">Converts cross-departmental natural language inputs into optimized SQL scripts, interpreting live datasets with instant analytics outputs.</p>
            <span class="badge">LangGraph</span><span class="badge">Groq Cloud</span><span class="badge">SQL API</span>
        </div>
        """, unsafe_allow_html=True)
        
    with col_p2:
        st.markdown("""
        <div class="tech-card">
            <h4 style="color:#0077ff; margin-top:0;">👁️ 2. Multimodal Factory Inspection Agent</h4>
            <p style="font-size:0.9rem; color:#cbd5e0;">Deployed completely on **GCP** utilizing **Google Gemini Multimodal APIs** for automated QA.</p>
            <p style="font-size:0.85rem; color:#a0aec0;">Processes live manufacturing floor imagery against massive hardware component databases to flags anomalies and dispatch restock webhooks.</p>
            <span class="badge">Gemini API</span><span class="badge">GCP Node</span><span class="badge">Computer Vision</span>
        </div>
        """, unsafe_allow_html=True)
        
    with col_p3:
        st.markdown("""
        <div class="tech-card">
            <h4 style="color:#ff007f; margin-top:0;">🔒 3. Sovereign Privacy-First RAG</h4>
            <p style="font-size:0.9rem; color:#cbd5e0;">Engineered using **LangChain, Ollama, and Llama 3** (the architecture driving this app).</p>
            <p style="font-size:0.85rem; color:#a0aec0;">Parses unstructured dark files (PDFs, manuals) locally into vectors, guaranteeing absolute corporate data privacy with zero public data leaks.</p>
            <span class="badge">LangChain</span><span class="badge">Llama 3</span><span class="badge">ChromaDB</span>
        </div>
        """, unsafe_allow_html=True)

    # SECTION: CHRONOLOGICAL EXPERIENCE MATRIX
    st.markdown("<div class='resume-header'>💼 PROFESSIONAL EXPERIENCE ARCHITECTURE</div>", unsafe_allow_html=True)
    
