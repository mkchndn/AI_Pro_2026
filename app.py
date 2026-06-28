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
# hello testing
TEMP_DIR = "./uploaded_txt_files123"
DB_DIR = "./chroma_db_streamlit"
LLM_MODEL = "llama3"
EMBED_MODEL = "nomic-embed-text" 

PDF_RESUME = "MITHILESH_15Years_Exp_Dotnet_SQL_AI.pdf"
WORD_RESUME = "MITHILESH_15Years_Exp_Dotnet_SQL_AI.docx"

os.makedirs(TEMP_DIR, exist_ok=True)

# ==========================================
# 2. PREMIUM DEVELOPER STYLING ENGINE
# ==========================================
st.set_page_config(page_title="Mithilesh Upadhyay | AI Solutions Architect", page_icon="⚡", layout="wide")

st.markdown("""
    <style>
    .gradient-title {
        font-size: 3.8rem !important;
        font-weight: 100;
        background: linear-gradient(45deg, #00ffcc, #0077ff, #ff007f);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        line-height: 1.1;
    }
    .subtitle-spec {
        color: #a0aec0;
        font-size: 1.4rem;
        margin-top: 5px;
        margin-bottom: 25px;
    }
    .tech-card {
        background-color: #1a1f2c;
        border: 1px solid #2d3748;
        border-radius: 8px;
        padding: 22px;
        margin-bottom: 18px;
    }
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
    }
    div[data-testid="stSidebarNav"] {display: none;} /* Hides native streamlit redundancy navigation list */
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 3. INTERACTIVE HUD SIDEBAR NAVIGATION (SHARED)
# ==========================================
with st.sidebar:
    st.markdown("### 🗺️ PORTFOLIO NAVIGATION")
    st.page_link("app.py", label="🌐 DASHBOARD", use_container_width=True)
    st.page_link("pages/chatbot.py", label="🌐 GEMINI CLOUD CHAT", use_container_width=True)
    st.page_link("pages/chatgroq.py", label="🌐 GROQ CLOUD CHAT", use_container_width=True)
    st.page_link("pages/ollamarag.py", label="🌐 OLAMMA RAG CHAT", use_container_width=True)
        
    st.markdown("---")
    st.markdown("### 📥 DOWNLOAD CV SUITE")
    
    if os.path.exists(PDF_RESUME):
        with open(PDF_RESUME, "rb") as f:
            st.download_button("📥 Download Resume (PDF)", data=f, file_name=PDF_RESUME, mime="application/pdf", use_container_width=True)
    else:
        st.caption("❌ PDF File Missing ('Mithilesh_Upadhyay_Resume.pdf')")
        
    if os.path.exists(WORD_RESUME):
        with open(WORD_RESUME, "rb") as f:
            st.download_button("📝 Download Resume (Word)", data=f, file_name=WORD_RESUME, mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document", use_container_width=True)
    else:
        st.caption("❌ Word File Missing ('Mithilesh_Upadhyay_Resume.docx')")
    
    st.markdown("---")
    st.markdown("### 📞 DIRECT CHANNELS")
    st.markdown("📍 **Ghaziabad, UP, India**")
    st.markdown("📱 **+91 9716372870**")
    st.markdown("✉️ [mithilesh25@gmail.com](mailto:mithilesh25@gmail.com)")
    st.markdown("[🔗 LinkedIn Profile](https://linkedin.com)")

# Top Navigation Links
col_nav, col_h, col_s ,col_g,col_o = st.columns([2,2.5,2.5,2.5,2.5])
with col_nav:
    st.caption("AI PROJECTS")
with col_h:
    st.page_link("app.py", label="🌐 Dashboard", use_container_width=True)
with col_s:
    st.page_link("pages/chatbot.py", label="🌐 GEMINI CLOUD CHAT", use_container_width=True)

with col_g:
    st.page_link("pages/chatgroq.py", label="🌐 GROQ CLOUD CHAT", use_container_width=True)

    with col_o:
     st.page_link("pages/ollamarag.py", label="🌐 OLAMMA RAG CHAT", use_container_width=True)

st.markdown("---")

# ==========================================
# 4. PARENT LANDING SCREEN VIEW
# ==========================================
st.markdown("<h1 class='gradient-title'>Mithilesh Kumar Upadhyay</h1>", unsafe_allow_html=True)
st.markdown("<div class='subtitle-spec'>AI Solutions Architect & Tech Manager — IT / Software Operations</div>", unsafe_allow_html=True)

# Prominent Main-Page Resume Downloads
# st.markdown("### 📄 EXECUTIVE RECRUITMENT SUITE")
col_dl1, col_dl2, _ = st.columns([1.5, 1.5, 4])
with col_dl1:
    if os.path.exists(PDF_RESUME):
        with open(PDF_RESUME, "rb") as f:
            st.download_button("📥 Download PDF CV", data=f, file_name=PDF_RESUME, mime="application/pdf", use_container_width=True, type="primary")
with col_dl2:
    if os.path.exists(WORD_RESUME):
        with open(WORD_RESUME, "rb") as f:
            st.download_button("📝 Download Word CV", data=f, file_name=WORD_RESUME, mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document", use_container_width=True)

st.markdown("---")
st.markdown("""
I am a dynamic IT Leader and AI Solutions Architect backed by **over 15 years of robust enterprise software industry experience**, 
headlined by a stellar **12-year progressive leadership career at Samsung India Electronics** (advancing through 4 rapid promotions from 
Engineer to Manager). I capitalize on a deep foundation in high-scale corporate architectures (.NET, C#, SQL Server, PHP) to architect, deploy, 
and scale advanced **Generative AI workflows, Autonomous Multi-Agent Systems (LangGraph), and secure, privacy-first Local RAG infrastructures**.
""")

# LIVE DEPLOYMENTS LINKS HUB
st.markdown("<div class='resume-header'>🌐 LIVE ENTERPRISE CLOUD APPLICATIONS & DEPLOYMENTS</div>", unsafe_allow_html=True)
col_link1, col_link2 = st.columns(2)
with col_link1:
    st.markdown("""
    <div class="tech-card">
        <h4 style="color:#00ffcc; margin-top:0;">🌐 Scalable Google Cloud (GCP) Cluster</h4>
        <p style="font-size:0.9rem; color:#cbd5e0;">Live multi-tenant compute layer orchestrating cloud pipelines and container workloads.</p>
        <a href="pages/chatbot.py" target="_blank" style="color:#0077ff; font-weight:bold; text-decoration:none;">🚀 Launch Live GCP Web Node →</a>
    </div>
    """, unsafe_allow_html=True)
    
with col_link2:
    st.markdown("""
    <div class="tech-card">
        <h4 style="color:#ff007f; margin-top:0;">⚡ High-Throughput Groq Cloud Engine</h4>
        <p style="font-size:0.9rem; color:#cbd5e0;">Production AI architecture optimized using Groq LPU models for ultra-low latency token execution.</p>
        <a href="pages/chatgroq.py" target="_blank" style="color:#0077ff; font-weight:bold; text-decoration:none;">🚀 Access Groq Engine Gateway →</a>
    </div>
    """, unsafe_allow_html=True)

# CHRONOLOGICAL EXPERIENCE EXPANDERS
st.markdown("<div class='resume-header'>💼 COMPLETE PROFESSIONAL EXPERIENCE ARCHITECTURE</div>", unsafe_allow_html=True)
st.markdown("### **SAMSUNG INDIA ELECTRONICS PVT. LTD. | Noida, India**")
st.markdown("#### **Manager – IT / Software Operations & AI Integration** | *September 2013 – Present*")
st.markdown(
    "- **AI Modernization Strategy:** Direct the digital transformation roadmap, spearheading the integration of Generative AI models and LangChain workflows into Samsung’s legacy manufacturing software ecosystem.\n"
    "- **Strategic Team Leadership:** Lead an agile, cross-functional engineering team, ensuring 100% uptime for high-scale enterprise operations.\n"
    "- **Agentic Workflows:** Designing autonomous AI Agents to automate complex data analysis, internal reporting, and root-cause monitoring within the production floor environment."
)

st.markdown("#### **Assistant Manager** | *2018 – 2025*")
st.markdown("- **Project Oversight & SDLC:** Managed full-lifecycle software delivery for mid-to-large scale internal applications.\n- **Process Optimization:** Standardized enterprise version control (Git/TFS) across multi-tiered software architectures.")

st.markdown("#### **Senior Engineer & Engineer** | *2013 – 2018*")
st.markdown("- **Database Architecture:** Engineered high-load SQL Server database systems and backend C# sync engines.\n- **Tier-3 Technical Escalation:** Core subject matter expert resolving high-impact systemic manufacturing bugs.")

st.markdown("---")
st.markdown("### **EXCEL SOFT TECHNOLOGY PVT. LTD. | Noida, India**")
st.markdown("#### **Software Engineer** | *February 2012 – September 2013*")
st.markdown("- Architected web components using ASP.NET and C#. Built high-performing metrics dashboards using Google Charts and SSRS.")

