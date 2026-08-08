import os
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# 1. CRITICAL: must be the absolute first execution command on child pages
st.set_page_config(page_title="Gemini Chatbot", page_icon="💬", layout="centered")

# Hide default navigation and match structural styles symmetrically
st.markdown("""
    <style>
    div[data-testid="stSidebarNav"] {display: none;}
    </style>
""", unsafe_allow_html=True)

PDF_RESUME = "MITHILESH_15Years_Exp_Dotnet_SQL_AI.pdf"
WORD_RESUME = "MITHILESH_15Years_Exp_Dotnet_SQL_AI.docx"

# ==========================================
# FILE PARSING HELPERS
# ==========================================
def extract_text_from_pdf(file_path):
    """Safely extracts text string from a PDF file."""
    from pypdf import PdfReader
    text = ""
    try:
        reader = PdfReader(file_path)
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
    except Exception as e:
        st.error(f"Error parsing PDF: {e}")
    return text

def extract_text_from_docx(file_path):
    """Safely extracts text string from a DOCX file."""
    import docx2txt
    try:
        return docx2txt.process(file_path)
    except Exception as e:
        st.error(f"Error parsing DOCX: {e}")
        return ""

# ==========================================
# FIXED: DYNAMIC RESUME VECTOR STORE INITIALIZATION
# ==========================================
if "vector_db" not in st.session_state:
    with st.spinner("Reading resume files and building semantic database..."):
        combined_resume_text = ""
        
        # Look for the physical PDF file first
        if os.path.exists(PDF_RESUME):
            combined_resume_text += extract_text_from_pdf(PDF_RESUME)
        
        # Look for the physical Word document file next
        if os.path.exists(WORD_RESUME):
            combined_resume_text += "\n" + extract_text_from_docx(WORD_RESUME)
            
        # Fallback if both physical files are missing from the folder
        # if not combined_resume_text.strip():
        #     combined_resume_text = (
        #         "Fallback Context: Mithilesh is an IT professional with 15 years of experience "
        #         "specializing in .NET architectures, SQL Server, and AI solutions."
        #     )
        #     st.warning("⚠️ Resume files not found locally or empty. Using basic fallback profile context.")

        # Chunk the raw text so it fits comfortably within the vector model limits
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
        chunks = text_splitter.split_text(combined_resume_text)
        
        # Wrap chunked strings back into standard LangChain Document wrappers
        docs = [Document(page_content=chunk) for chunk in chunks]
        
        # Generate semantic embeddings locally via Hugging Face
        model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        st.session_state.vector_db = FAISS.from_documents(docs, model)
    st.toast("⚡ Resume vector data indexed successfully!")

# ==========================================
# 2. EXACT MATCH INTERACTIVE HUD SIDEBAR ENVIRONMENT
# ==========================================
with st.sidebar:
    st.markdown("### 🗺️ PORTFOLIO NAVIGATION")
    st.page_link("app.py", label="🌐 DASHBOARD", use_container_width=True)
    st.page_link("pages/chatbot.py", label="🌐 GEMINI CLOUD CHAT", use_container_width=True)
    st.page_link("pages/chatgroq.py", label="🌐 GROQ CLOUD CHAT", use_container_width=True)
    st.page_link("pages/agent.py", label="🌐 CUSTOM AGENT", use_container_width=True)
    st.page_link("pages/vector_resume.py", label="🌐 Vector RESUME CHAT", use_container_width=True)
        
    st.markdown("---")
    st.markdown("### 📥 DOWNLOAD CV SUITE")
    
    if os.path.exists(PDF_RESUME):
        with open(PDF_RESUME, "rb") as f:
            st.download_button("📥 Download Resume (PDF)", data=f, file_name=PDF_RESUME, mime="application/pdf", use_container_width=True, key="child_pdf")
    else:
        st.caption("❌ PDF File Missing ('Mithilesh_Upadhyay_Resume.pdf')")
        
    if os.path.exists(WORD_RESUME):
        with open(WORD_RESUME, "rb") as f:
            st.download_button("📝 Download Resume (Word)", data=f, file_name=WORD_RESUME, mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document", use_container_width=True, key="child_word")
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
    st.page_link("pages/agent.py", label="🌐 CUSTOM AGENT", use_container_width=True)

st.markdown("---")

# ==========================================
# 3. INTERACTIVE CHAT ENGINE ENVIRONMENT
# ==========================================
with st.expander("🔑 SECURE KEY MANAGER", expanded=False):
    google_api_key = st.text_input(
        "Enter Google API Key:", 
        type="password", 
        placeholder="AIzaSy...",
        value=os.environ.get("GOOGLE_API_KEY", "")
    )

col_config1, col_config2 = st.columns(2)
with col_config1:
    selected_model = st.selectbox(
        "🧠 Active Gemini Architecture:",
        ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-1.5-flash", "gemini-1.5-pro"],
        index=0
    )
with col_config2:
    selected_temp = st.slider(
        "🎛️ Creativity Settings (Temperature):",
        min_value=0.0, max_value=1.0, value=0.7, step=0.1
    )

st.markdown("---")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if user_input := st.chat_input("Type your message here..."):
    if not google_api_key:
        st.error("Operation Denied: Secure Key token array is missing.")
    else:
        os.environ["GOOGLE_API_KEY"] = google_api_key
        
        with st.chat_message("user"):
            st.markdown(user_input)
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        with st.chat_message("assistant"):
            with st.spinner("Querying resume context indexes..."):
                try:
                    # Initialize LLM Architecture
                    llm = ChatGoogleGenerativeAI(
                        model=selected_model, 
                        temperature=selected_temp,
                        api_key=google_api_key
                    )
                    
                    # FIXED RETRIEVAL LOGIC: Grab the list cleanly
                    search_results = st.session_state.vector_db.similarity_search(user_input, k=2)
                    
                    # Parse each document block safely using precise list extraction loops
                    context_chunks = []
                    for doc in search_results:
                        context_chunks.append(doc.page_content)
                    
                    # Flatten into single systemic context string boundary
                    context_chunk = "\n\n".join(context_chunks) if context_chunks else ""
                    
                    # DEBUG VISUALIZER: Confirms that data is actively being extracted from the file
                    if context_chunk:
                        with st.expander("🔍 RAG Context Retrieved From Document", expanded=False):
                            st.info(context_chunk)
                    
                    # System Prompt framing retrieved information as a ground truth boundary
                    system_prompt = (
                        "You are an intelligent, professional portfolio assistant for Mithilesh. "
                        "Answer questions accurately using ONLY the extracted resume text context provided below. "
                        "If the answer cannot be found in the context, politely explain that it is not documented in the resume.\n\n"
                        f"Extracted Resume Context:\n{context_chunk}"
                    )
                    
                    # Package chat strings into clean historical list
                    langchain_messages = [SystemMessage(content=system_prompt)]
                    
                    for msg in st.session_state.chat_history:
                        if msg["role"] == "user":
                            langchain_messages.append(HumanMessage(content=msg["content"]))
                        else:
                            langchain_messages.append(AIMessage(content=msg["content"]))
                    
                    # Prompt the language model
                    res = llm.invoke(langchain_messages)
                    st.markdown(res.content)
                    st.session_state.chat_history.append({"role": "assistant", "content": res.content})
                except Exception as e:
                    st.error(f"Inference error: {e}")
