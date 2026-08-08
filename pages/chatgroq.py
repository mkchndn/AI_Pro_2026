import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage

# 1. CRITICAL: must be the absolute first execution command on child pages
st.set_page_config(page_title="Groq AI Chatbot", page_icon="⚡", layout="centered")

# Hide default navigation and match structural styles symmetrically
st.markdown("""
    <style>
    div[data-testid="stSidebarNav"] {display: none;}
    </style>
""", unsafe_allow_html=True)

# Updated exact resume file targets from your configurations
PDF_RESUME = "MITHILESH_15Years_Exp_Dotnet_SQL_AI.pdf"
WORD_RESUME = "MITHILESH_15Years_Exp_Dotnet_SQL_AI.docx"

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
        st.caption(f"❌ PDF File Missing ('{PDF_RESUME}')")
        
    if os.path.exists(WORD_RESUME):
        with open(WORD_RESUME, "rb") as f:
            st.download_button("📝 Download Resume (Word)", data=f, file_name=WORD_RESUME, mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document", use_container_width=True, key="child_word")
    else:
        st.caption(f"❌ Word File Missing ('{WORD_RESUME}')")
    
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
# 3. INTERACTIVE CHAT ENGINE ENVIRONMENT (GROQ CLOUD)
# ==========================================
# st.title("⚡ GROQ CLOUD CHAT")
# st.write("Start chatting! Powered by Groq's high-speed LPU Inference Engine.")
# st.markdown("---")

with st.expander("🔑 SECURE KEY MANAGER", expanded=False):
    groq_api_key = st.text_input(
        "Enter Groq API Key:", 
        type="password", 
        placeholder="gsk_...",
        value=os.environ.get("GROQ_API_KEY", "")
    )

col_config1, col_config2 = st.columns(2)
with col_config1:
    # Top production open-weights models available on Groq Cloud
    selected_model = st.selectbox(
        "🧠 Active Groq Model Architecture:",
            [
            "llama-3.3-70b-versatile", 
            "mixtral-8x7b-32768", 
            "gemma2-9b-it",
            "groq/compound",
            "groq/compound-mini",
            "llama-3.1-8b-instant",
            "openai/gpt-oss-120b",
            "openai/gpt-oss-20b",
            "openai/gpt-oss-safeguard-20b",
            "qwen/qwen3-32b",
            "qwen/qwen3.6-27b"
        ],
        index=0
    )
with col_config2:
    selected_temp = st.slider(
        "🎛️ Creativity Settings (Temperature):",
        min_value=0.0, max_value=1.0, value=0.2, step=0.1,
        help="Lower values are highly focused/factual, higher values generate unique text variances."
    )

st.markdown("---")

if "chat_historyGroq" not in st.session_state:
    st.session_state.chat_historyGroq = []

# Display entire dialogue trace
for msg in st.session_state.chat_historyGroq:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if user_input := st.chat_input("Type your message here..."):
    if not groq_api_key:
        st.error("Operation Denied: Secure Groq API Key token array is missing.")
    else:
        with st.chat_message("user"):
            st.markdown(user_input)
        st.session_state.chat_historyGroq.append({"role": "user", "content": user_input})

        with st.chat_message("assistant"):
            with st.spinner("Processing token stream via Groq LPUs..."):
                try:
                    # Initialize modern standalone ChatGroq class instance
                    llm = ChatGroq(
                        model=selected_model, 
                        temperature=selected_temp,
                        groq_api_key=groq_api_key
                    )
                    
                    langchain_messages = []
                    for msg in st.session_state.chat_historyGroq:
                        if msg["role"] == "user":
                            langchain_messages.append(HumanMessage(content=msg["content"]))
                        else:
                            langchain_messages.append(AIMessage(content=msg["content"]))
                    
                    res = llm.invoke(langchain_messages)
                    st.markdown(res.content)
                    
                    st.session_state.chat_historyGroq.append({"role": "assistant", "content": res.content})
                except Exception as e:
                    st.error(f"Groq Inference Engine error: {e}")
