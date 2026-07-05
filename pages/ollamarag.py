import streamlit as st
import requests
import os
import logging
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser

logging.basicConfig(level=logging.INFO)


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

st.markdown("""
    <style>
    .stApp { max-width: auto; margin: 0 auto; }
    .brand-container {
        text-align: center;
        padding: 1.5rem 0;
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        border-radius: 12px;
        color: white;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .brand-container h1 { margin: 0; font-size: 2.2rem; font-weight: 700; color: white; }
    .brand-container p { margin: 5px 0 0 0; opacity: 0.90; font-size: 1rem; }
    .status-badge {
        display: inline-block;
        padding: 4px 12px;
        background-color: #ffffff33;
        color: white;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: bold;
        margin-top: 10px;
        border: 1px solid #ffffff66;
    }
    .upfront-title {
        font-weight: bold;
        font-size: 1.1rem;
        margin-bottom: 0.5rem;
    }
    </style>
""", unsafe_allow_html=True)


# ==========================================
# 2. EXACT MATCH INTERACTIVE HUD SIDEBAR ENVIRONMENT
# ==========================================
with st.sidebar:
    st.markdown("### 🗺️ PORTFOLIO NAVIGATION")
    st.page_link("app.py", label="🌐 DASHBOARD", use_container_width=True)
    st.page_link("pages/chatbot.py", label="🌐 GEMINI CLOUD CHAT", use_container_width=True)
    st.page_link("pages/chatgroq.py", label="🌐 GROQ CLOUD CHAT", use_container_width=True)
    st.page_link("pages/ollamarag.py", label="🌐 OLAMMA RAG CHAT", use_container_width=True)
    st.page_link("pages/vector1.py", label="🌐 Vector RAG CHAT", use_container_width=True)
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
     st.page_link("pages/ollamarag.py", label="🌐 OLAMMA RAG CHAT", use_container_width=True)
st.markdown("---")


# 1. Premium Page & Modern Styling Setup
st.set_page_config(
    page_title="E-Commerce AI Assistant",
    page_icon="🛍️",
    layout="centered"
)


# Main Branding Header Dashboard
# st.markdown("""
#     <div class='brand-container'>
#         <h1>🛍️ FakeStore RAG Assistant</h1>
#     </div>
# """, unsafe_allow_html=True)

# 2. IMMEDIATE ON-PAGE-LOAD DATABASE SYNCHRONIZATION
if "api_catalog_payload" not in st.session_state:
    with st.spinner("📦 Contacting server... Fetching live Web API catalog..."):
        try:
            url = "https://fakestoreapi.com/products"
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                st.session_state.api_catalog_payload = response.json()
            else:
                raise Exception(f"Server replied with code {response.status_code}")
        except Exception as e:
            logging.error(f"Failed pulling data from FakeStoreAPI: {e}")
            # Reliable fallback production data loop if server times out
            st.session_state.api_catalog_payload = [
                {"id": 1, "title": "Fjallraven Foldsack", "category": "men's clothing", "price": 109.95, "description": "Perfect backpack for everyday use.", "rating": {"rate": 3.9, "count": 120}},
                {"id": 2, "title": "Mens Casual Premium Slim Fit T-Shirts", "category": "men's clothing", "price": 22.30, "description": "Comfortable slim fit shirt.", "rating": {"rate": 4.1, "count": 259}}
            ]

# Expose data records for consumption
raw_catalog = st.session_state.api_catalog_payload

# --- DISPLAY DATA UPFRONT - CLOSED BY DEFAULT ---
st.markdown("<div class='upfront-title'>📋 Live Web API Source Database</div>", unsafe_allow_html=True)
with st.expander(f"📥 Current Live Fetched Payload ({len(raw_catalog)} items loaded)", expanded=False): # <-- UPDATED: expanded=False closes it on load
    st.json(raw_catalog)

st.divider()

# Helper logic to parse unstructured context strings
def process_docs_into_strings(raw_json):
    formatted_docs = []
    for item in raw_json:
        doc_string = (
            f"Product Title: {item.get('title')} | "
            f"Category: {item.get('category')} | "
            f"Price: ${item.get('price')} | "
            f"Description: {item.get('description')} | "
            f"Rating: {item.get('rating', {}).get('rate')}/5 based on {item.get('rating', {}).get('count')} reviews."
        )
        formatted_docs.append(doc_string)
    return formatted_docs

# 3. Initialize RAG Chain Pipeline
@st.cache_resource(show_spinner="Training Local Vector Embeddings...")
def init_rag_chain(raw_data):
    CHAT_MODEL_NAME = "llama3" 
    EMBED_MODEL_NAME = "nomic-embed-text" 
    OLLAMA_URL = "http://127.0.0.1:11434"
    
    embeddings = OllamaEmbeddings(model=EMBED_MODEL_NAME, base_url=OLLAMA_URL)
    llm = OllamaLLM(model=CHAT_MODEL_NAME, temperature=0.2, base_url=OLLAMA_URL)
    
    catalog_documents = process_docs_into_strings(raw_data)
    
    vector_store = InMemoryVectorStore.from_texts(
        texts=catalog_documents,
        embedding=embeddings
    )
    retriever = vector_store.as_retriever(search_kwargs={"k": 2})
    
    template = """You are an enthusiastic e-commerce shop assistant for FakeStore.
    Answer the buyer's question based ONLY on the following item catalog context. 
    If you don't know or if the item is missing from the context, politely say that we don't carry that item.

    Store Catalog Context:
    {context}

    Buyer Question: {question}
    Assistant Response:"""
    
    prompt = ChatPromptTemplate.from_template(template)
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    rag_chain = RunnableParallel({
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }) | {
        "context": lambda x: x["context"],
        "answer": prompt | llm | StrOutputParser()
    }
    
    return rag_chain

# Execute building routine safely
chain = None
try:
    chain = init_rag_chain(raw_catalog)
except Exception as e:
    st.error("⚠️ **Ollama Initialization Failed**")
    st.markdown("Ensure you pulled the embedding library using `ollama pull nomic-embed-text` in your command line window.")
    st.exception(e)
    st.stop()

# 4. Handle Persistent Conversation Session States
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {
            "role": "assistant", 
            "content": "Welcome to FakeStore! 🛍️ The full catalog dataset is fetched and loaded in the closed expander upfront. What would you like to buy today?",
            "context": None
        }
    ]

# Render persistent historical context on session refreshes
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if message.get("context"):
            with st.expander("🔍 Extracted API Context for this Response"):
                st.code(message["context"], language="text")

# 5. Handle Live Incoming Prompts
if chain and (user_prompt := st.chat_input("Ask about a product...")):
    st.session_state.chat_history.append({"role": "user", "content": user_prompt, "context": None})
    with st.chat_message("user"):
        st.write(user_prompt)
        
    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching matching items..."):
                output = chain.invoke(user_prompt)
            
            retrieved_context = output["context"]
            with st.expander("🔍 Extracted API Context for this Response", expanded=True):
                st.code(retrieved_context, language="text")
                
            final_answer = output["answer"]
            st.write(final_answer)
            
            st.session_state.chat_history.append({
                "role": "assistant", 
                "content": final_answer, 
                "context": retrieved_context
            })
            
        except Exception as e:
            st.error("An error occurred during response synthesis.")
            st.exception(e)
