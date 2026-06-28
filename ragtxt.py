import streamlit as st
import os
import shutil
from langchain_unstructured import UnstructuredLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 1. Application Configuration
TEMP_DIR = "./uploaded_txt_files"
DB_DIR = "./chroma_db_streamlit"

# Models optimized for their specific tasks
LLM_MODEL = "llama3"
EMBED_MODEL = "nomic-embed-text" 

# Ensure scratch directory exists
os.makedirs(TEMP_DIR, exist_ok=True)

# 2. Streamlit Page UI Setup
st.set_page_config(page_title="Llama 3 Document QA", layout="wide")
st.title("📄 Local Text Search with Llama 3 & Ollama")
st.write("Upload your text files, index them into a Vector DB, and ask questions.")

# Sidebar for file upload and processing
with st.sidebar:
    st.header("1. Upload Documents")
    uploaded_files = st.file_uploader(
        "Choose .txt files", 
        type=["txt"], 
        accept_multiple_files=True
    )
    
    process_button = st.button("🚀 Index Documents", disabled=not uploaded_files)

# 3. Core Logic: Document Parsing & Indexing
if process_button and uploaded_files:
    with st.spinner("Processing files and updating Vector DB..."):
        # Clear previous files to avoid duplication
        if os.path.exists(TEMP_DIR):
            shutil.rmtree(TEMP_DIR)
        os.makedirs(TEMP_DIR, exist_ok=True)
        
        # Save uploaded files locally
        file_paths = []
        for uploaded_file in uploaded_files:
            file_path = os.path.join(TEMP_DIR, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            file_paths.append(file_path)
        
        # Load documents via modern langchain-unstructured package
        loader = UnstructuredLoader(file_paths)
        docs = loader.load()
        
        # Split text into bite-sized chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(docs)
        
        # Initialize specialized embedding model
        embeddings = OllamaEmbeddings(model=EMBED_MODEL)
        
        # Build local Chroma Vector Database
        vector_store = Chroma.from_documents(chunks, embeddings, persist_directory=DB_DIR)
        
        st.sidebar.success(f"Indexed {len(docs)} files into {len(chunks)} chunks!")
        st.session_state["vector_db_ready"] = True

# Helper function to format context blocks for the LLM prompt
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# 4. Main Panel: Searching and Querying
st.header("2. Ask Your Documents")

if st.session_state.get("vector_db_ready", False):
    query = st.text_input("Enter your question:")
    
    if query:
        with st.spinner("Llama 3 is thinking..."):
            try:
                # Reload Vector DB using the specialized embedding model
                embeddings = OllamaEmbeddings(model=EMBED_MODEL)
                vector_store = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
                retriever = vector_store.as_retriever(search_kwargs={"k": 3})
                
                # Setup Llama 3 LLM for text generation
                llm = OllamaLLM(model=LLM_MODEL)
                
                system_prompt = (
                    "You are a helpful assistant. Use the following pieces of retrieved context to answer "
                    "the question. If you don't know the answer, say that you don't know.\n\n"
                    "Context:\n{context}\n\n"
                    "Question: {question}"
                )
                prompt = ChatPromptTemplate.from_template(system_prompt)
                
                # Core LCEL Pipeline
                rag_chain = (
                    {"context": retriever | format_docs, "question": RunnablePassthrough()}
                    | prompt
                    | llm
                    | StrOutputParser()
                )
                
                # Fetch original chunks to show source documents inside UI layout
                retrieved_docs = retriever.invoke(query)
                
                # Generate final answer from Llama 3
                answer = rag_chain.invoke(query)
                
                # Display Results
                st.markdown("### Answer")
                st.write(answer)
                
                # Display Expandable Sources Section
                with st.expander("View Source Chunks Used"):
                    for i, doc in enumerate(retrieved_docs):
                        source_name = os.path.basename(doc.metadata.get('filename', 'Unknown'))
                        st.markdown(f"**Chunk {i+1} from `{source_name}`:**")
                        st.info(doc.page_content)
            
            except Exception as e:
                st.error(f"Error executing pipeline: {e}. Make sure Ollama application server is running.")
else:
    st.info("Please upload and index some .txt files using the sidebar to start searching.")
