import streamlit as st
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Set up the Streamlit page layout
st.set_page_config(page_title="Simple Text Search Agent", page_icon="🤖")
st.title("🤖 Local Text Search Agent")
st.write("Ask a question, and the agent will retrieve the most contextually relevant answer from its database.")

# 1. Initialize the Vector DB once and cache it in the session state
if "vector_db" not in st.session_state:
    with st.spinner("Initializing knowledge base... Please wait..."):
        # Simple text dataset
        texts = [
            "Dogs are loyal, friendly, and make excellent domestic pets.",
            "Python is a versatile programming language widely used for data science and AI development.",
            "Baking a cake requires essential ingredients like flour, sugar, and baking powder.",
            "The Eiffel Tower is a famous historic monument located in Paris, France."
        ]
        docs = [Document(page_content=t) for t in texts]
        
        # Load embedding model and build FAISS index
        model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        st.session_state.vector_db = FAISS.from_documents(docs, model)
    st.success("Knowledge base ready!")

# 2. Initialize chat message history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I am an agent connected to a simple vector database. Ask me anything!"}
    ]

# 3. Display existing conversation history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# 4. Handle new user input
if user_query := st.chat_input("Type your question here (e.g., 'Tell me about coding')"):
    
    # Display the user's message in the chat
    with st.chat_message("user"):
        st.write(user_query)
    st.session_state.messages.append({"role": "user", "content": user_query})

    # Retrieve answer from Vector DB
    with st.chat_message("assistant"):
        with st.spinner("Searching database..."):
            # Search for the top match (k=1)
            search_results = st.session_state.vector_db.similarity_search(user_query, k=1)
            
            # FIXED: Correctly grab the first item [0] from the returned results list
            if search_results:
                best_match = search_results[0].page_content
                response = f"Based on my database, the closest match is:\n\n> {best_match}"
            else:
                response = "I couldn't find any relevant context in my database."
            
            st.write(response)
            
    # Append the assistant's response to conversation history
    st.session_state.messages.append({"role": "assistant", "content": response})
