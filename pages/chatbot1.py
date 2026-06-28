import os
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage

# 1. CRITICAL CORRECTION: st.set_page_config must be the FIRST command executed
st.set_page_config(page_title="Gemini Chatbot", page_icon="💬", layout="centered")

st.title("💬 Gemini Chatbot")

st.write("Start chatting! The AI remembers your previous messages.")
st.markdown("---")

# 2. INTERACTIVE SIDEBAR FOR CREDENTIAL PROTECTION
with st.sidebar:
    st.subheader("🔑 SECURE KEY CONFIGURATION")
    # Tries to pull from env or defaults to an interactive entry bar
    google_api_key = st.text_input(
        "Enter Google API Key:", 
        type="password", 
        placeholder="AIzaSy...",
        value=os.environ.get("GOOGLE_API_KEY", "")
    )
    if not google_api_key:
        st.warning("⚠️ Please provide a valid Google API Key to authorize request blocks.")

# 3. DYNAMIC MODEL AND PARAMETER SELECTOR CONFIGURATION BLOCK
col_config1, col_config2 = st.columns(2)

with col_config1:
    selected_model = st.selectbox(
        "🧠 Select Active Gemini Model Architecture:",
        [
            "gemini-2.5-flash",
            "gemini-2.5-pro",
            "gemini-1.5-flash",
            "gemini-1.5-pro",
            "gemini-1.0-pro",
            "gemini-3.5-flash",
        ],
        index=0
    )

with col_config2:
    selected_temp = st.slider(
        "🎛️ Creativity Settings (Temperature):",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.1,
        help="Lower values are highly focused/factual, higher values generate unique/creative text."
    )

st.markdown("---")

# 4. Initialize memory array inside Streamlit's persistent session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# 5. Permanently display the entire chat history on screen
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 6. Provide a fixed bottom chat entry bar
if user_input := st.chat_input("Type your message here..."):
    
    if not google_api_key:
        st.error("Operation halted: Secure API Key array is empty. Insert credentials in the side manager.")
    else:
        # Render the user message box immediately
        with st.chat_message("user"):
            st.markdown(user_input)
        
        # Save user message to memory state
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        # Generate response from LangChain model
        with st.chat_message("assistant"):
            with st.spinner("Typing..."):
                try:
                    # FIX: Explicitly pass the API Key variable here into the class constructor
                    llm = ChatGoogleGenerativeAI(
                        model=selected_model, 
                        temperature=selected_temp,
                        api_key=google_api_key
                    )
                    
                    # Convert tracking array into specialized LangChain message structures
                    langchain_messages = []
                    for msg in st.session_state.chat_history:
                        if msg["role"] == "user":
                            langchain_messages.append(HumanMessage(content=msg["content"]))
                        else:
                            langchain_messages.append(AIMessage(content=msg["content"]))
                    
                    # Request response with full past text blocks attached
                    res = llm.invoke(langchain_messages)
                    
                    # Render only the output string on-screen
                    st.markdown(res.content)
                    
                    # Save assistant response to memory state
                    st.session_state.chat_history.append({"role": "assistant", "content": res.content})
                    
                except Exception as e:
                    st.error(f"An error occurred during inference routing: {e}")
