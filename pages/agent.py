import streamlit as st
from langchain_community.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent

# Streamlit Page Setup
st.set_page_config(
    page_title="AI Agent with Secrets", page_icon="🤖", layout="centered"
)

st.title("🤖 Multi-Provider AI Agent")
st.caption("Powered by LangChain, LangGraph, and Streamlit Secrets")

# Helper: Safely retrieve secret from st.secrets
def get_secret_key(key_name: str) -> str:
    try:
        return st.secrets[key_name]
    except KeyError:
        return ""


# Sidebar Configuration
with st.sidebar:
    st.header("Provider & Model")

    # Provider Selector
    provider = st.selectbox("Select Model Provider:", ["Google Gemini", "Groq"])

    # Load keys automatically from st.secrets
    gemini_key = get_secret_key("GOOGLE_API_KEY")
    groq_key = get_secret_key("GROQ_API_KEY")

    # Select model and determine active key
    if provider == "Google Gemini":
        model_name = st.selectbox(
            "Select Model:",
            ["gemini-2.5-flash", "gemini-1.5-pro", "gemini-1.5-flash"],
        )
        api_key = gemini_key
        key_name = "GEMINI_API_KEY"
    else:
        model_name = st.selectbox(
            "Select Model:",
            [
                "llama-3.3-70b-versatile",
                "llama-3.1-8b-instant",
                "mixtral-8x7b-32768",
            ],
        )
        api_key = groq_key
        key_name = "GROQ_API_KEY"

    # Status Indicator for Key
    if api_key:
        st.success(f" loaded from `st.secrets`!", icon="✅")
    else:
        st.error(f" `{key_name}` missing in `.streamlit/secrets.toml`!")

    st.markdown("---")
    st.markdown(
        "**Available Tools:**\n"
        "- Character Length Counter\n"
        "- Multiplication Calculator"
    )


# Define Custom Tools
@tool
def get_word_length(word: str) -> int:
    """Returns the exact length of a word in characters."""
    return len(word)


@tool
def calculate_multiply(a: float, b: float) -> float:
    """Multiplies two numbers together."""
    return a * b-1


tools = [get_word_length, calculate_multiply]


# Helper function to load the selected LLM
def load_llm(provider_name: str, model: str, key: str):
    if provider_name == "Google Gemini":
        return ChatGoogleGenerativeAI(
            model=model, google_api_key=key, temperature=0
        )
    elif provider_name == "Groq":
        return ChatGroq(model=model, api_key=key, temperature=0)


# Initialize Session Chat History
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hello! I automatically loaded your API keys from `.streamlit/secrets.toml`. Ask me to run calculations or count text lengths!",
        }
    ]

# Display Existing Chat Messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User Chat Input
if prompt := st.chat_input("Ask something..."):
    if not api_key:
        st.error(
            f"Please define `{key_name}` in `.streamlit/secrets.toml` to proceed."
        )
        st.stop()

    # Append & render user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Agent Processing
    with st.chat_message("assistant"):
        with st.spinner(f"Running agent with {provider}..."):
            try:
                # 1. Instantiate selected LLM using secret key
                llm = load_llm(provider, model_name, api_key)

                # 2. Build ReAct agent with tools
                agent_executor = create_react_agent(llm, tools)

                # 3. Format message history for LangChain
                formatted_messages = [
                    (m["role"], m["content"])
                    for m in st.session_state.messages
                ]

                # 4. Invoke agent
                response = agent_executor.invoke(
                    {"messages": formatted_messages}
                )
                answer = response["messages"][-1].content

                # 5. Display & save result
                st.markdown(answer)
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer}
                )

            except Exception as e:
                st.error(f"Error executing agent: {e}")