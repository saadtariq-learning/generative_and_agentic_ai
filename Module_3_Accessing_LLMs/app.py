import streamlit as st
import os
import base64
from pathlib import Path
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_ollama import ChatOllama
from langchain_openrouter import ChatOpenRouter
from langchain_core.messages import HumanMessage, SystemMessage

# Set page config
st.set_page_config(
    page_title="LLM Access Hub",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title
st.title("🤖 LLM Access Hub")
st.markdown("Access multiple LLM providers in one place!")

# Sidebar for provider selection
with st.sidebar:
    st.header("Configuration")
    provider = st.selectbox(
        "Select LLM Provider",
        ["OpenAI", "Claude (Anthropic)", "Gemini (Google)", "GROQ", 
         "Hugging Face", "Ollama", "OpenRouter"]
    )
    
    # Temperature slider
    temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.1)
    
    # Max tokens
    max_tokens = st.number_input("Max Tokens", min_value=1, value=1024)

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.header("Chat Interface")
    user_input = st.text_area("Enter your prompt:", height=100)

with col2:
    st.header("Settings")
    st.info(f"**Provider:** {provider}\n\n**Temperature:** {temperature}\n\n**Max Tokens:** {max_tokens}")

# Function to initialize model based on provider
def get_model(provider, temperature, max_tokens):
    try:
        if provider == "OpenAI":
            return ChatOpenAI(
                model="gpt-4o-mini",
                temperature=temperature,
                max_tokens=max_tokens,
                api_key=os.getenv("OPENAI_API_KEY")
            )
        
        elif provider == "Claude (Anthropic)":
            return ChatAnthropic(
                model="claude-3-5-sonnet-20241022",
                temperature=temperature,
                max_tokens=max_tokens,
                api_key=os.getenv("ANTHROPIC_API_KEY")
            )
        
        elif provider == "Gemini (Google)":
            return ChatGoogleGenerativeAI(
                model="gemini-2.0-flash",
                temperature=temperature,
                api_key=os.getenv("GOOGLE_API_KEY")
            )
        
        elif provider == "GROQ":
            return ChatGroq(
                model="mixtral-8x7b-32768",
                temperature=temperature,
                max_tokens=max_tokens,
                api_key=os.getenv("GROQ_API_KEY")
            )
        
        elif provider == "Hugging Face":
            endpoint = HuggingFaceEndpoint(
                repo_id="meta-llama/Llama-3.3-70B-Instruct",
                task="text-generation",
                max_new_tokens=max_tokens,
                api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
            )
            return ChatHuggingFace(llm=endpoint)
        
        elif provider == "Ollama":
            return ChatOllama(
                model="llama3",
                temperature=temperature,
                base_url="http://localhost:11434"
            )
        
        elif provider == "OpenRouter":
            return ChatOpenRouter(
                model_name="anthropic/claude-3.5-sonnet",
                max_tokens=max_tokens,
                temperature=temperature,
                api_key=os.getenv("OPENROUTER_API_KEY")
            )
    
    except Exception as e:
        return None, str(e)

# Generate response button
if st.button("🚀 Generate Response", type="primary"):
    if not user_input.strip():
        st.warning("Please enter a prompt!")
    else:
        try:
            with st.spinner(f"Generating response from {provider}..."):
                model = get_model(provider, temperature, max_tokens)
                
                if model is None:
                    st.error("Failed to initialize model")
                else:
                    messages = [
                        SystemMessage(content="You are a helpful AI assistant."),
                        HumanMessage(content=user_input)
                    ]
                    
                    response = model.invoke(messages)
                    
                    st.success("✅ Response generated!")
                    st.markdown("### Response:")
                    st.write(response.content)
                    
                    # Copy button
                    st.button("📋 Copy Response")
        
        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.info("💡 Make sure you have the required API key set as an environment variable.")

