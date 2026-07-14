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
                model="llama2",
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

# Tabs for additional features
tab1, tab2, tab3 = st.tabs(["📝 About Providers", "🔑 API Keys", "📚 Usage Examples"])

with tab1:
    st.header("About LLM Providers")
    
    providers_info = {
        "OpenAI": {
            "description": "GPT-4 and GPT-3.5 models from OpenAI",
            "features": ["Text generation", "Image generation", "Image understanding"],
            "models": ["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"]
        },
        "Claude (Anthropic)": {
            "description": "Claude models from Anthropic",
            "features": ["Text generation", "Long context", "Code understanding"],
            "models": ["claude-3-5-sonnet", "claude-3-opus", "claude-3-haiku"]
        },
        "Gemini (Google)": {
            "description": "Google's Gemini models",
            "features": ["Text generation", "Image understanding", "Audio processing"],
            "models": ["gemini-2.0-flash", "gemini-1.5-pro", "gemini-1.5-flash"]
        },
        "GROQ": {
            "description": "High-speed inference with GROQ",
            "features": ["Fast inference", "Cost-effective", "Multiple models"],
            "models": ["mixtral-8x7b-32768", "llama-2-70b", "qwen-32b"]
        },
        "Hugging Face": {
            "description": "Open-source models from Hugging Face Hub",
            "features": ["Open-source", "Customizable", "Wide model selection"],
            "models": ["Llama-3.3-70B", "Mistral-7B", "Neural Chat"]
        },
        "Ollama": {
            "description": "Run models locally with Ollama",
            "features": ["Local execution", "Privacy", "No API calls"],
            "models": ["llama2", "mistral", "neural-chat", "bakllava"]
        },
        "OpenRouter": {
            "description": "Unified API for multiple LLM providers",
            "features": ["Multiple providers", "Load balancing", "Usage tracking"],
            "models": ["Claude", "GPT-4", "Llama", "Mistral"]
        }
    }
    
    for name, info in providers_info.items():
        with st.expander(f"ℹ️ {name}", expanded=False):
            st.write(f"**Description:** {info['description']}")
            st.write(f"**Features:** {', '.join(info['features'])}")
            st.write(f"**Available Models:** {', '.join(info['models'])}")

with tab2:
    st.header("API Keys Setup")
    
    keys_info = {
        "OpenAI": "https://platform.openai.com/api-keys",
        "Claude": "https://console.anthropic.com/",
        "Gemini": "https://ai.google.dev/",
        "GROQ": "https://console.groq.com/",
        "Hugging Face": "https://huggingface.co/settings/tokens",
        "OpenRouter": "https://openrouter.ai/keys"
    }
    
    st.markdown("""
    **To use each provider, set the following environment variables:**
    
    ```bash
    # OpenAI
    export OPENAI_API_KEY="your-key-here"
    
    # Claude (Anthropic)
    export ANTHROPIC_API_KEY="your-key-here"
    
    # Gemini (Google)
    export GOOGLE_API_KEY="your-key-here"
    
    # GROQ
    export GROQ_API_KEY="your-key-here"
    
    # Hugging Face
    export HUGGINGFACEHUB_API_TOKEN="your-token-here"
    
    # OpenRouter
    export OPENROUTER_API_KEY="your-key-here"
    
    # For Ollama, no API key needed (runs locally)
    ```
    """)
    
    st.markdown("**Get your API keys:**")
    for provider, url in keys_info.items():
        st.markdown(f"- [{provider}]({url})")

with tab3:
    st.header("Usage Examples")
    
    examples = {
        "General Question": "What is machine learning and how is it used in real-world applications?",
        "Code Generation": "Write a Python function to calculate the factorial of a number",
        "Explanation": "Explain quantum computing in simple terms",
        "Translation": "Translate 'Hello, how are you?' to Spanish and French",
        "Creative Writing": "Write a short story about a robot learning to paint"
    }
    
    st.markdown("**Click an example to use it:**")
    for example_name, example_text in examples.items():
        if st.button(example_name, key=f"example_{example_name}"):
            st.session_state.user_input = example_text
            st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>🚀 Powered by LangChain | Supporting 7+ LLM Providers</p>
    <p><small>Module 3: Accessing LLMs</small></p>
</div>
""", unsafe_allow_html=True)
