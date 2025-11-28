"""
Streamlit UI for RAG Knowledge Assistant
=========================================
Beautiful chat interface for the Ollama-powered RAG system
"""

import asyncio
import os
import streamlit as st
from dotenv import load_dotenv
import httpx
from datetime import datetime

# Load environment variables
load_dotenv(".env")

# Page configuration
st.set_page_config(
    page_title="RAG Knowledge Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
    }
    
    .user-message {
        background-color: #1e3a5f;
        border-left: 4px solid #2196F3;
        color: #ffffff;
    }
    
    .assistant-message {
        background-color: #2d2d2d;
        border-left: 4px solid #764ba2;
        color: #ffffff;
    }
    
    .message-role {
        font-weight: bold;
        margin-bottom: 0.5rem;
        color: #e0e0e0;
    }
    
    .message-content {
        color: #ffffff;
        line-height: 1.6;
    }
    
    .message-time {
        font-size: 0.75rem;
        color: #b0b0b0;
        margin-top: 0.5rem;
    }
    
    /* Streamlit chat message styling */
    .stChatMessage {
        background-color: #1a1a1a;
        border-radius: 0.5rem;
        padding: 1rem;
        margin-bottom: 1rem;
        color: #ffffff;
    }
    
    .stChatMessage[data-testid="chat-message-assistant"] {
        background-color: #2d2d2d;
        border-left: 4px solid #764ba2;
    }
    
    .stChatMessage[data-testid="chat-message-user"] {
        background-color: #1e3a5f;
        border-left: 4px solid #2196F3;
    }
    
    /* Make all text in chat messages white */
    .stChatMessage p, .stChatMessage div, .stChatMessage span {
        color: #ffffff !important;
    }
    
    .sidebar-info {
        padding: 1rem;
        background-color: #f8f9fa;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    
    .stButton>button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# API endpoint
API_URL = os.getenv("API_URL", "http://localhost:8000")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []


async def get_health_status():
    """Get API health status."""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{API_URL}/health")
            if response.status_code == 200:
                return response.json()
            return None
    except Exception as e:
        return None


async def send_message_stream(message: str):
    """Send message to the API and get streaming response."""
    try:
        async with httpx.AsyncClient(timeout=360.0) as client:
            async with client.stream(
                "POST",
                f"{API_URL}/chat/stream",
                json={
                    "message": message,
                    "conversation_history": st.session_state.conversation_history
                }
            ) as response:
                if response.status_code != 200:
                    error_text = await response.aread()
                    yield {
                        "error": f"API returned status {response.status_code}: {error_text.decode()}"
                    }
                    return
                
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        try:
                            import json
                            data = json.loads(line[6:])  # Remove "data: " prefix
                            yield data
                        except json.JSONDecodeError:
                            continue
                            
    except httpx.ReadTimeout:
        yield {
            "error": "⏱️ The model is taking too long to respond (>6 minutes). Try a shorter question or consider using a faster model."
        }
    except Exception as e:
        yield {
            "error": f"Error connecting to API: {str(e)}"
        }


async def send_message(message: str):
    """Send message to the API and get response."""
    try:
        async with httpx.AsyncClient(timeout=360.0) as client:  # Increased to 6 minutes
            response = await client.post(
                f"{API_URL}/chat",
                json={
                    "message": message,
                    "conversation_history": st.session_state.conversation_history
                }
            )
            
            if response.status_code == 200:
                data = response.json()
                # Ensure we have the expected structure
                if "response" in data and "conversation_history" in data:
                    return data
                else:
                    return {
                        "response": str(data),
                        "conversation_history": st.session_state.conversation_history
                    }
            else:
                error_text = response.text if hasattr(response, 'text') else str(response.status_code)
                return {
                    "response": f"Error: API returned status {response.status_code}: {error_text}",
                    "conversation_history": st.session_state.conversation_history
                }
    except httpx.ReadTimeout:
        return {
            "response": "⏱️ The model is taking too long to respond (>6 minutes). Try a shorter question or consider using a faster model.",
            "conversation_history": st.session_state.conversation_history
        }
    except Exception as e:
        return {
            "response": f"Error connecting to API: {str(e)}",
            "conversation_history": st.session_state.conversation_history
        }


def display_message(role: str, content: str, timestamp: str = None):
    """Display a chat message."""
    css_class = "user-message" if role == "user" else "assistant-message"
    icon = "👤" if role == "user" else "🤖"
    
    if timestamp is None:
        timestamp = datetime.now().strftime("%H:%M:%S")
    
    st.markdown(f"""
    <div class="chat-message {css_class}">
        <div><strong>{icon} {role.title()}</strong></div>
        <div>{content}</div>
        <div class="message-time">{timestamp}</div>
    </div>
    """, unsafe_allow_html=True)


# Header
st.markdown('<h1 class="main-header">🤖 RAG Knowledge Assistant</h1>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Health status
    with st.spinner("Checking API status..."):
        health = asyncio.run(get_health_status())
    
    if health:
        st.success("✅ API Connected")
        
        st.markdown('<div class="sidebar-info">', unsafe_allow_html=True)
        st.markdown("**Knowledge Base Stats**")
        kb_info = health.get("knowledge_base", {})
        st.metric("Documents", kb_info.get("documents", 0))
        st.metric("Chunks", kb_info.get("chunks", 0))
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="sidebar-info">', unsafe_allow_html=True)
        st.markdown("**Model Info**")
        model_info = health.get("model_info", {})
        st.text(f"Provider: {model_info.get('provider', 'N/A')}")
        st.text(f"Model: {model_info.get('model', 'N/A')}")
        st.text(f"Embeddings: {model_info.get('embedding_model', 'N/A')}")
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.error("❌ API Disconnected")
        st.warning("Make sure the API is running:\n```\nuv run python api_ollama.py\n```")
    
    st.divider()
    
    # Clear chat button
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.session_state.conversation_history = []
        st.rerun()
    
    st.divider()
    
    # Information
    st.markdown("### 💡 Tips")
    st.markdown("""
    - Ask questions about the documents in the knowledge base
    - The assistant will search and cite sources
    - Conversations maintain context
    - Clear history to start fresh
    """)

# Main chat area
chat_container = st.container()

# Display existing messages
with chat_container:
    for msg in st.session_state.messages:
        display_message(
            msg["role"],
            msg["content"],
            msg.get("timestamp")
        )

# Chat input
user_input = st.chat_input("Ask me anything about the knowledge base...")

if user_input:
    # Add user message
    timestamp = datetime.now().strftime("%H:%M:%S")
    st.session_state.messages.append({
        "role": "user",
        "content": user_input,
        "timestamp": timestamp
    })
    
    # Display user message
    with chat_container:
        display_message("user", user_input, timestamp)
    
    # Create placeholder for streaming response
    assistant_timestamp = datetime.now().strftime("%H:%M:%S")
    
    with chat_container:
        with st.chat_message("assistant"):
            st.text(f"🤖 Assistant • {assistant_timestamp}")
            status_placeholder = st.empty()
            message_placeholder = st.empty()
            
            status_placeholder.info("🔍 Searching knowledge base...")
            
            full_response = ""
            error_occurred = False
            
            # Collect streaming response
            async def collect_response():
                response_text = ""
                error = False
                
                async for event in send_message_stream(user_input):
                    if "error" in event:
                        status_placeholder.error(f"❌ {event['error']}")
                        response_text = event['error']
                        error = True
                        break
                    elif "status" in event:
                        if event["status"] == "searching":
                            status_placeholder.info("🔍 Searching knowledge base...")
                        elif event["status"] == "generating":
                            status_placeholder.info("✍️ Generating response...")
                        elif event["status"] == "done":
                            status_placeholder.empty()
                            if "response" in event:
                                response_text = event["response"]
                    elif "chunk" in event:
                        response_text += event["chunk"]
                        # Update display
                        message_placeholder.markdown(response_text + "▌")
                        
                status_placeholder.empty()
                return response_text, error
            
            full_response, error_occurred = asyncio.run(collect_response())
            message_placeholder.markdown(full_response)
    
    # Add assistant message to history
    st.session_state.messages.append({
        "role": "assistant",
        "content": full_response,
        "timestamp": assistant_timestamp
    })
    
    # Update conversation history
    if not error_occurred:
        st.session_state.conversation_history.append({
            "role": "user",
            "content": user_input
        })
        st.session_state.conversation_history.append({
            "role": "assistant",
            "content": full_response
        })
    
    # Rerun to update the UI
    st.rerun()

# Footer
st.divider()
st.markdown(
    "<div style='text-align: center; color: #666;'>"
    "Powered by Ollama & PostgreSQL/PGVector"
    "</div>",
    unsafe_allow_html=True
)
