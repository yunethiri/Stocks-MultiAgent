import streamlit as st
import requests
import json
import os
import base64
from io import BytesIO
from PIL import Image
import uuid
from datetime import datetime

# Set the backend URL from environment (defaults to 127.0.0.1 for local testing)
BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")

# constants
STOCK_SYMBOLS = ["AAPL"]
DEFAULT_SYMBOL = "AAPL"
EXAMPLE_PROMPTS = [
    "What are the main takeaways from Apple's 10K Report for 2024?",
    "Summarise the revenue trends in Apple's 10Q report in Q2 2024.",
    "What was the tone of Apple’s CEO in the Q1 2024 earnings call?",
    "Summarize how the media portrayed Apple after its Q2 2024 earnings release.",
    "What is the sentiment of financial news about Apple Stocks in November 2024?",
    "What consistent issues are highlighted across Apple’s 10Q reports and news in 2024?",
    "Provide a visualisation of Apple Stocks in Q3 2024.",
]


def init_session_state() -> None:
    """creating session state variables."""
    session_defaults = {
        "messages": [],
        "current_symbol": DEFAULT_SYMBOL,
        "pending_prompt": None,
        "needs_rerun": False,
        "sessions": {},
        "rename_mode": False,
        "session_loaded": {},
    }

    for key, value in session_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    if "current_session_id" not in st.session_state:
        # Create a default session
        session_id = str(uuid.uuid4())
        st.session_state.sessions[session_id] = {
            "name": "New Chat",
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "messages": []
        }
        st.session_state.current_session_id = session_id

def create_new_session():
    """Create a new chat session"""
    session_id = str(uuid.uuid4())
    st.session_state.sessions[session_id] = {
        "name": "New Chat",
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "messages": []
    }
    st.session_state.current_session_id = session_id
    st.session_state.rename_mode = False
    st.session_state.session_loaded[session_id] = True

def delete_session(session_id):
    """Delete a chat session"""
    if session_id in st.session_state.sessions:
        # Delete from backend
        response = requests.delete(f"{BACKEND_URL}/history/{session_id}")

        if response.status_code == 200: # success
            # Delete from frontend
            del st.session_state.sessions[session_id]
            if session_id in st.session_state.session_loaded:
                del st.session_state.session_loaded[session_id]
            
            # If we deleted the current session, select another one
            if session_id == st.session_state.current_session_id:
                if st.session_state.sessions:
                    # Switch to the first available session
                    st.session_state.current_session_id = next(iter(st.session_state.sessions))
                else:
                    # No sessions left, create a new one
                    create_new_session()

def rename_session(session_id, new_name):
    """Rename a chat session"""
    if session_id in st.session_state.sessions and new_name.strip():
        st.session_state.sessions[session_id]["name"] = new_name.strip()
    st.session_state.rename_mode = False

def load_session_if_needed(session_id):
    """Load session from backend if not loaded already"""
    if session_id not in st.session_state.session_loaded or not st.session_state.session_loaded[session_id]:
        try:
            # Get chat history from backend
            response = requests.get(f"{BACKEND_URL}/history/{session_id}")
            history = response.json()
            
            # Format messages for display
            messages = []
            for item in history:
                if "user_message" in item and item["user_message"]:
                    messages.append({"role": "user", "content": item["user_message"]})
                if "assistant_message" in item and item["assistant_message"]:
                    messages.append({"role": "assistant", "content": item["assistant_message"]})
            
            # Parse first user message for naming the chat if name is default
            name = "New Chat"
            created_at = datetime.now().strftime("%Y-%m-%d %H:%M")

            if messages and len(messages) >= 1:
                first_msg = next((msg for msg in messages if msg["role"] == "user"), None)
                if first_msg:
                    # Use the first few words of the first query as the chat name
                    words = first_msg["content"].split()
                    name = " ".join(words[:3]) + ("..." if len(words) > 3 else "")
                
                # Try to extract creation time from history if available
                if isinstance(history, list) and history and "timestamp" in history[0]:
                    try:
                        # Assuming timestamp is in ISO format or similar
                        timestamp = history[0]["timestamp"]
                        if isinstance(timestamp, str):
                            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                            created_at = dt.strftime("%Y-%m-%d %H:%M")
                    except (ValueError, TypeError):
                        pass
            
            # Update session info
            if session_id not in st.session_state.sessions:
                st.session_state.sessions[session_id] = {
                    "name": name,
                    "created_at": created_at,
                    "messages": messages
                }
            else:
                st.session_state.sessions[session_id]["messages"] = messages

                # Update name only if it's still the default
                if st.session_state.sessions[session_id]["name"] == "New Chat" and name != "New Chat":
                    st.session_state.sessions[session_id]["name"] = name
            
            st.session_state.session_loaded[session_id] = True

        except Exception as e:
            st.error(f"Error loading session: {str(e)}")
            # Create empty session if we can't load it
            if session_id not in st.session_state.sessions:
                st.session_state.sessions[session_id] = {
                    "name": "New Chat",
                    "created_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "messages": []
                }
            st.session_state.session_loaded[session_id] = True

def display_chat_messages() -> None:
    """Display the chat interface for the current session"""
    session_id = st.session_state.current_session_id

    # Load session from backend if needed
    load_session_if_needed(session_id)
    
    current_session = st.session_state.sessions[session_id]

    # Display session title
    st.title(f"💬 {current_session['name']}")

    for message in current_session["messages"]:
        with st.chat_message(message["role"]):
            st.write(message["content"])


def display_example_prompts() -> None:
    """Display example prompts."""
    st.write("**💡 Here are some example questions to get you started:**")
    cols = st.columns(2)
    for idx, prompt in enumerate(EXAMPLE_PROMPTS):
        with cols[idx % 2]:
            if st.button(
                prompt,
                key=f"ex_{idx}",
                use_container_width=True,
                help="Click to use this example question",
            ):
                st.session_state.pending_prompt = prompt


def process_prompt(prompt: str, current_session) -> None:
    """Common processing for both example prompts and direct input."""
    # add user's message to history
    current_session["messages"].append({"role": "user", "content": prompt})

    # display the user's message instantly
    with st.chat_message("user"):
        st.write(prompt)

    # generate and display chatbot response
    with st.chat_message("assistant"):
        with st.spinner("Generating Response..."):
            try:
                response_json = generate_chat_response(prompt)
                response_text = response_json.get("response")
                image_b64 = response_json.get("image_base64", None)

                if response_text:
                    current_session["messages"].append({"role": "assistant", "content": response_text})

                    if current_session["name"] == "New Chat" and len(current_session["messages"]) == 2:
                        # Use the first few words of the first query as the chat name
                        words = prompt.split()
                        name = " ".join(words[:3]) + ("..." if len(words) > 3 else "")
                        current_session["name"] = name

                    safe_response = response_text.replace("$", "\$")
                    st.write(safe_response)

                # if visualisation agent
                if image_b64:
                    try:
                        image_bytes = BytesIO(base64.b64decode(image_b64))
                        image = Image.open(image_bytes)
                        st.image(image, use_container_width=True)

                        # provide download button
                        st.download_button(
                            label="📥 Download Image",
                            data=image_bytes,
                            file_name="chart.png",
                            mime="image/png",
                        )

                    except Exception as e:
                        st.error(f"Error decoding image: {e}")

            except Exception as e:
                # logging.error(f"Chat error: {str(e)}")
                st.error("Failed to generate response. Please try again.")

    st.session_state.needs_rerun = True


def generate_chat_response(prompt: str) -> str:
    payload = {
        "query": prompt,
    }
    try:
        # Use 'params' to send as URL query parameters
        response = requests.post(url=f"{BACKEND_URL}/query", params=payload)
        response_text = response.content.decode("utf-8")
        response_json = json.loads(response_text)
        return response_json
    except requests.exceptions.RequestException as e:
        st.error("Error connecting to backend.")
        return "Failed to connect to backend."
    except Exception as e:
        st.error(f"Failed to generate response: {str(e)}")
        return "Error occurred. Please try again later."


def handle_chat_input() -> None:
    """Process and display chat messages."""
    # checking for pending prompt from examples
    if st.session_state.pending_prompt:
        prompt = st.session_state.pop("pending_prompt")
        session_id = st.session_state.current_session_id
        current_session = st.session_state.sessions[session_id]
        process_prompt(prompt, current_session)

    # direct user input
    if prompt := st.chat_input("Ask about the stock..."):
        session_id = st.session_state.current_session_id
        current_session = st.session_state.sessions[session_id]
        process_prompt(prompt, current_session)

def display_sidebar():
    """Display the sidebar with session management options"""
    with st.sidebar:
        st.header("💬 Chat Sessions")
        # New chat button
        if st.button("➕ New Chat", use_container_width=True):
            create_new_session()
            st.rerun()
        
        st.divider()

        # Sort sessions by creation time (newest first)
        sorted_sessions = sorted(
            st.session_state.sessions.items(),
            key=lambda x: datetime.strptime(x[1]["created_at"], "%Y-%m-%d %H:%M"),
            reverse=True
        )

        # List all sessions
        for session_id, session in sorted_sessions:
            col1, col2, col3 = st.columns([0.7, 0.15, 0.15])
            
            with col1:
                if st.session_state.get("rename_mode") == session_id:
                    # Rename mode
                    new_name = st.text_input("New name", 
                                            value=session["name"], 
                                            key=f"rename_{session_id}",
                                            label_visibility="collapsed")
                    if st.button("Save", key=f"save_{session_id}"):
                        rename_session(session_id, new_name)
                        st.rerun()
                else:
                    # Display mode - clickable session name
                    if st.button(session["name"], 
                                key=f"session_{session_id}", 
                                use_container_width=True,
                                type="secondary" if session_id != st.session_state.current_session_id else "primary"):
                        st.session_state.current_session_id = session_id
                        st.rerun()
            
            with col2:
                # Edit button
                if st.button("✏️", key=f"edit_{session_id}"):
                    st.session_state.rename_mode = session_id
                    st.rerun()
            
            with col3:
                # Delete button
                if st.button("🗑️", key=f"delete_{session_id}"):
                    delete_session(session_id)
                    st.rerun()
            
            # Show creation date in small text
            st.caption(f"Created: {session['created_at']}")

        st.header("⚙️ Settings")
        new_symbol = st.selectbox(
            "Select Stock Symbol:",
            options=STOCK_SYMBOLS,
            index=STOCK_SYMBOLS.index(st.session_state.current_symbol),
        )

        st.subheader("💡 How to Use:")
        st.markdown(
            """
            1. **Choose a stock** from the dropdown menu.  
            2. **Ask a question** related to the stock’s performance, reports, or news.  
            3. Press **Enter** to receive data-driven insights from the chatbot.
            """
        )

        st.subheader("ℹ️ About:")
        st.markdown(
            """
            This app uses **OpenAI's chat model** to provide smart analysis of **stock data from 2024**.  
            Insights are powered by:  
            - 📄 **10K / 10Q Reports**  
            - 📰 **Financial News**  
            - 🗣️ **Earnings Calls**  
            """
        )

def main():
    """Main application entry point."""
    # page settings
    st.set_page_config(
        page_title="Stock Analysis AI Chatbot",
        layout="wide",
        initial_sidebar_state="expanded",
        page_icon="📈",
    )

    # initialise the application state
    init_session_state()

    # page title
    st.title("📊 Stocks Chatbot")  ## to edit to make it more special
    display_sidebar()

    try:
        st.write(
            "📅 Note: This app currently provides AAPL data exclusively for the year 2024."
        )
        display_example_prompts()
        display_chat_messages()
        handle_chat_input()
        """Handle stock symbol selection in sidebar and sidebar information is displayed."""

    except Exception as e:
        st.error("Error: " + str(e))


if __name__ == "__main__":
    main()
