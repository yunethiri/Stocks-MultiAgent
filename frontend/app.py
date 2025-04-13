from typing import Dict

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import requests
import json
import os
import io
import base64
from io import BytesIO
from PIL import Image

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
        "stock_data": pd.DataFrame(),
        "pending_prompt": None,
        "chart_image": None,
        "needs_rerun": False,
    }

    for key, value in session_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def get_stock_metrics(df: pd.DataFrame) -> Dict[str, float]:
    """Extract key stock metrics from DataFrame."""
    if df.empty:
        return {}

    return {
        "current_price": df["Close"].iloc[-1],
        "open_price": df["Open"].iloc[-1],
        "high_price": df["High"].max(),
        "low_price": df["Low"].min(),
    }


def create_candlestick_chart(df: pd.DataFrame, symbol: str) -> go.Figure:
    """create plotly candlestick chart from stock data."""
    fig = go.Figure()

    if not df.empty:
        fig.add_trace(
            go.Candlestick(
                x=df.index,  # xaxis
                open=df["Open"],
                high=df["High"],
                low=df["Low"],
                close=df["Close"],
                name="Candlestick",
            )
        )

    fig.update_layout(
        title=f"{symbol} Stock Price",
        yaxis_title="Price ($)",
        xaxis_title="Date",
        xaxis_rangeslider_visible=False,
        height=500,
    )

    fig.update_xaxes(
        dtick="M1",  # Tick every month
        tick0="2024-01-02",
        tickformat="%b %Y",  # Format ticks as "Jan 2024", "Feb 2024", etc.
    )

    return fig


def display_stock_metrics(metrics: Dict[str, float]) -> None:
    """Display stock metrics in columns."""
    cols = st.columns(4)
    metric_labels = {
        "current_price": "Current Price",
        "open_price": "Open Price",
        "high_price": "High Price",
        "low_price": "Low Price",
    }

    for (metric, label), col in zip(metric_labels.items(), cols):
        value = metrics.get(metric, "N/A")
        if isinstance(value, (float, int)):
            value = f"${value:.2f}"
        col.metric(label=label, value=value)


def display_chat_messages() -> None:
    """Display chat messages from session history."""
    for message in st.session_state.messages:
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


def process_prompt(prompt: str) -> None:
    """Common processing for both example prompts and direct input."""
    # add user's message to history
    st.session_state.messages.append({"role": "user", "content": prompt})

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
                    st.session_state.messages.append(
                        {"role": "assistant", "content": response_text}
                    )
                    safe_response = response_text.replace('$','\$')
                    st.write(safe_response)
                
                # if visualisation agent 
                if image_b64:
                    try:
                        image_bytes = BytesIO(base64.b64decode(image_b64))
                        image = Image.open(image_bytes)
                        st.session_state.chart_image = image
                        st.image(image, use_container_width=True)

                        # provide download button
                        st.download_button(
                            label="📥 Download Image",
                            data=image_bytes,
                            file_name="chart.png",
                            mime="image/png"
                        )

                    except Exception as e:
                        st.error(f"Error decoding image: {e}")

            except Exception as e:
                # logging.error(f"Chat error: {str(e)}")
                st.error("Failed to generate response. Please try again.")
    
    st.session_state.needs_rerun = True


def generate_chat_response(prompt: str) -> str:
    payload = {"query": prompt,}
    try:
        # Use 'params' to send as URL query parameters
        response = requests.post(url=f'{BACKEND_URL}/query', params=payload)
        response_text = response.content.decode('utf-8')
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
        process_prompt(prompt)

    # direct user input
    if prompt := st.chat_input("Ask about the stock..."):
        process_prompt(prompt)


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
    st.title("📊 Smart Stock Insight Chatbot")  ## to edit to make it more special
    with st.sidebar:
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

    try:
        df = pd.read_csv("AAPL_2024_stock_data.csv")
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date", ascending=True)
        df.set_index("date", inplace=True)

        st.session_state.stock_data = df

        # Display stock data section
        #if not st.session_state.stock_data.empty:
            #metrics = get_stock_metrics(st.session_state.stock_data)
            #display_stock_metrics(metrics)

            #fig = create_candlestick_chart(
            #    st.session_state.stock_data, st.session_state.current_symbol
            #)
            #st.plotly_chart(fig, use_container_width=True)
        #else:
        #    st.warning("No stock data available for the selected symbol.")

        # Chat interface
        #st.subheader("Stock Analysis Chat")
        st.write("📅 Note: This app currently provides AAPL data exclusively for the year 2024.")
        display_example_prompts()
        display_chat_messages()
        handle_chat_input()
        """Handle stock symbol selection in sidebar and sidebar information is displayed."""


    except Exception as e:
        st.error("Error: " + str(e))

if __name__ == "__main__":
    main()
