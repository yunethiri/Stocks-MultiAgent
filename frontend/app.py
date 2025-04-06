from typing import Dict

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from api import get_stock_data
from openai import OpenAI

openai_api_key = st.secrets["openai_api_key"]  # Use OpenAI API key
client = OpenAI(api_key = openai_api_key)

# constants
STOCK_SYMBOLS = [
    "AAPL"
]
DEFAULT_SYMBOL = "AAPL"
EXAMPLE_PROMPTS = [
    "What is your role?",
    "Give me an overview of the stock",
    "What is the highest Close rate?",
    "What are the key technical indicators suggesting?",
    "What is the lowest Open rate?",
    "Show me support and resistance levels"
]


def init_session_state() -> None:
    """creating session state variables."""
    session_defaults = {
        'messages': [],
        'current_symbol': DEFAULT_SYMBOL,
        'stock_data': pd.DataFrame(),
        'pending_prompt': None
    }

    for key, value in session_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def get_stock_metrics(df: pd.DataFrame) -> Dict[str, float]:
    """Extract key stock metrics from DataFrame."""
    if df.empty:
        return {}

    return {
        'current_price': df['Close'].iloc[-1],
        'open_price': df['Open'].iloc[-1],
        'high_price': df['High'].max(),
        'low_price': df['Low'].min()
    }


def create_candlestick_chart(df: pd.DataFrame, symbol: str) -> go.Figure:
    """create plotly candlestick chart from stock data."""
    fig = go.Figure()

    if not df.empty:
        fig.add_trace(go.Candlestick(
            x=df.index,    #xaxis
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='Candlestick'
        ))

    fig.update_layout(
        title=f"{symbol} Stock Price",
        yaxis_title="Price ($)",
        xaxis_title="Date",
        xaxis_rangeslider_visible=False,
        height=500
    )

    fig.update_xaxes(
        dtick="M1",           # Tick every month
        tick0="2024-01-02",
        tickformat="%b %Y"    # Format ticks as "Jan 2024", "Feb 2024", etc.
    )

    return fig


def display_stock_metrics(metrics: Dict[str, float]) -> None:
    """Display stock metrics in columns."""
    cols = st.columns(4)
    metric_labels = {
        'current_price': "Current Price",
        'open_price': "Open Price",
        'high_price': "High Price",
        'low_price': "Low Price"
    }

    for (metric, label), col in zip(metric_labels.items(), cols):
        value = metrics.get(metric, 'N/A')
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
    st.write("**Try these example questions:**")
    cols = st.columns(2)
    for idx, prompt in enumerate(EXAMPLE_PROMPTS):
        with cols[idx % 2]:
            if st.button(
                prompt,
                key=f"ex_{idx}",
                use_container_width=True,
                help="Click to use this example question"
            ):
                st.session_state.pending_prompt = prompt


# def handle_stock_selection() -> None:
#     """Handle stock symbol selection in sidebar and sidebar information is displayed."""
#     with st.sidebar:
#         st.header("Settings")
#         new_symbol = st.selectbox(
#             "Select Stock Symbol:",
#             options=STOCK_SYMBOLS,
#             index=STOCK_SYMBOLS.index(st.session_state.current_symbol)
#         )

#         st.subheader("How to use:")
#         st.write("1. Select a stock symbol from the dropdown")
#         st.write("2. Type a question about the stock in the input box")
#         st.write("3. Press Enter to get the answer from the chatbot")

#         st.subheader("About:")
#         st.write("This app uses OpenAI's chat model to analyse stock data.")
#         st.write(
#             "The chatbot is limited to a maximum of 5 days of data.") #to be changed 
#         st.write("API pulls the last 5 months of data.")

#         if new_symbol != st.session_state.current_symbol:
#             st.session_state.current_symbol = new_symbol
#             st.session_state.messages = []
#             try:
#                 st.session_state.stock_data = get_stock_data(
#                     st.secrets["stock_api_key"],
#                     new_symbol
#                 )
#             except Exception as e:
#                 st.error(f"Failed to load data: {str(e)}")
#             st.rerun()


def process_prompt(prompt: str) -> None:
    """Common processing for both example prompts and direct input."""
    # add user's message to history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # display the user's message instantly
    with st.chat_message("user"):
        st.write(prompt)

    # generate and display chatbot response
    with st.chat_message("assistant"):
        with st.spinner("Analysing stock data..."):
            try:
                response = generate_chat_response(prompt)
                st.session_state.messages.append(
                    {"role": "assistant", "content": response}
                )
                st.write(response)
            except Exception as e:
                # logging.error(f"Chat error: {str(e)}")
                st.error("Failed to generate response. Please try again.")

    # rerun to update display
    st.rerun()


def generate_chat_response(prompt: str) -> str:
    """Generate chatbot response using OpenAI API."""
    stock_name = st.session_state.current_symbol  # stock name pulled for context
    context = f"Stock data for {stock_name}: {st.session_state.stock_data.head(5).to_string()}"
    messages = [{
        "role": "user",
        "content": f"{context}\n\nQuestion: {prompt}"
    }]

    try:
        # Call OpenAI's ChatCompletion endpoint
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",  # or "gpt-4", depending on the model you want to use
            messages=messages,
            max_tokens=1000  # Adjust max tokens as needed
        )
        
        # Extract and return the response from OpenAI API
        return response.choices[0].message.content

    except Exception as e:
        st.error(f"Failed to generate response: {str(e)}")
        return "Error occurred. Please try again later."


def handle_chat_input() -> None:
    """Process and display chat messages."""
    # checking for pending prompt from examples
    if st.session_state.pending_prompt:
        prompt = st.session_state.pop('pending_prompt')
        process_prompt(prompt)

    #direct user input
    if prompt := st.chat_input("Ask about the stock..."):
        process_prompt(prompt)


def main():
    """Main application entry point."""
    # page settings
    st.set_page_config(
        page_title="Stock Analysis AI Chatbot",
        layout="wide",
        initial_sidebar_state="expanded",
        page_icon="📈"
    )

    # initialise the application state
    init_session_state()

    # page title
    st.title("Stock Analysis Chatbot")  ## to edit to make it more special 
    with st.sidebar:
        st.header("Settings")
        new_symbol = st.selectbox(
            "Select Stock Symbol:",
            options=STOCK_SYMBOLS,
            index=STOCK_SYMBOLS.index(st.session_state.current_symbol)
        )

        st.subheader("How to use:")
        st.write("1. Select a stock symbol from the dropdown")
        st.write("2. Type a question about the stock in the input box")
        st.write("3. Press Enter to get the answer from the chatbot")

        st.subheader("About:")
        st.write("This app uses OpenAI's chat model to analyse stock data.")
        st.write("API pulls 2024 data.")


    try:
        df = pd.read_csv('frontend/AAPL_2024_stock_data.csv')
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date', ascending=True)
        df.set_index('date', inplace=True)

        st.session_state.stock_data = df

        
        # Display stock data section
        if not st.session_state.stock_data.empty:
            metrics = get_stock_metrics(st.session_state.stock_data)
            display_stock_metrics(metrics)

            fig = create_candlestick_chart(
                st.session_state.stock_data,
                st.session_state.current_symbol
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No stock data available for the selected symbol.")

        # Chat interface
        st.subheader("Stock Analysis Chat")
        st.write("Displaying stock data for the full year 2024.")
        display_example_prompts()
        display_chat_messages()
        handle_chat_input()
        """Handle stock symbol selection in sidebar and sidebar information is displayed."""
   
    except Exception as e:
        st.error("Error: " + str(e))

if __name__ == "__main__":
    main()
