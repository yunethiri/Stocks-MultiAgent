from datetime import datetime, timedelta
from typing import Dict, List

import pandas as pd
import requests
import streamlit as st
from openai import OpenAI


# constants
FMP_BASE_URL = "https://financialmodelingprep.com/api/v3"
SYSTEM_PROMPT = """You are a stock market analyst. Your role is to use this data 
to answer questions about the stock"""  # OpenAI initial prompt context
stock_api_key = st.secrets["stock_api_key"]
openai_api_key = st.secrets["openai_api_key"]  # Use OpenAI API key

# OpenAI API setup
# openai.api_key = openai_api_key
client = OpenAI(api_key = openai_api_key)

def get_stock_data(api_key: str, symbol: str) -> pd.DataFrame:
    """
    Fetch historical stock data using the Financial Modeling Prep API.

    Args:
        api_key (str): FMP API key
        symbol (str): Stock symbol ('AAPL', 'GOOGL', etc)

    Returns:
        pd.DataFrame: Historical stock data with columns:
            - Open: Opening price
            - High: Highest price
            - Low: Lowest price
            - Close: Closing price
            - Adjusted Close: Adjusted closing price
            - Volume: Trading volume

    Raises:
        ValueError: If API request fails or no data is found
    """
    # calculate date range 
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 12, 31)

    try:
        url = f"{FMP_BASE_URL}/historical-price-full/{symbol}"
        params = {
            "from": start_date.strftime("%Y-%m-%d"),
            "to": end_date.strftime("%Y-%m-%d"),
            "apikey": api_key
        }
        response = requests.get(url, params=params)
        response.raise_for_status()

        data = response.json()
        historical_data = data.get('historical', [])
        if not historical_data:
            raise ValueError(f"No historical data found for symbol: {symbol}")

        df = pd.DataFrame(historical_data)
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)

        # column name mapping
        column_mapping = {
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'adjClose': 'Adjusted Close',
            'volume': 'Volume'
        }
        df.rename(columns=column_mapping, inplace=True)

        return df

    except requests.RequestException as e:
        raise ValueError(f"Failed to fetch stock data: {str(e)}")
    except Exception as e:
        raise ValueError(f"Error processing stock data: {str(e)}")


## this function is not used in app.py
def openai_chat(api_key: str, messages: List[Dict[str, str]]) -> str:
    """
    Generate chat responses using the OpenAI API.

    Args:
        api_key (str): OpenAI API key
        messages (List[Dict[str, str]]): Messages to send to the OpenAI model

    Returns:
        str: response from the model
    """
    try:
        print(f"Starting OpenAI client with model: gpt-3.5-turbo")

        # Constructing the full conversation context with the system prompt
        full_messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            *messages
        ]

        # API request to OpenAI's chat endpoint
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",  # Change to gpt-4 if needed
            messages=full_messages,
            max_tokens=500  # adjust this as needed
        )

        if not response or not hasattr(response, 'choices') or len(response.choices) == 0:
            print("Empty or invalid response received:", response)
            raise ValueError("Invalid response from API")

        print("Successful response received")
        return response.choices[0].message.content

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error details: {error_details}")
        st.error(f"API error: {str(e)}")
        return "Cannot connect to the API. Please try again later."


# save data to local directory
if __name__ == "__main__":
    api_key = stock_api_key
    symbol = "AAPL"
    
    df = get_stock_data(api_key, symbol)
    
    csv_filename = f"{symbol}_2024_stock_data.csv"
    df.to_csv(csv_filename)
    
    print(f"Data saved to {csv_filename}")