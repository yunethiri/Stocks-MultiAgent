import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Import required classes from LangChain for prompting and output parsing.
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

class StockVisualizer:
    def __init__(self, ticker='AAPL'):
        """
        Initializes the visualizer for a specific stock ticker.
        
        Parameters:
        - ticker (str): The stock symbol (default: 'AAPL' for Apple)
        """
        self.ticker = ticker
        self.data = None

    def download_data(self, start_date: str, end_date: str):
        """
        Downloads historical stock data for the given date range.
        
        Parameters:
        - start_date (str): Format 'YYYY-MM-DD'
        - end_date (str): Format 'YYYY-MM-DD'
        """
        print(f"Downloading data for {self.ticker} from {start_date} to {end_date}...")
        self.data = yf.download(self.ticker, start=start_date, end=end_date)
        if self.data.empty:
            raise ValueError("No data found. Please check the date range or ticker.")
        print("Download complete.")

    def plot_closing_price(self):
        """
        Plots the closing price of the stock and returns a matplotlib Figure.
        """
        if self.data is None:
            raise ValueError("No data to plot. Please call download_data() first.")

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(self.data.index, self.data['Close'], label=f'{self.ticker} Closing Price', linewidth=2)
        ax.set_title(f"{self.ticker} Stock Closing Price")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price (USD)")
        ax.grid(True)
        ax.legend()
        fig.tight_layout()
        return fig

    def process_query(self, query: str):
        """
        Uses an LLM to extract the start and end dates from the query,
        downloads stock data between those dates, generates a plot of the
        closing price, and returns the matplotlib Figure.
        
        Parameters:
        - query (str): The user's query containing date information.
        
        Returns:
        - fig: matplotlib Figure with the closing price plot.
        """
        # Instantiate an LLM (make sure your environment is set up with your OpenAI API key)
        llm = ChatOpenAI(temperature=0, model="gpt-4")

        # Create a prompt that extracts dates from the query.
        prompt = ChatPromptTemplate.from_template("""
        You are a date extraction assistant for stock visualization.
        Given the following query, extract the start date and end date in the format YYYY-MM-DD.
        Query: {query}
        Return your answer as a JSON object with the keys "start_date" and "end_date".
        If the query does not mention specific dates, return a default date range covering the past 6 months.
        """)
        
        # Create the chain that pipes the prompt to the LLM and then parses the output as JSON.
        chain = prompt | llm | JsonOutputParser()
        result = chain.invoke({"query": query})

        # Extract dates from the result.
        start_date = result.get("start_date")
        end_date = result.get("end_date")

        # Use default values if dates are not extracted.
        if not start_date or not end_date:
            end_date = datetime.today().strftime("%Y-%m-%d")
            start_date = (datetime.today() - timedelta(days=180)).strftime("%Y-%m-%d")

        print(f"Extracted dates: start_date = {start_date}, end_date = {end_date}")

        # Download data using the extracted (or default) dates.
        self.download_data(start_date, end_date)
        # Generate the plot and return the figure.
        fig = self.plot_closing_price()
        return fig