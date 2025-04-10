import yfinance as yf
import matplotlib.pyplot as plt

class StockVisualizer:
    def __init__(self, ticker='AAPL'):
        """
        Initializes the visualizer for a specific stock ticker.
        
        Parameters:
        - ticker (str): The stock symbol (default: 'AAPL' for Apple)
        """
        self.ticker = ticker
        self.data = None

    def download_data(self, start_date, end_date):
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
        Plots the closing price of the stock.
        """
        if self.data is None:
            raise ValueError("No data to plot. Please call download_data() first.")
        
        plt.figure(figsize=(12, 6))
        plt.plot(self.data.index, self.data['Close'], label=f'{self.ticker} Closing Price', linewidth=2)
        plt.title(f"{self.ticker} Stock Closing Price")
        plt.xlabel("Date")
        plt.ylabel("Price (USD)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

