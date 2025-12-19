from langchain_core.tools.structured import StructuredTool
from pydantic import BaseModel, Field
# from langchain.tools import tool
import yfinance as yf


class StockArgs(BaseModel):
    ticker: str = Field(
        description="The stock ticker symbol, e.g., AAPL for Apple."
    )


class StockTool(StructuredTool):
    name: str = "stock_tool"
    description: str = "Get the latest stock price for a given ticker symbol"
    args_schema: type[BaseModel] = StockArgs

    def _run(self, ticker: str) -> str:
        stock = yf.Ticker(ticker)
        price = stock.info['regularMarketPrice']
        if price:
            return f"The latest price for {ticker.upper()} is ${price}."
        else:
            return f"Ticker symbol {ticker.upper()} not found."

# Alternatively, I can use a function-based tool:
# @tool
# def get_stock_price(ticker: str) -> str:
#     """
#     Get the latest stock price for a given ticker symbol.

#     Args:
#         ticker (str): The stock ticker symbol, e.g., AAPL for Apple.

#     Returns:
#         str: The latest stock price or an error message
#         if the ticker is not found.
#     """
#     stock = yf.Ticker(ticker)
#     price = stock.info['regularMarketPrice']
#     if price:
#         return f"The latest price for {ticker.upper()} is ${price}."
#     else:
#         return f"Ticker symbol {ticker.upper()} not found."
