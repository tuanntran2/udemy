from langchain_core.tools import tool

from langchain_community.tools import (
    DuckDuckGoSearchRun,
    WikipediaQueryRun,
)
from langchain_community.utilities import WikipediaAPIWrapper

from .weather_tool import get_weather_forecast
from .stock_tool import StockTool


@tool
def multiply(a: int, b: int) -> int:
    """Multiply two integers."""
    return a * b


test_tools = [
    multiply,
    get_weather_forecast,
    StockTool(),
    DuckDuckGoSearchRun(),
    WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
]
