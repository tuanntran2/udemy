from langchain_openai import ChatOpenAI

from dotenv import load_dotenv

from weather_tool import get_weather_forecast
from stock_tool import StockTool


load_dotenv()

llm = ChatOpenAI()
llm_bind_tools = llm.bind_tools(
    [
        get_weather_forecast,
        StockTool()
    ]
)

result = llm_bind_tools.invoke(
    # "What is the current temperature in Richmond, TX?"
    "What is the latest stock price for Tesla?"
)
print(result)
