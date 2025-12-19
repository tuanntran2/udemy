import asyncio
from langchain.tools import tool
import python_weather as weather


@tool
def get_weather_forecast(city: str) -> str:
    """
    Retrieve the current weather forecast for a given city

    Args:
        city (str): The city name (e.g., "Tokyo").

    Returns:
        str: A summary of the forecast.
    """
    async def fetch_weather():
        try:
            async with weather.Client(unit=weather.IMPERIAL) as client:
                weather_data = await client.get(city)

                # Current conditions
                current_temp = weather_data.temperature
                current_desc = weather_data.description

                # Next forecast day
                forecast_day = weather_data.daily_forecasts[0]
                forecast_temp = forecast_day.temperature

                return (
                    f"Weather in {city}:\n"
                    f"\tCurrent: {current_temp}°F.  {current_desc}\n"
                    f"\tForecast: {forecast_temp}°F.\n"
                )
        except Exception as e:
            return f"Error retrieving forecast for {city}: {e}"

    return asyncio.run(fetch_weather())
