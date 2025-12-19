from langchain_core.prompts import PromptTemplate

import tkinter as tk
from tkinter import simpledialog

ROOT = tk.Tk()
ROOT.withdraw()

destination_city = simpledialog.askstring(
    title="Travel Planner",
    prompt="Enter your destination city:"
)

departure_city = simpledialog.askstring(
    title="Travel Planner",
    prompt="Enter your departure city:"
)

template = """
You are an expert travel guide. Given the following input city,
suggest the top 5 attractions.

Input: {destination_city}

Rank the attractions from most popular to least popular.
And the transportation to get there from {departure_city}.

Return the result in the language used in the {departure_city}.
"""

prompt = PromptTemplate(
    input_variables=["destination_city", "departure_city"],
    template=template,
)
prompt = prompt.format(
    destination_city=destination_city,
    departure_city=departure_city,
)

print(prompt)
