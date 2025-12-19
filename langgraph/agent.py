from typing import TypedDict
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph

from dotenv import load_dotenv


load_dotenv()


# 5. Build the graph
class AgentState(TypedDict):
    messages: list


# 1. Define a simple tool
@tool
def add_numbers(a: int, b: int) -> int:
    """Add two numbers together."""

    # Intentional mistake for testing purpose
    return a * b


# 2. Initialize the model
model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# 3. Create the agent with the tool
# agent = create_openai_functions_agent(model, [add_numbers])
agent = create_agent(model, [add_numbers])

# 4. Add memory (optional but useful)
graph = StateGraph(AgentState)
graph.add_node("agent", agent)
graph.set_entry_point("agent")

memory = MemorySaver()
app = graph.compile(checkpointer=memory)

result = app.invoke(
    {
        "messages": [
            (
                "user",
                "Using the provided tool to calculate the sum of 3 and 5"
            )
        ]
    },
    configurable={
        "thread_id": "demo-thread",
        "checkpoint_ns": "default",
        "checkpoint_id": "run-1"
    }
)
print(result)
