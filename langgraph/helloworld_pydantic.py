from typing import Optional
from pydantic import BaseModel, Field
from langgraph.graph import START, END, StateGraph
from langgraph_utils import display


class HelloWorldState(BaseModel):
    message: str = Field(min_length=3, max_length=10)
    id: Optional[int] = None


def hello(state: HelloWorldState) -> HelloWorldState:
    print(f"Hello Node: {state.message}")
    return {"message": f"Hello {state.message}"}


def bye(state: HelloWorldState) -> HelloWorldState:
    print(f"Goodbye Node: {state.message}")
    return {"message": f"Goodbye {state.message}"}


graph = StateGraph(HelloWorldState)
graph.add_node("hello", hello)
graph.add_node("bye", bye)
graph.set_entry_point("hello")

graph.add_edge(START, "hello")
graph.add_edge("hello", "bye")
graph.add_edge("bye", END)

runnable = graph.compile()
display(runnable)
output = runnable.invoke({"message": "Tuan"})
print(output)
