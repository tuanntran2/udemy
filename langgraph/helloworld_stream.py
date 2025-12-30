from typing import TypedDict
from langgraph.graph import START, END, StateGraph
from langgraph.types import StreamWriter


class HelloWorldState(TypedDict):
    message: str
    id: int


def hello(state: HelloWorldState, writer: StreamWriter) -> HelloWorldState:
    # print(f"Hello Node: {state.get('message')}")
    writer(f"Hello node stream:  messsage: {state.get('message')}\n")
    return {"message": f"Hello {state.get('message')}"}


def bye(state: HelloWorldState) -> HelloWorldState:
    # print(f"Goodbye Node: {state.get('message')}")
    return {"message": f"Goodbye {state.get('message')}"}


graph = StateGraph(HelloWorldState)
graph.add_node("hello", hello)
graph.add_node("bye", bye)
# the following statement is not needed
# as we already have START -> hello edge
# graph.set_entry_point("hello")

graph.add_edge(START, "hello")
graph.add_edge("hello", "bye")
graph.add_edge("bye", END)

runnable = graph.compile()
for chunk in runnable.stream(
    {"message": "Tuan"},
    stream_mode="debug"
):
    print("chunk:", chunk)
