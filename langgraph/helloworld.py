import time
from typing import TypedDict
from langgraph.graph import START, END, StateGraph
# from langgraph_utils import display


class HelloWorldState(TypedDict):
    message: str
    id: int


def hello(state: HelloWorldState) -> HelloWorldState:
    print(f"Entering Hello Node: {state.get('message')}")
    time.sleep(2)
    updated_state = {"message": f"Hello {state.get('message')}"}
    print(f"Exiting Hello Node: {updated_state.get('message')}")
    return updated_state


def bye(state: HelloWorldState) -> HelloWorldState:
    print(f"Entering Goodbye Node: {state.get('message')}")
    time.sleep(1)
    updated_state = {"message": f"Goodbye {state.get('message')}"}
    print(f"Exiting Goodbye Node: {updated_state.get('message')}")
    return updated_state


graph = StateGraph(HelloWorldState)
graph.add_node("hello_node", hello)
graph.add_node("bye_node", bye)
graph.set_entry_point("hello_node")

graph.add_edge(START, "hello_node")
graph.add_edge("hello_node", "bye_node")
graph.add_edge("bye_node", END)

runnable = graph.compile()
# display(runnable)
output = runnable.invoke({"id": 123, "message": "Tuan"})
print(output)
