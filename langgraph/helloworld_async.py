import asyncio
from operator import add
from typing import Annotated, TypedDict
from langgraph.graph import START, END, StateGraph
# from langgraph_utils import display


class HelloWorldState(TypedDict):
    message: Annotated[str, add]
    id: Annotated[int, add]


async def hello(state: HelloWorldState) -> HelloWorldState:
    print(f"Entering Hello Node: {state.get('message')}")
    await asyncio.sleep(1)
    updated_state = {"message": f"Hello {state.get('message')}"}
    print(f"Exiting Hello Node: {updated_state.get('message')}")
    return updated_state


async def bye(state: HelloWorldState) -> HelloWorldState:
    print(f"Entering Goodbye Node: {state.get('message')}")
    await asyncio.sleep(2)
    updated_state = {"message": f"Goodbye {state.get('message')}"}
    print(f"Exiting Goodbye Node: {updated_state.get('message')}")
    return updated_state


graph = StateGraph(HelloWorldState)
graph.add_node("hello_node", hello)
graph.add_node("bye_node", bye)
graph.set_entry_point("hello_node")

graph.add_edge(START, "hello_node")
graph.add_edge(START, "bye_node")
graph.add_edge("hello_node", END)
graph.add_edge("bye_node", END)


async def main():
    runnable = graph.compile()
    # display(runnable)

    output = await runnable.ainvoke({"id": 123, "message": "Tuan"})
    print(output)


if __name__ == "__main__":
    asyncio.run(main())
