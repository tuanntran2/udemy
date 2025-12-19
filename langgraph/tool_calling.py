from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, END, StateGraph, MessagesState
from langgraph_utils import display
from langgraph.prebuilt import ToolNode
from langchain_community.tools import DuckDuckGoSearchRun

from dotenv import load_dotenv

load_dotenv()


@tool
def lookup_phone(name: str) -> str:
    """Looks up a phone number by name."""
    phone_numbers = {
        "Alice": "123-456-7890",
        "Bob": "234-567-8901",
        "Charlie": "345-678-9012",
        "Elon Musk": "777-777-7777",
        "Jeff Bezos": "888-888-8888"
    }
    return phone_numbers.get(name, "Not found")


tools = [
    DuckDuckGoSearchRun(),
    lookup_phone
]

messages = [
    HumanMessage(content=("What is Alice's phone number?"))
]

llm = ChatOpenAI()


def test_tool_calling_with_llm():
    llm_with_tools = llm.bind_tools(tools=tools)
    llm_output = llm_with_tools.invoke(messages)
    print(llm_output)


# This test does not work
def test_tool_calling_with_tool_node():
    """
    Docstring for test_tool_calling_with_tool_node
    Remarks: This function is implemented as instructed by the Udemy course
    But it's not working.
    """
    tool_node = ToolNode(tools)

    message_with_tool_call = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "lookup_phone",
                "args": {
                    "name": "Alice"
                },
                "id": "tool_call_1",
            }
        ]
    )

    result = tool_node.invoke(
        {"messages": [message_with_tool_call]},
        config={"configurable": {"thread_id": "1"}}
    )
    print(result)


def test_tool_with_workflow():
    llm_with_tools = llm.bind_tools(tools=tools)
    tool_node = ToolNode(tools)

    def agent_node(state: MessagesState):
        response = llm_with_tools.invoke(state["messages"])
        return {"messages": response}

    def should_continue(state: MessagesState):
        last_message = state["messages"][-1]
        return "tools" if last_message.tool_calls else END

    workflow = StateGraph(MessagesState)
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tool_node)

    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        ["tools", END]
        # {
        #     "tools": "tools",
        #     END: END
        # }
    )
    workflow.add_edge("tools", "agent")

    enable_memory = True
    graph = (
        workflow.compile(checkpointer=MemorySaver())
        if enable_memory
        else workflow.compile()
    )
    display(graph)

    # the "thread_id" will be used as a session id for memory
    config = {"configurable": {"thread_id": "1"}}
    messages = [
        HumanMessage(
            content=(
                (
                    "Find the richest person in the world "
                    "and look up their phone number using the tools."
                )
            )
        )
    ]
    response = graph.invoke(
        {"messages": messages},
        config=config
    )
    print(response["messages"][-1].content)

    # Let's test on memory recall
    messages = [HumanMessage(content="How old is he?")]
    response = graph.invoke(
        {"messages": messages},
        config=config
    )
    print(response["messages"][-1].content)


if __name__ == "__main__":
    # test_tool_calling_with_llm(),
    # test_tool_calling_with_tool_node()
    test_tool_with_workflow()
