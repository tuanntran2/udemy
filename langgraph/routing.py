from typing import TypedDict
from langgraph.graph import START, END, StateGraph
from langgraph_utils import display


class SupportRequest(TypedDict):
    message: str
    priority: int


def categorize_request(request: SupportRequest) -> str:
    if "urgent" in request["message"].lower() or request["priority"] == 1:
        return "urgent"

    return "standard"


def handle_urgent(request: SupportRequest) -> str:
    print(f"Route to Urgent Support Team: {request}")
    return request


def handle_standard(request: SupportRequest) -> str:
    print(f"Route to Standard Support Team: {request}")
    return request


graph = StateGraph(SupportRequest)
graph.add_node("urgent", handle_urgent)
graph.add_node("standard", handle_standard)

graph.add_conditional_edges(
    START,
    categorize_request,
    ["urgent", "standard"]
)

graph.add_edge("urgent", END)
graph.add_edge("standard", END)

runnable = graph.compile()
display(runnable)

print(runnable.invoke(
    {"message": "My internet is down!", "priority": 1}
))
print(runnable.invoke(
    {"message": "I have a question about my bill.", "priority": 3}
))
print(runnable.invoke(
    {"message": "This is an urgent issue!", "priority": 5}
))
print(runnable.invoke(
    {"message": "Just a regular inquiry.", "priority": 4}
))
print(runnable.invoke(
    {"message": "Need help ASAP!", "priority": 2}
))
print(runnable.invoke(
    {"message": "No rush, just checking in.", "priority": 5}
))
print(runnable.invoke(
    {"message": "URGENT: System failure!", "priority": 1}
))
print(runnable.invoke(
    {"message": "General feedback.", "priority": 4}
))
print(runnable.invoke(
    {"message": "Please address this urgent matter.", "priority": 2}
))
print(runnable.invoke(
    {"message": "Inquiry about services.", "priority": 3}
))
