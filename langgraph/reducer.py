from typing import Annotated
# from langchain_core.messages import AnyMessage
from operator import add
from langgraph.graph import END, START, StateGraph, MessagesState
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph_utils import display


# Define chatbot state with accumulated messages
# class ChatBotState(MessagesState):
#     discount: Annotated[int, add]
ChatBotState = MessagesState


# Responses based on intent level
def connect_to_sales(state: ChatBotState):
    return {
        "messages": [
            AIMessage(content="Great! Let me connect you with our sales team.")
        ],
        # "discount": 10
    }


def sales_response(state: ChatBotState):
    return {
        "messages": [
            AIMessage(content="We have the best offer for you")
        ],
        # "discount": 20
    }


# Build chatbot conversation flow
graph_builder = StateGraph(ChatBotState)

# Add nodes

graph_builder.add_node("connect_to_sales", connect_to_sales)
graph_builder.add_node("sales_response", sales_response)

# Define conversation flow
graph_builder.add_edge(START, "connect_to_sales")
graph_builder.add_edge("connect_to_sales", "sales_response")
graph_builder.add_edge("sales_response", END)

# Compile chatbot
chatbot = graph_builder.compile()
display(chatbot)

# Simulate different conversations
test_inputs = "I want to buy your product."


messages = chatbot.invoke({"messages": [HumanMessage(content=test_inputs)]})

for message in messages["messages"]:
    print(f"🤖 **Bot:** {message.content}")


# print("Final Discount: ", messages['discount'], '%')
