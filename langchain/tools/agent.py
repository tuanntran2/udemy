from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent

from dotenv import load_dotenv

from langchain_community.tools import (
    DuckDuckGoSearchRun,
    WikipediaQueryRun,
)
from langchain_community.utilities import WikipediaAPIWrapper
# from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory


store = {}


def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant.  For answers requiring current"
            "information, use the tools provided."
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)

load_dotenv()

wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
tools = [
    DuckDuckGoSearchRun(),
    WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper()),
]
llm = ChatOpenAI()
agent = create_agent(
    model=llm,
    tools=tools,
)

agent_with_history = RunnableWithMessageHistory(
    agent,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
)

formatted_prompt = prompt.format_messages(
    history=[],
    input=(
        "Use the provided wikipedia tool to search "
        "for Vietnamese singer, Hoa Minzy."
    ),
    agent_scratchpad=[],
)
# response = agent_with_history.invoke({"messages": formatted_prompt})
response = agent_with_history.invoke(
    {"messages": formatted_prompt},
    config={"configurable": {"session_id": "session_1"}}
)
print(response["messages"][-1].content)

# formatted_prompt = prompt.format_messages(
#     history=[],
#     input=("How old is she and where was she born?"),
#     agent_scratchpad=[],
# )
# response = agent_with_history.invoke({"messages": formatted_prompt})
response = agent_with_history.invoke(
    {"messages": "How old is she and where was she born?"},
    config={"configurable": {"session_id": "session_1"}}
)
print(response["messages"][-1].content)
