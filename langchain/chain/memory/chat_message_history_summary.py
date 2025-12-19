from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

from dotenv import load_dotenv


load_dotenv()

llm = ChatOpenAI()
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You chat with a human"),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ]
)

chain = prompt | llm | StrOutputParser()

store = {}


def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]


runnable = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)

result = runnable.invoke(
    {"input": "What is the capital of Puerto Rico?"},
    config={"configurable": {"session_id": "session_1"}}
)
print(result)

result = runnable.invoke(
    {"input": "What is its population?"},
    config={"configurable": {"session_id": "session_2"}}
)
print(result)

result = runnable.invoke(
    {"input": "What language is spoken there?"},
    config={"configurable": {"session_id": "session_1"}}
)
print(result)

result = runnable.invoke(
    {"input": "Where is this?"},
    config={"configurable": {"session_id": "session_1"}}
)
print(result)
