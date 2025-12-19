from langchain_openai import ChatOpenAI
from langchain_core.runnables.history import RunnableWithMessageHistory

from chat_message_history import ChatMessageHistorySummary

from dotenv import load_dotenv


load_dotenv()

llm = ChatOpenAI()

chat_history = ChatMessageHistorySummary()

chain = RunnableWithMessageHistory(
    runnable=llm,
    get_session_history=lambda session_id: chat_history,
    input_key="input",
    output_key="output"
)

result = chain.invoke(
    {"input": "Hello, my name is Tuan."},
    config={"configurable": {"session_id": "abc123"}}
)
print(result.content)

result = chain.invoke(
    {"input": "what is today date?"},
    config={"configurable": {"session_id": "abc123"}}
)
print(result.content)

result = chain.invoke(
    {"input": "Is it raining today?"},
    config={"configurable": {"session_id": "abc123"}}
)
print(result.content)

result = chain.invoke(
    {"input": "Do you still remember my name?"},
    config={"configurable": {"session_id": "abc123"}}
)
print(result.content)
