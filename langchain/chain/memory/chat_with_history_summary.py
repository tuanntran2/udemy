from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.prompts import ChatPromptTemplate

# 1. Base model
llm = ChatOpenAI()

# 2. Prompt for summarization
summary_prompt = ChatPromptTemplate.from_messages([
    ("system", "Summarize the following conversation history:"),
    ("human", "{history}")
])

summary_chain = summary_prompt | llm

# 3. Message history backend
history_store = InMemoryChatMessageHistory()

# 4. Wrap your runnable with message history
chat_with_history = RunnableWithMessageHistory(
    runnable=llm,
    get_message_history=lambda session_id: history_store,
)

# 5. Use it in a session
config = {"configurable": {"session_id": "abc123"}}

# Add some conversation
chat_with_history.invoke([HumanMessage(content="Hello!")], config=config)
chat_with_history.invoke([HumanMessage(content="Can you explain LangChain memory?")], config=config)

# 6. Summarize history when needed
summary = summary_chain.invoke({"history": history_store.messages})
print("Conversation summary:", summary.content)