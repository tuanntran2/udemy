from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from dotenv import load_dotenv


load_dotenv()


def test_agent():
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant."),
        ("user", "{input}")
    ])
    llm = ChatOpenAI()
    agent = prompt | llm | StrOutputParser()
    print(agent.invoke({"input": "Multiply 12 and 7"}))


def test_agent_with_tools():
    from langchain.agents import create_agent
    from langchain_community.tools import DuckDuckGoSearchRun

    agent = create_agent(
        model=ChatOpenAI(),
        tools=[DuckDuckGoSearchRun]
    )
    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Using the provided tools, look up last night NBA "
                        "scores between the Wizards and the Pacers"
                    )
                }
            ]
        }
    )
    print(result)


def test_agent_with_tools_and_system_prompt():
    from langchain.agents import create_agent
    from langchain_core.tools import tool

    @tool
    def multiply(a: int, b: int) -> str:
        """Multiply two integers."""

        # Purposely incorrect implementation for testing
        return f"The result of the multiplication is {a + b}"

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant that can use tools."),
        ("user", "{messages}"),
    ])

    agent = create_agent(
        model=ChatOpenAI(),
        tools=[multiply]
    )

    chain = prompt | agent
    result = chain.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "there is a multiply tool attached to the agent."
                        "use that tool to calculate 6 multiply by 7."
                        "Use the result from the tool to answer the question."
                    )
                }
            ]
        }
    )
    print(result)


def test_agent_with_tool_and_memory():
    from langchain_core.runnables.history import RunnableWithMessageHistory
    from langchain_core.chat_history import InMemoryChatMessageHistory
    from langchain.agents import create_agent
    from tools.tools import test_tools

    def get_session_history(session_id: str):
        if session_id not in memory_store:
            memory_store[session_id] = InMemoryChatMessageHistory()
        return memory_store[session_id]

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant that can use tools."),
        ("placeholder", "{chat_history}"),
        ("user", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])

    memory_store = {}

    agent = create_agent(
        model=ChatOpenAI(),
        tools=test_tools,
    )

    chain = prompt | agent

    # result = chain.invoke({"input": "Hello, my name is Tuan.  Who are you?"})
    # print(result)

    # result = chain.invoke({"input": "what is the current TSLA stock price?"})
    # print(result)

    # result = chain.invoke({"input": "What is my name?"})
    # print(result)

    chain_with_memory = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
    )

    config = {"configurable": {"session_id": "abc123"}}

    result = chain_with_memory.invoke(
        {"input": "Hello, my name is Tuan.  Who are you?"},
        config=config)
    print(result)

    result = chain_with_memory.invoke(
        {"input": "what is the current TSLA stock price?"},
        config=config)
    print(result)

    result = chain_with_memory.invoke(
        {"input": "What is my name?"},
        config=config)
    print(result)


def test_llm_execute_tool():
    from langchain_core.tools import tool

    @tool
    def multiply(a: int, b: int) -> int:
        """Multiply two integers."""
        # Purposefully incorrect implementation for testing
        return a + b

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant."),
        ("user", "{input}")
    ])

    llm = ChatOpenAI()
    llm = llm.bind_tools([multiply])
    response = llm.invoke(
        prompt.format_messages(input="Multiply 12 and 7")
    )
    if response.tool_calls:
        tool_result = multiply.invoke(response.tool_calls[0]["args"])
        print(tool_result)
    else:
        print(response.content)


if __name__ == "__main__":
    # test_agent()
    # test_agent_with_tools()
    test_agent_with_tools_and_system_prompt()
    # test_llm_execute_tool()
