from langchain_openai import ChatOpenAI
from langchain_core.prompts import (
    PromptTemplate,
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    # AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
    FewShotPromptTemplate,
)

from dotenv import load_dotenv

load_dotenv()
llm = ChatOpenAI(model="gpt-4", temperature=1)


def test_message():
    result = llm.invoke(["What is the current weather in Richmond, TX?"])
    print(result.content)


def test_message2():
    messages = [
        (
            "system",
            """You are a helpful AI that helps the user make travel plans.
            Respond only in a single line."""
        ),
        (
            "human",
            "I want to go skiing.  Which city should I go to?"
        )
    ]

    response = llm.invoke(messages)
    print(response.content)


def test_template():
    prompt = PromptTemplate(
        input_variables=["city"],
        template="What is the current weather in {city}?",
    )
    chain = prompt | llm
    result = chain.invoke({"city": "Richmond, TX"})
    print(result.content)


def test_template2():
    prompt = PromptTemplate(
        input_variables=["from", "to"],
        template="What is the best way to travel from {from} to {to}?",
    )
    chain = prompt | llm
    result = chain.invoke(
        {
            "from": "Richmond, TX",
            "to": "Houston, TX"
        }
    )
    print(result.content)


def test_template3():
    system_template = SystemMessagePromptTemplate.from_template(
        """You are a helpful AI that helps the user make travel plans.
        Respond only in a single line."""
    )
    human_template = HumanMessagePromptTemplate.from_template(
        "What is the must see in {city}?"
    )
    chat_prompt = ChatPromptTemplate.from_messages(
        [system_template, human_template]
    )
    chain = chat_prompt | llm
    result = chain.invoke({"city": "Richmond, TX"})
    print(result.content)


def test_template4():
    prompt = PromptTemplate(
        input_variables=["city", "attractions"],
        template="""
            Input: {city}
            Suggestion: {attractions}
            """,
    )

    examples = [
        {
            "city": "Paris",
            "attractions": "Eiffel Tower, Louvre Museum, Notre-Dame Cathedral"
        },
        {
            "city": "New York",
            "attractions": "Statue of Liberty, Central Park, Times Square"
        },
    ]

    prefix = (
        "You are an expert travel guide."
        "Given the following input city,"
        "suggest all the attractions."
    )
    suffix = "Input: {city}\n\nSuggestion:"

    few_shot_prompt = FewShotPromptTemplate(
        example_prompt=prompt,
        examples=examples,
        prefix=prefix,
        suffix=suffix,
        input_variables=["city"],
        example_separator="\n\n",
    )

    chain = few_shot_prompt | llm
    result = chain.invoke({"city": "Tokyo"})
    print(result.content)


if __name__ == "__main__":
    # test_message()
    # test_template()
    # test_template2()
    # test_template3()
    test_template4()
