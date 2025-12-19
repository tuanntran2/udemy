from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

from langchain_core.runnables import (
    RunnablePassthrough,
    RunnableLambda,
    RunnableParallel
)

from dotenv import load_dotenv


load_dotenv()

classifier_template = PromptTemplate(
    input_variables=["question", "answer"],
    template="""
    you are given a question and the user's answer.
    Your task is to classify whether the answer is
    correct, incorrect, or partially correct.
    Return only one of these three options:
    "correct", "incorrect", "partially correct".
    ###
    Question: {question}
    Answer: {answer}
    Classification:
    """
)

llm = ChatOpenAI()
chain = RunnableParallel(
    {
        "classification": (classifier_template | llm | StrOutputParser()),
        "input": RunnablePassthrough()
    }
)

# result = chain.invoke({
#     "question": "What are chickens?",
#     "answer": "Chicken are animals"
# })

# print(result)

correct_answer_template = PromptTemplate(
    input_variables=["question"],
    template="""
    The user answers the question correctly.
    Now ask a more difficult question on the same topic.
    ###
    Question: {question}
    ###
    New Question:
    """
)
correct_chain = correct_answer_template | llm | StrOutputParser()

incorrect_answer_template = PromptTemplate(
    input_variables=["question", "answer"],
    template="""
    The user answered the question incorrectly.
    Explain why the answer is incorrect and provide the correct answer.
    ###
    Question: {question}
    User's Answer: {answer}
    Correct Answer:
    """
)
incorrect_chain = incorrect_answer_template | llm | StrOutputParser()


def route(info):
    return (
        correct_chain if info["result"]["classification"].lower() == "correct"
        else incorrect_chain
    )


final_chain = RunnableParallel(
    {
        "result": chain,
        "question": lambda x: x["question"],
        "answer": lambda x: x["answer"]
    }
) | RunnableParallel(
    {
        "response": RunnableLambda(route),
        "input": RunnablePassthrough()
    }
)

result = final_chain.invoke({
    "question": "What are chickens?",
    "answer": "Chickens are domesticated birds raised for their meat and eggs."
})

print(result)
