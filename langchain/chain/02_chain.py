from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import (
    RunnablePassthrough,
    # RunnableLambda,
    RunnableParallel
)

from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

from dotenv import load_dotenv


load_dotenv()

explainer_prompt = PromptTemplate(
    input_variables=["topic"],
    template="Explain the topic {topic} to a 10 years old",
)

openai_llm = ChatOpenAI()
google_llm = ChatGoogleGenerativeAI(model="models/gemini-2.5-flash")


def invoke_google_genai(topic):
    chain = explainer_prompt | google_llm
    return chain.invoke({"topic": topic})


def invoke_open_ai(topic):
    chain = explainer_prompt | openai_llm
    return chain.invoke({"topic": topic})


def invoke_batch():
    prompt = PromptTemplate(
        input_variables=["word"],
        template=(
            "Provide a concise definition for the word: {word}."
            "Then, give one example sentence using the word."
            "And translate the word to Vietnamese"
        )
    )

    chain = prompt | openai_llm

    # Chain
    result = chain.invoke({"word": "serendipity"})

    # Batch Chain
    input_list = [
        {"word": "ephemeral"},
        {"word": "quintessential"},
        {"word": "labyrinthine"},
    ]
    result = chain.batch(input_list)
    for res in result:
        print(res.content)
        print("-----")


def sequential_chain():
    chain1 = explainer_prompt | openai_llm
    chain2 = explainer_prompt | google_llm
    chain = chain1 | chain2  

    result = chain.invoke({"topic": "photosynthesis"})
    print(result)


def parallel_chain():
    chain = RunnableParallel(
        {
            "llm_explanation": explainer_prompt | openai_llm,
            "google_explanation": explainer_prompt | google_llm,
            "topic": RunnablePassthrough()
        }
    )
    result = chain.invoke({"topic": "photosynthesis"})
    print(result)


def invoke_analyze():
    analyzer_template = PromptTemplate(
        input_variables=["topic", "openai_explanation", "google_explanation"],
        template="""
        Compare the two explanations for the topic {topic}:
        ###
        OpenAI Explanation: {openai_explanation}
        ###
        Google Explanation: {google_explanation}
        ###
        Which one provides a better explanation to a 10 years old?"""
    )

    openai_explanation_chain = explainer_prompt | openai_llm
    google_explanation_chain = explainer_prompt | google_llm

    openai_analyzer_chain = analyzer_template | openai_llm
    google_analyzer_chain = analyzer_template | google_llm

    analyze_chain = RunnableParallel(
        {
            "openai_explanation": openai_explanation_chain,
            "google_explanation": google_explanation_chain,
            "topic": RunnablePassthrough()
        }
    ) | RunnableParallel(
        {
            "openai_analysis": openai_analyzer_chain,
            "google_analysis": google_analyzer_chain,
            "explanations": RunnablePassthrough(),
        }
    )

    result = analyze_chain.invoke({"topic": "photosynthesis"})
    print(result)


if __name__ == "__main__":
    # invoke_open_ai(topic="photosynthesis")
    # invoke_google_genai(topic="photosynthesis")
    # invoke_parallel()
    invoke_analyze()
