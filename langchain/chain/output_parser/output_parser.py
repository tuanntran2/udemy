from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from chain.openai_chain import OpenAIChain

parser = StrOutputParser()

prompt = PromptTemplate(
    input_variables=["word"],
    template="""
    Provide the meaning , an example of how the word
    is used in a sentence and the etymology of the word: {word}
    """
)

chain = OpenAIChain(template=prompt, output_parser=parser)
result = chain.invoke({"word": "Onomatopoeia"})
print(result)
