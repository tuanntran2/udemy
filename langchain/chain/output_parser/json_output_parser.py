from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from chain.openai_chain import OpenAIChain

parser = JsonOutputParser()

prompt = PromptTemplate(
    template="""
        Provide the meaning , an example of how the word
        is used in a sentence and the etymology of the word: {word}
        {format_instructions}
    """,
    input_variables=["word"],
    partial_variables={
        "format_instructions": parser.get_format_instructions()
    },
)

openai_chain = OpenAIChain(template=prompt, output_parser=parser)
result = openai_chain.invoke({"word": "Onomatopoeia"})
print(result)
