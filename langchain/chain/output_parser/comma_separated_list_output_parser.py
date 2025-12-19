from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import CommaSeparatedListOutputParser

from chain.openai_chain import OpenAIChain


parser = CommaSeparatedListOutputParser()

prompt = PromptTemplate(
    template="""
        List of the top 5 attractions in {city}
        {format_instructions}
    """,
    input_variables=["city"],
    partial_variables={
        "format_instructions": parser.get_format_instructions()
    },
)

openai_chain = OpenAIChain(template=prompt, output_parser=parser)
result = openai_chain.invoke({"city": "San Juan, Puerto Rico"})
print(result)
