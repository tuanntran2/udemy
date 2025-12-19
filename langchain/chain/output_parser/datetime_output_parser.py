from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

from chain.openai_chain import OpenAIChain


class DateTimeInfo(BaseModel):
    DateTime: str = Field(description="The date and time in ISO 8601 format")


parser = PydanticOutputParser(pydantic_object=DateTimeInfo)

prompt = PromptTemplate(
    template="""
        when was {president} get elected?
        {format_instructions}
    """,
    input_variables=["president"],
    partial_variables={
        "format_instructions": parser.get_format_instructions()
    },
)

openai_chain = OpenAIChain(template=prompt, output_parser=parser)
result = openai_chain.invoke({"president": "George Washington"})
print(result)
