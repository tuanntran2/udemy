from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv


class OpenAIChain:
    def __init__(
        self,
        template: PromptTemplate = None,
        output_parser: StrOutputParser = None,
    ):
        load_dotenv()
        self._llm = ChatOpenAI()
        self._template = template

        self._chain = self._template | self._llm \
            if self._template else None

        if self._chain and output_parser:
            self._chain = self._chain | output_parser

    def invoke(self, input: dict):
        return self._chain.invoke(input)
