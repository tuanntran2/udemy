from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_openai import ChatOpenAI


class ChatMessageHistory(InMemoryChatMessageHistory):
    def __init__(self, k: int = -1):
        super().__init__()
        self._k = k

    def add_message(self, message):
        super().add_message(message)
        if self._k == -1:
            return

        if len(self.messages) > self._k:
            self.messages = self.messages[-self._k:]


class ChatMessageHistorySummary(InMemoryChatMessageHistory):
    def __init__(self):
        super().__init__()
        self._llm = ChatOpenAI()

    async def add_message(self, message):
        if len(self.messages) > 1:
            summary = await self._llm.invoke(
                {"input": f"Summarize the following messages: {self.messages}"}
            )
            self.messages = [summary]
            await super().add_message(
                {"role": "system", "content": f"Summary: {summary}"}
            )

        await super().add_message(message)
        print(self.messages)
