from colorama import Fore
from services.memory_factory import MemoryFactory
from services.document_base import DocumentBase
from langchain.chains import ConversationChain
from services.enums import ModelType


class ConversationalBot(DocumentBase):
    def __init__(
        self, model_name=ModelType.GPT_3_5_TURBO_16K, memory_type=None
    ) -> None:
        super().__init__(model_name)
        if memory_type:
            memory = MemoryFactory.get_memory(memory_type, self.llm)
            if memory:
                self.conversation_chain = ConversationChain(llm=self.llm, memory=memory)
            else:
                self.conversation_chain = ConversationChain(llm=self.llm)
        else:
            self.conversation_chain = ConversationChain(llm=self.llm)

    def chat(self, message):
        DocumentBase.pretty_print(f"memory: {self.conversation_chain.memory}")
        response = self.conversation_chain.run(message)
        DocumentBase.pretty_print(response, Fore.RED)
        return response

    def test(self, tests: list[str]):
        for test in tests:
            self.chat(test)
