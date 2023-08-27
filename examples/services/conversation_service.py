from services.memory_factory import MemoryFactory
from services.document_base import DocumentBase
from langchain.chains import ConversationChain  
from services.enums import ModelType

class ConversationalBot(DocumentBase):
    def __init__(self, model_name=ModelType.GPT_3_5_TURBO_16K, memory_type=None):
        super().__init__(model_name)
        if memory_type:
            memory = MemoryFactory.get_memory(memory_type,self.llm)
            self.conversation_chain = ConversationChain(llm=self.llm,memory=memory)
        else:
            self.conversation_chain = ConversationChain(llm=self.llm)

    

    def chat(self, message):
        self.pretty_print(f"memory: {self.conversation_chain.memory}")
        respons = self.conversation_chain.run(message)
        self.pretty_print(respons)
        return respons
    
    def test(self):
        self.chat("Hello, my name is ilia")
        self.chat("I'm from Israel, where you are from?")
        self.chat("I'm 37, and you?")
        self.chat("Tell me a joke")
        self.chat("What is my name?")
        self.chat("Translate the entire conversation to Hebrew")

    