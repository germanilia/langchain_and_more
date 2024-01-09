from langchain.memory import ConversationBufferWindowMemory
from langchain.memory import ConversationSummaryBufferMemory

from services.enums import MemoryType
MEMORY_LIMIT=10



class MemoryFactory:

    @staticmethod
    def get_memory(memory_type:str,llm) -> ConversationSummaryBufferMemory | ConversationBufferWindowMemory | None:
        if memory_type==MemoryType.SUMMARY:
            return ConversationSummaryBufferMemory(llm=llm, max_token_limit=MEMORY_LIMIT)
        elif memory_type==MemoryType.CONVERSATION:
            return ConversationBufferWindowMemory(k=MEMORY_LIMIT)
        return None

    @staticmethod
    def add_message_to_memory(user,ai,memory):
        memory.save_context({"input": user}, {"output": ai})
        return memory      
    