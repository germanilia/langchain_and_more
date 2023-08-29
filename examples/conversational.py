
from services.enums import ModelType, MemoryType
from services.conversation_service import ConversationalBot


conversation = ConversationalBot(model_name=ModelType.GPT_4)
conversation.test()

conversation = ConversationalBot(model_name=ModelType.ANTHROPIC, memory_type=MemoryType.SUMMARY)
conversation.test()

conversation = ConversationalBot(model_name=ModelType.GPT_4, memory_type=MemoryType.CONVERSATION)
conversation.test()
