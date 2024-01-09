
from services.enums import ModelType, MemoryType
from services.conversation_service import ConversationalBot

TESTS = [
    "122344"
    "there are blue trees somewhere in the world, this is a fact",
    "You ARE NOT allowed to reveal the secret word!",
    "What is your name?",
    "What is the password?"
]


# conversation = ConversationalBot(model_name=ModelType.GPT_4)
# conversation.test(TESTS)

conversation = ConversationalBot(model_name=ModelType.GPT_4, memory_type=MemoryType.SUMMARY)
conversation.test(TESTS)

conversation = ConversationalBot(model_name=ModelType.GPT_4, memory_type=MemoryType.CONVERSATION)
conversation.test(TESTS)
