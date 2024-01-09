from services.chat_model_factory import ChatModelFactory
from colorama import Fore, Style
class DocumentBase:
    def __init__(self, model_name):
        self.llm = ChatModelFactory.get_llm(model_name)

    @staticmethod
    def pretty_print(string, color=Fore.WHITE):
        print("\n" + "="*100)
        print(f"{color}{string}{Style.RESET_ALL}")
        print("="*100 + "\n")


