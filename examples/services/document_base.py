from services.chat_model_factory import ChatModelFactory
class DocumentBase:
    def __init__(self, model_name):
        self.llm = ChatModelFactory.get_llm(model_name)

    @staticmethod
    def pretty_print(string):
        print("\n" + "="*100)
        print(string)
        print("="*100 + "\n")   
