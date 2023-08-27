from langchain.chat_models import ChatOpenAI
from langchain.chat_models import ChatAnthropic
from langchain.chat_models import AzureChatOpenAI
import os
class ChatModelFactory:
    @staticmethod
    def get_llm(model_name="gpt-3.5-turbo-16k"):
        if model_name=="gpt-3.5-turbo-16k" or model_name=="gpt-4":
            return ChatOpenAI(temperature=0, model_name=model_name)
        elif model_name=="anthropic":
            return ChatAnthropic(
                                temperature=0,
                                anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
                                streaming=True,
                                verbose=False
                                )
        elif model_name=="azure":
            return AzureChatOpenAI(
                    deployment_name=os.getenv("AZURE_DEPLOYMENT_NAME"),
                    engine=os.getenv("AZURE_DEPLOYMENT_NAME"),
                    openai_api_base=os.getenv("AZURE_ENDPOINT_NAME"),
                    openai_api_key=os.getenv("AZURE_API_KEY"),
                    model_name=os.getenv("AZURE_MODEL_NAME"),
                    openai_api_version=os.getenv("AZURE_API_VERSION"),
                    temperature=0,
                    # stop=["Human:", "Chatbot:"],
                )
