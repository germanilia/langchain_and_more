from langchain.chat_models import ChatOpenAI
from langchain.chat_models import ChatAnthropic
from langchain.chat_models import AzureChatOpenAI
from services.enums import ModelType
import os

class ChatModelFactory:
    @staticmethod
    def get_llm(model_name=ModelType.GPT_3_5_TURBO_16K,temperature=0):
        if model_name==ModelType.GPT_3_5_TURBO_16K or model_name==ModelType.GPT_4:
            return ChatOpenAI(temperature=temperature, model_name=model_name.value)
        elif model_name==ModelType.ANTHROPIC:
            return ChatAnthropic(
                                temperature=temperature,
                                anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
                                streaming=True,
                                verbose=False
                                )
        elif model_name==ModelType.AZURE:
            return AzureChatOpenAI(
                    deployment_name=os.getenv("AZURE_DEPLOYMENT_NAME"),
                    engine=os.getenv("AZURE_DEPLOYMENT_NAME"),
                    openai_api_base=os.getenv("AZURE_ENDPOINT_NAME"),
                    openai_api_key=os.getenv("AZURE_API_KEY"),
                    model_name=os.getenv("AZURE_MODEL_NAME"),
                    openai_api_version=os.getenv("AZURE_API_VERSION"),
                    temperature=temperature,
                    # stop=["Human:", "Chatbot:"],
                )
