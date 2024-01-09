from typing import Union
from langchain.chat_models import ChatOpenAI
from langchain.chat_models import ChatAnthropic
from langchain.chat_models import AzureChatOpenAI
from pydantic import SecretStr
from services.enums import ModelType
import os


class ChatModelFactory:
    @staticmethod
    def get_llm(
        model_name=ModelType.GPT_3_5_TURBO_16K, temperature:float=0
    ) -> Union[ChatOpenAI, ChatAnthropic, AzureChatOpenAI]:
        if model_name == ModelType.GPT_3_5_TURBO_16K or model_name == ModelType.GPT_4:
            return ChatOpenAI(temperature=temperature, model_name=model_name.value)
        elif model_name == ModelType.ANTHROPIC:
            return ChatAnthropic(
                temperature=temperature,
                anthropic_api_key=os.getenv("ANTHROPIC_API_KEY","None"),
                streaming=True,
                verbose=False,
            )
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
