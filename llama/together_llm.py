from dotenv import load_dotenv
load_dotenv('llama/.env')
import together
import os
from typing import Any, Dict

from pydantic import Extra, root_validator

from langchain.llms.base import LLM
from langchain.utils import get_from_dict_or_env
class TogetherLLM(LLM):
    """Together large language models."""

    model: str = "togethercomputer/llama-2-70b-chat"
    """model endpoint to use"""

    together_api_key: str = os.getenv("TOGETHER_API_KEY")
    """Together API key"""

    temperature: float = 0.7
    """What sampling temperature to use."""

    max_tokens: int = 512
    """The maximum number of tokens to generate in the completion."""

    class Config:
        extra = Extra.forbid

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that the API key is set."""
        api_key = get_from_dict_or_env(
            values, "together_api_key", "TOGETHER_API_KEY"
        )
        values["together_api_key"] = api_key
        return values

    @property
    def _llm_type(self) -> str:
        """Return type of LLM."""
        return "together"

    def _call(
        self,
        prompt: str,
        **kwargs: Any,
    ) -> str:
        """Call to Together endpoint."""
        together.api_key = self.together_api_key
        output = together.Complete.create(prompt,
                                          model=self.model,
                                          max_tokens=self.max_tokens,
                                          temperature=self.temperature,
                                          )
        text = output['output']['choices'][0]['text']
        return text
