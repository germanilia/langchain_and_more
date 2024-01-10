from autogen import AssistantAgent, UserProxyAgent
import os
from dotenv import load_dotenv

load_dotenv(".env")

config_list = [
    {
        "model": "gpt-4-32k",
        "api_key": os.getenv("AZURE_API_KEY"),
        "base_url": "https://ilia-gpt.openai.azure.com/",
        "api_type": "azure",
        "api_version": "2023-08-01-preview",
    },
    {"model": "gpt-4-1106-preview", "api_key": os.getenv("OPENAI_API_KEY")},
    {"model": "gpt-3.5-turbo-1106", "api_key": os.getenv("OPENAI_API_KEY")},
]


llm_config = {"seed": 45, "temperature": 0.4, "config_list": config_list}

coder = AssistantAgent(
    name="coder",
    code_execution_config={
        "work_dir": "_output/groupchat",
        "use_docker": "python-pdf:latest",
    },
    llm_config=llm_config,
)
user_proxy = UserProxyAgent(
    name="user_proxy",
    code_execution_config={
        "work_dir": "_output/groupchat",
        "use_docker": "python-pdf:latest",
    },
    llm_config=llm_config,
    human_input_mode="TERMINATE",
    max_consecutive_auto_reply=10,
    is_termination_msg=lambda x: x.get("content", "")
    and x.get("content", "").rstrip().endswith("TERMINATE"),
)

task = """
Build a snake game in python
"""

user_proxy.initiate_chat(
    coder,
    message=task,
)
