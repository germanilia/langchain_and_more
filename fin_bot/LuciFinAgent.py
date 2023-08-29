import streamlit as st
from PIL import Image

from langchain.chat_models import ChatOpenAI
from langchain.experimental.plan_and_execute import PlanAndExecute, load_agent_executor, load_chat_planner
from langchain.llms import OpenAI
from langchain import SerpAPIWrapper
from langchain.agents.tools import Tool
from lucidate.tools import StockSchema, StockTool, StockInfoTool
from typing import Optional, Type
import os
from dotenv import load_dotenv

load_dotenv()
serpapi_api_key = os.getenv("SERPAPI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Load your logo image (ensure it's in the same directory or provide the full path)
logo = Image.open('static/Colour Logo.png')

# Sidebar with a title, a logo and a slider
st.sidebar.title('Langchain Agents & Tools🦜')
st.sidebar.image(logo, use_column_width=True)
temperature = st.sidebar.slider('Select a value', min_value=0.0, max_value=1.0, value=0.2, step=0.1)

models = ["gpt-4", "gpt-4-0613", "gpt-4-32k",
          "gpt-3.5-turbo", "gpt-3.5-turbo-16k", "gpt-3.5-turbo-0613",
          "gpt-3.5-turbo-16k-0613", "text-davinci-003 (Legacy)",
          "text-davinci-002 (Legacy)", "code-davinci-002 (Legacy)"]

model = st.sidebar.selectbox("Select a GPT Model", models, index=0)

# Main window with a text box and an "Enter" button
st.title('Agent/Tools Demo 🦜🔗')
user_input = st.text_input('Your financial question?')
enter_button = st.button('Enter')

stock_tool_instance = StockInfoTool()
search = SerpAPIWrapper()
tools = [
    Tool(
        name = "Search",
        func=search.run,
        description="useful for when you need to answer questions about current events"
    ),
    Tool(
        name="stock_tool",
        func=stock_tool_instance.run,
        description="useful for when you need to get stock history and fundamentals"
    ),
]

if enter_button:
    if user_input:
        st.write(f'You entered: {user_input}, Model is: {model}, with temperature: {temperature}')
        model = ChatOpenAI(model=model, temperature=temperature)
        planner = load_chat_planner(model)
        executor = load_agent_executor(model, tools, verbose=True)
        agent = PlanAndExecute(planner=planner, executor=executor, verbose=True)
        try:
            # response = agent.run("What is the current stock price of AAPL? Is AAPL a good buy based on fundamentals at this price?")
            response = agent.run(user_input)
            st.write(response)
        except Exception as e:
            print(f"Caught an exception: {e}")
    else:
        st.write('You did not enter anything.')
