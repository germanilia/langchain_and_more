import re

import streamlit as st
from PIL import Image

from langchain.chat_models import ChatOpenAI
from langchain.experimental.plan_and_execute import PlanAndExecute, load_agent_executor, load_chat_planner
from langchain.llms import OpenAI
from langchain import SerpAPIWrapper
from langchain.agents.tools import Tool
from lucidate.robustTools import (StockSchema,
                                  StockTool,
                                  NewsTool,
                                  NewsSchema,
                                  get_company_news,
                                  StockInfoTool,
                                  _handle_error,
                                  LuciMessageCollector)
from typing import Optional, Type
import os
from dotenv import load_dotenv



load_dotenv()
news_api_key = os.getenv("NEWS_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Load your logo image (ensure it's in the same directory or provide the full path)
logo = Image.open('static/Colour Logo.png')
bears = Image.open('static/bears.png')

# Sidebar with a title, a logo and a slider
st.sidebar.title('Langchain Agents & Tools🦜')
st.sidebar.image(logo, use_column_width=True)
temperature = st.sidebar.slider('Select a value', min_value=0.0, max_value=1.0, value=0.2, step=0.1)

models = ["gpt-4", "gpt-4-0613", "gpt-4-32k",
          "gpt-3.5-turbo", "gpt-3.5-turbo-16k", "gpt-3.5-turbo-0613",
          "gpt-3.5-turbo-16k-0613", "text-davinci-003 (Legacy)",
          "text-davinci-002 (Legacy)", "code-davinci-002 (Legacy)"]

model = st.sidebar.selectbox("Select a GPT Model", models, index=0)
detail = st.sidebar.radio('Detail', ['Show', 'Hide'], horizontal=True)
info = st.sidebar.radio('Info', ['Show', 'Hide'], horizontal=True)
thoughts = st.sidebar.radio('Thoughts', ['Show', 'Hide'], horizontal=True)
reflections = st.sidebar.radio('Reflections', ['Show', 'Hide'], horizontal=True)

# Main window with a text box and an "Enter" button
st.image(bears, use_column_width=True)
st.title("Lucidate's TBIC AI FinBot")
user_input = st.text_input('Your financial question?')
enter_button = st.button('Enter')

stock_tool_instance = StockTool()
stock_info_tool_instance = StockInfoTool()
# multi = stock_recommendation()
news = NewsTool()
tools = [
    Tool(
        name = "News",
        func=news.run,
        description="useful for when you need to answer questions about recent news",
        handle_tool_error = _handle_error,
    ),
    Tool(
        name="stock_info_tool",
        func=stock_info_tool_instance.run,
        description="useful for when you need to get stock fundamentals and ratios",
        handle_tool_error = _handle_error,
    ),
    Tool(
        name="stock_tool",
        func=stock_tool_instance.run,
        description="useful for when you need to get stock price and stock price history",
        handle_tool_error = _handle_error,
    ),



]

collector = None
if enter_button:
    if user_input:
        st.write(f'You entered: {user_input}, Model is: {model}, with temperature: {temperature}')
        model = ChatOpenAI(model=model, temperature=temperature)
        planner = load_chat_planner(model)
        executor = load_agent_executor(model, tools, verbose=True)
        collector = LuciMessageCollector()
        agent = PlanAndExecute(planner=planner, executor=executor, verbose=True)
        try:
            with collector:
                response = agent.run(user_input)
            # print(collector.messages)
            st.title("Summary:")
            st.write(response)
            if detail == 'Show':
                st.title("Detail:")
                for text in collector.messages["Actions"]:
                    if len(text) > 35:
                        td = re.sub(r'\$', r'\$', text)
                        t = td.split("\n")
                        st.markdown("\n".join(t))
            if info == 'Show':
                st.title("Info. based on the following plan:")
                n = 1
                for step in collector.messages["Steps"]:
                    st.write(f"{n}: {step.split('Step: ')[1]}")
                    n+=1
            if thoughts == 'Show':
                st.title("Some key thoughts and observations:")
                for text in collector.messages["Thoughts"]:
                    if len(text) > 35:
                        td = re.sub(r'\$', r'\$', text)
                        t = td[21:]
                        st.write(t)

            if reflections == 'Show':
                st.title("Some reflections...")
                for text in list(set(collector.messages["Responses"])):
                    if len(text) > 135:
                        td = re.sub(r'\$', r'\$', text)
                        t = td[9:]
                        st.write(t)




        except Exception as e:
            print(f"Caught an exception: {e}")
    else:
        st.write('You did not enter anything.')



