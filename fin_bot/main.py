from dataclasses import dataclass
from typing import Literal
import streamlit as st
import streamlit.components.v1 as components
from langchain import OpenAI, ConversationChain, LLMChain, PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts.chat import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, \
    MessagesPlaceholder
from dotenv import load_dotenv
import os
from langchain.callbacks import get_openai_callback
from utils.get_sp500 import scrape_sp500, get_ticker_info, get_sp500_tickers_from_text, format_portfolio, \
    get_most_common_ticker, check_string
import pandas as pd
import plotly.express as px
import plotly.io as pio
from abc import ABC, abstractmethod
import yfinance as yf
from bokeh.palettes import Category20
from plotly.subplots import make_subplots
import spacy
import plotly.graph_objects as go
import plotly.subplots as sp
from PIL import Image

nlp = spacy.load("en_core_web_sm")
pio.templates.default = "plotly"


class ChartStrategy(ABC):
    @abstractmethod
    def execute(self, df, portfolio, merged_portfolio, uploaded_file, ticker):
        pass


class DataStrategy(ChartStrategy):
    def execute(self, df, portfolio, merged_portfolio, uploaded_file, ticker):
        if ticker:
            fundamental_data = yf.Ticker(ticker).info

            market_cap = fundamental_data.get('marketCap', None)
            if market_cap is not None:
                market_cap = market_cap / 1e9  # Convert to billions
                market_cap_formatted = f"${market_cap:.2f}B"
            else:
                market_cap_formatted = None
            gross_profit = fundamental_data.get('grossProfits', None)
            if gross_profit is not None:
                gross_profit = gross_profit / 1e6  # Convert to billions
                gross_profit_formatted = f"${gross_profit:.2f}M"
            else:
                gross_profit_formatted = None

            revenue = fundamental_data.get('totalRevenue', None)
            if revenue is not None:
                revenue = revenue / 1e6  # Convert to billions
                revenue_formatted = f"${revenue:.2f}M"
            else:
                gross_profit_formatted = None

            pe_ratio = fundamental_data.get('trailingPE', None)
            eps = fundamental_data.get('trailingEps', None)
            dividend_yield = fundamental_data.get('dividendYield', None)
            roe = fundamental_data.get('returnOnEquity', None)
            debt_to_equity = fundamental_data.get('debtToEquity', None)
            current_ratio = fundamental_data.get('currentRatio', None)
            quick_ratio = fundamental_data.get('quickRatio', None)
            forward_pe_ratio = fundamental_data.get('forwardPE', None)
            peg_ratio = fundamental_data.get('pegRatio', None)
            price_to_book = fundamental_data.get('priceToBook', None)
            price_to_sales = fundamental_data.get('priceToSalesTrailing12Months', None)
            dividend_rate = fundamental_data.get('dividendRate', None)
            dividend_payout_ratio = fundamental_data.get('payoutRatio', None)
            beta = fundamental_data.get('beta', None)
            fifty_two_week_high = fundamental_data.get('fiftyTwoWeekHigh', None)
            fifty_two_week_low = fundamental_data.get('fiftyTwoWeekLow', None)
            two_hundred_day_ma = fundamental_data.get('twoHundredDayAverage', None)
            fifty_day_ma = fundamental_data.get('fiftyDayAverage', None)
            gross_profit = fundamental_data.get('grossProfits', None)
            operating_margin = fundamental_data.get('operatingMargins', None)
            net_profit_margin = fundamental_data.get('profitMargins', None)

            table_data = {
                'Metric': ['Ticker', 'Company Name', 'Sector', 'Industry', 'Market Cap', 'P/E Ratio', 'EPS',
                           'Dividend Yield', 'ROE', 'Debt-to-Equity Ratio', 'Current Ratio', 'Quick Ratio',
                           'Forward P/E Ratio', 'PEG Ratio', 'Price/Book Ratio', 'Price/Sales Ratio',
                           'Dividend Rate', 'Dividend Payout Ratio', 'Beta', '52-Week High', '52-Week Low',
                           '200-Day MA', '50-Day MA', 'Revenue (ttm)', 'Gross Profit (ttm)',
                           'Operating Margin (ttm)', 'Net Profit Margin (ttm)'],
                'Value': [fundamental_data.get('symbol', None), fundamental_data.get('longName', None),
                          fundamental_data.get('sector', None), fundamental_data.get('industry', None),
                          market_cap_formatted, pe_ratio, eps, dividend_yield, roe, debt_to_equity,
                          current_ratio, quick_ratio, forward_pe_ratio, peg_ratio, price_to_book,
                          price_to_sales, dividend_rate, dividend_payout_ratio, beta, fifty_two_week_high,
                          fifty_two_week_low, two_hundred_day_ma, fifty_day_ma, revenue_formatted,
                          gross_profit_formatted,
                          operating_margin, net_profit_margin]
            }

            table_df = pd.DataFrame(table_data)
            table_html = table_df.to_html(index=False)

            return table_html
        else:
            return None


class StarburstStrategy(ChartStrategy):
    def execute(self, df, portfolio, merged_portfolio, uploaded_file, ticker):
        unique_sectors = list(set(df['Sector'].unique()).union(set(merged_portfolio['Sector'].unique())))

        # Ensure there are no more unique sectors than colors in the palette
        assert len(unique_sectors) <= 20

        # Create a consistent color mapping dictionary
        color_dict = {sector: color for sector, color in zip(unique_sectors, Category20[20])}

        df['MarketCapWeight'] = df['MarketCap'] / df['MarketCap'].sum()

        merged_portfolio['MarketCapWeight'] = merged_portfolio['Value ($)'] / merged_portfolio['Value ($)'].sum()

        fig1 = px.sunburst(df, path=['Sector', 'SubIndustry', 'Ticker'], color='Sector', color_discrete_map=color_dict,
                           values='MarketCapWeight')
        fig2 = px.sunburst(merged_portfolio, path=['Sector', 'SubIndustry', 'Ticker'], color='Sector',
                           color_discrete_map=color_dict, values='MarketCapWeight')

        fig1.update_layout(autosize=True, width=None, title_text="S&P Composition", height=650)
        fig2.update_layout(autosize=True, width=None, title_text="Composition of your portfolio", height=650)

        return [fig1, fig2]  # return both figures


class TableStrategy(ChartStrategy):
    def execute(self, df, portfolio, merged_portfolio, uploaded_file, ticker):

        if merged_portfolio is not None and not merged_portfolio.empty:
            merged_portfolio_html = merged_portfolio.to_html(index=False)
            return merged_portfolio_html
        else:
            return None


class LineChartStrategy(ChartStrategy):
    def execute(self, df, portfolio, merged_portfolio, uploaded_file, ticker):

        if ticker:
            # Retrieve OCHL data for the specified ticker for the past six months
            data = yf.download(ticker, period='12mo')

            fig = go.Figure(data=[go.Ohlc(x=data.index,
                                          open=data['Open'],
                                          high=data['High'],
                                          low=data['Low'],
                                          close=data['Close'])])
            fig.update_layout(title=f"{ticker} OCHL Chart",
                              xaxis_title='Date',
                              yaxis_title='Price',
                              width=1600,
                              height=800)
            return [fig]
        else:
            return None


class TextStrategy(ChartStrategy):
    def execute(self, df, portfolio, merged_portfolio, uploaded_file, ticker):
        return None


@dataclass
class Message:
    origin: Literal["human", "ai"]
    message: str


class ChatApplication:
    def __init__(self):
        load_dotenv()
        file_path = 'static/sp500 mkt cap June 27 2023.xlsx'

        # Check if the file exists
        if os.path.isfile(file_path):
            # If the file exists, load it into a pandas dataframe

            self.df = pd.read_excel(file_path)
        else:
            # If the file doesn't exist, create the dataframe using the scrape_sp500 function

            self.df = scrape_sp500()
            # Save the dataframe to an Excel file
            self.df.to_excel(file_path)
        st.set_page_config(layout="wide")
        icon = Image.open("static/Colour Logo.png")
        st.sidebar.image(icon, width=300)
        st.sidebar.title("Robo Advisor:")
        st.sidebar.title("Powered by LangChain 🦜🔗")
        # Upload control in the sidebar
        self.uploaded_file = st.sidebar.file_uploader(
            "Upload your portfolio", type=["xlsx"]
        )
        st.session_state.temperature = st.sidebar.slider(
            "Set AI response temperature", min_value=0.1, max_value=1.0, value=0.7, step=0.1
        )

        self.chart_strategy = None
        self.user = None
        self.portfolio = pd.DataFrame()
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(
                "The following is an informative conversation between a human and an AI financial adviser . The financial adviser will ask lots of questions. The financial adviser will attempt to answer any question asked and will always probe for the human's risk appetite and investment goals by asking questions of its own. If the human's risk appetite is low it will offer conservative financial advice, if the risk appetite of the human is higher it will offer more aggressive advice. It may ask if the user has an existing portfolio and request it drops it into the dropzone in the sidebar. The adviser's response and it's follow-up question are: "
            ),
            MessagesPlaceholder(variable_name="history"),
            HumanMessagePromptTemplate.from_template("{input}")
        ])

        self.merged_portfolio = pd.DataFrame()
        self.load_css()
        self.initialize_session_state()

        self.chat_placeholder = st.container()
        self.chart_placeholder = st.empty()  # New chart placeholder

        self.prompt_placeholder = st.form("chat-form")
        self.log_placeholder = st.empty()

    def display_chart(self, chart_result):
        if isinstance(chart_result, str):  # If it's a table
            div = f"""
                <div class="chat-row">
                    <img class="chat-icon" src="app/static/ai_icon.png" width=32 height=32>
                    <div class="chat-bubble ai-bubble">
                        &#8203;{chart_result}
                    </div>
                </div>
            """
            st.markdown(div, unsafe_allow_html=True)
        elif isinstance(chart_result, list):  # If it's a list of plotly figures
            for fig in chart_result:
                components.html(fig.to_html(), width=800, height=600)

    def update_portfolio(self):
        portfolio_tickers = self.portfolio['Ticker']
        sp500_tickers = self.df['Ticker']
        is_in_sp500 = portfolio_tickers.isin(sp500_tickers)

        merged_portfolio = pd.merge(self.portfolio, self.df[['Ticker', 'Name', 'Sector', 'SubIndustry']], on='Ticker',
                                    how='left')

        tickers_str = ' '.join(portfolio_tickers)
        data = yf.download(tickers_str, period='1d', interval='1d')
        latest_prices = data['Close'].iloc[-1]

        merged_portfolio['Price ($)'] = merged_portfolio['Ticker'].map(latest_prices)
        merged_portfolio['Value ($)'] = merged_portfolio['Shares'] * merged_portfolio['Price ($)']

        total_portfolio_value = merged_portfolio['Value ($)'].sum()
        merged_portfolio['Percentage'] = merged_portfolio['Value ($)'] / total_portfolio_value

        merged_portfolio['Price ($)'] = merged_portfolio['Price ($)'].round(2)
        merged_portfolio['Value ($)'] = merged_portfolio['Value ($)'].round(2)
        merged_portfolio['Percentage'] = (merged_portfolio['Percentage'] * 100).round(1).astype(str) + '%'

        self.merged_portfolio = merged_portfolio

    def load_css(self):
        with open("static/styles.css", "r") as f:
            css = f"<style>{f.read()}</style>"
            st.markdown(css, unsafe_allow_html=True)

    def initialize_session_state(self):
        if "history" not in st.session_state:
            st.session_state.history = []
        if "portfolio" not in st.session_state:
            st.session_state.portfolio = []
        if "token_count" not in st.session_state:
            st.session_state.token_count = 0
        if "conversation" not in st.session_state:
            llm = ChatOpenAI(
                temperature=st.session_state.temperature,
                openai_api_key=self.openai_api_key,
            )
            st.session_state.conversation = ConversationChain(
                llm=llm,
                memory=ConversationBufferMemory(return_messages=True),
                prompt=self.prompt
            )

    def on_click_callback(self):
        if self.uploaded_file is not None:
            print("Uploaded")
            with get_openai_callback() as cb:
                template_cot = """1. A financial expert looks at this question {human} and answer {ai}.
                    2. Think about the question step by step and decide whether a chart or table will help illustrate the answer to the question
                    3. The expert will write down its reasoning based on the question at hand
                    4. The expert will then append its reasoning with a single word based on what type of chart or table it feels would be most helpful
                    5. For a benchmark comparison they will write the single word STARBURST
                    6. For a display of a table of the user's portfolio they will write the single word TABLE
                    7. For a display of a stock chart they will write the single word CHART
                    8. For a display of a fundamentals and ratios table they will write the single word DATA
                    9. If they think that no chart or graph is needed they will write NONE

                    The expert's reasoning and single word answer is...
                    """

                llm_decide = OpenAI(temperature=0.7, max_tokens=3000)
                prompt = PromptTemplate(template=template_cot, input_variables=["human", "ai"])

                llm_chain = LLMChain(prompt=prompt, llm=llm_decide)

                human_prompt = st.session_state.human_prompt

                history = st.session_state.history

                portfolio = st.session_state.history

                self.portfolio = pd.read_excel(self.uploaded_file)

                self.update_portfolio()
                p_mod = format_portfolio(self.merged_portfolio)
                g_and_a = st.session_state.conversation.predict(
                    input="Can you summarise my investment goals and risk appetite in a single phrase starting with the string 'Goals and appetite are:'?"
                )

                p_string = "1: The following portfolio" + p_mod + "is under discussion between a human, who's " + g_and_a + " and a financial advisor. The advisor will make rational comments about how to diversify the portfolio to help the investor manage its investment goals. Given the stocks in the portfolio the advisor responds: "
                self.prompt = ChatPromptTemplate.from_messages([
                    SystemMessagePromptTemplate.from_template(
                        p_string
                        # "The following is an informative conversation between a human and an AI financial adviser. The adviser will NEVER provide facts and figures, only advice. It would be dangerous to provide facts and figures as the adviser is not aware of the current market conditions or valuations of stocks. The financial adviser knows exactly what is in the user's portfolio. The financial adviser will remind the user that it knows about its portfolio. The users portfolio consists of " + p_mod + ".  The financial adviser will ask lots of questions. The financial adviser will attempt to answer any general question asked and will probe for the human's risk appetite by asking questions of its own. If the human's risk appetite is low it will offer conservative financial advice, if the risk appetite of the human is higher it will offer more aggressive advice. Most importantly: The adviser will never, ever provide any financial data directly, e.g. price, market cap, ratios, percentages etc. It doesn't have access to this information as it is now 2023 and the adviser was trained in 2021. Instead it will say 'details in the information below'. The adviser's response and it's follow-up question are: "
                    ),
                    MessagesPlaceholder(variable_name="history"),
                    HumanMessagePromptTemplate.from_template("{input}")
                ])

                if "uploaded" not in st.session_state:
                    llm2 = ChatOpenAI(
                        temperature=st.session_state.temperature,
                        openai_api_key=self.openai_api_key,
                    )
                    st.session_state.uploaded = ConversationChain(
                        llm=llm2,
                        memory=ConversationBufferMemory(return_messages=True),
                        prompt=self.prompt
                    )
                llm_response = st.session_state.uploaded.predict(
                    input=human_prompt
                )

                llm_chat = check_string(llm_response)

                st.session_state.history.append(
                    Message("human", human_prompt)
                )
                st.session_state.history.append(
                    Message("ai", llm_chat)
                )
                st.session_state.token_count += cb.total_tokens

                st.sidebar.write("Portfolio uploaded successfully!")

                res = llm_chain.run({"ai": llm_response + human_prompt, "human": human_prompt})

                res = res.lower()
                if "starburst" in res:

                    self.chart_strategy = StarburstStrategy()
                elif "table" in res:

                    self.chart_strategy = TableStrategy()
                elif "data" in res:

                    self.chart_strategy = DataStrategy()
                elif "chart" in res:

                    self.chart_strategy = LineChartStrategy()
                else:

                    self.chart_strategy = TextStrategy()

                ticker = get_most_common_ticker(human_prompt, self.df['Ticker'].to_list())


                if ticker == None:
                    t = "MSFT"
                else:
                    t = ticker[0]
                chart_result = self.chart_strategy.execute(self.df, self.portfolio, self.merged_portfolio,
                                                           self.uploaded_file, t)

                if chart_result is not None:
                    if isinstance(chart_result, str):  # If it's a table
                        st.session_state.history.append(Message("ai", chart_result))
                    elif isinstance(chart_result, list):  # If it's a list of plotly figures
                        st.session_state.history.append(Message("ai", chart_result))
        else:

            with get_openai_callback() as cb:
                human_prompt = st.session_state.human_prompt

                history = st.session_state.history

                llm_response = st.session_state.conversation.predict(
                    input=human_prompt
                )

                st.session_state.history.append(
                    Message("human", human_prompt)
                )
                st.session_state.history.append(
                    Message("ai", llm_response)
                )
                st.session_state.token_count += cb.total_tokens

    def run(self):
        with self.chat_placeholder:
            for chat in st.session_state.history:
                if chat.origin == "ai":
                    if isinstance(chat.message, str):  # If it's a table
                        div = f"""
                            <div class="chat-row">
                                <img class="chat-icon" src="app/static/ai_icon.png" width=32 height=32>
                                <div class="chat-bubble ai-bubble">
                                    &#8203;{chat.message}
                                </div>
                            </div>
                        """
                        st.markdown(div, unsafe_allow_html=True)
                    elif isinstance(chat.message, list):  # If it's a list of plotly figures
                        for fig in chat.message:
                            st.plotly_chart(fig)  # Use Streamlit's plotly_chart function to render the chart
                else:
                    div = f"""
                        <div class="chat-row row-reverse">
                            <img class="chat-icon" src="app/static/user_icon.png" width=32 height=32>
                            <div class="chat-bubble human-bubble">
                                &#8203;{chat.message}
                            </div>
                        </div>
                                """
                    st.markdown(div, unsafe_allow_html=True)

            for _ in range(3):
                st.markdown("")

        with self.prompt_placeholder:
            st.markdown("**Chat**")
            cols = st.columns((6, 1))
            cols[0].text_input(
                "Chat",
                value="Hi Lucidate FinBot!",
                label_visibility="collapsed",
                key="human_prompt",
            )
            cols[1].form_submit_button(
                "Submit",
                type="primary",
                on_click=self.on_click_callback,
            )

        self.log_placeholder.caption(f"""
    Used {st.session_state.token_count} tokens \n
    """)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    app = ChatApplication()
    app.run()
