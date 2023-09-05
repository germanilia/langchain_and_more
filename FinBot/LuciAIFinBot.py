import re

import streamlit as st
from PIL import Image
import pandas as pd
import yfinance as yf
import pandas_ta as ta


from langchain.chat_models import ChatOpenAI
from langchain_experimental.plan_and_execute import PlanAndExecute, load_agent_executor, load_chat_planner
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
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from streamlit_option_menu import option_menu
from bokeh.palettes import Category20

st.set_page_config(layout='wide')
def load_css():
    with open("static/styles.css", "r") as f:
        css = f"<style>{f.read()}</style>"
        st.markdown(css, unsafe_allow_html=True)

sp500_file = 'static/sp500.xlsx'
try:
    sp500_data = pd.read_excel(sp500_file)
except FileNotFoundError:
    sp500_data = None

load_css()



# Define pages
page1 = "AI Portfolio Advice"
page2 = "Composition vs S&P 500"
page3 = "Stock Analysis"
page4 = "Benchmarks"

# Define menu options
selected = option_menu(None, [page1, page2, page3, page4],
    icons=['house', 'pie-chart-fill', 'graph-up', 'calculator'],
    menu_icon="cast", default_index=0, orientation='horizontal')






load_dotenv()
news_api_key = os.getenv("NEWS_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Load your logo image (ensure it's in the same directory or provide the full path)
logo = Image.open('static/Colour Logo.png')
bears = Image.open('static/inv bear.jpeg')



# Sidebar with a title, a logo and a slider

st.sidebar.title('Langchain Agents & Tools🦜')
st.sidebar.image(logo, use_column_width=False)
if st.sidebar.button('Refresh S&P500'):
    # Fetch fresh data from API
    sp500_tickers = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]['Symbol'].tolist()

    # Create an empty DataFrame to store the data
    columns = ['Name', 'Industry', 'Sector', 'MarketCap']

    sp500_data = pd.DataFrame(columns=columns)

    # Fetch data for each ticker
    for ticker in sp500_tickers:
        stock = yf.Ticker(ticker)
        info = stock.info

        name = info.get('longName', 'N/A')
        industry = info.get('industry', 'N/A')
        sector = info.get('sector', 'N/A')
        market_cap = info.get('marketCap', 'N/A')

        sp500_data.loc[ticker] = [name, industry, sector, market_cap]

        # Set the index of the DataFrame as the ticker
        sp500_data['MarketCap'] = pd.to_numeric(sp500_data['MarketCap'], errors='coerce').fillna(0).astype(int)

        sp500_data.index.name = 'Ticker'


    # Cache to file for next run
    sp500_data.to_excel(sp500_file, index=True)


    st.success('S&P500 data refreshed!')
excel_file = st.sidebar.file_uploader("Upload Your portfolio")

if excel_file:

  # Read excel into pandas dataframe
  df = pd.read_excel(excel_file)

  sp500_data = sp500_data.reset_index()
  merged_df = df.merge(sp500_data, on='Ticker',  how='left')

  # Fetch the current price of each share
  for index, row in merged_df.iterrows():
      ticker = row['Ticker']
      stock = yf.Ticker(ticker)
      current_price = stock.history(period="1d")["Close"].iloc[0]
      merged_df.at[index, 'CurrentPrice'] = current_price

  # Calculate the cash value of each position
  merged_df['CashValue'] = merged_df['Shares'] * merged_df['CurrentPrice']

  # Calculate the total portfolio value
  total_portfolio_value = merged_df['CashValue'].sum()

  # Calculate the percentage of the portfolio that each position makes up
  merged_df['PortfolioPercentage'] = (merged_df['CashValue'] / total_portfolio_value) * 100

  cols_to_show = ["Ticker", "Name", "Shares", "Industry", "Sector", "MarketCap",
                  "CurrentPrice", "CashValue", "PortfolioPercentage"]
  disp = merged_df.loc[:, cols_to_show]
  with st.expander("Show portfolio"):
    st.write(merged_df.loc[:, cols_to_show])


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
def ai_chat(model):

    st.image(bears, use_column_width=False, width=800)
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
def sp_starburst():
    sp500_grouped = sp500_data.groupby(['Sector', 'Industry', 'Name']).sum().reset_index()
    merged_grouped = merged_df.groupby(['Sector', 'Industry', 'Name']).sum().reset_index()
    unique_sectors = list(set(sp500_grouped['Sector']))
    color_dict = {sector: color for sector, color in zip(unique_sectors, Category20[20])}

    # Create a starburst chart
    def create_starburst_chart(data, col):

        color_dict = {sector: color for sector, color in zip(unique_sectors, Category20[20])}
        fig = px.sunburst(data, path = ['Sector', 'Industry', 'Name'], color='Sector', color_discrete_map=color_dict, values=col)
        fig.update_layout(margin=dict(t=0, l=0, r=0, b=0))

        return fig

    # Streamlit app
    st.title("Starburst Charts")

    # Create and display starburst chart for S&P 500
    st.subheader("S&P 500 Starburst Chart")
    sp500_chart = create_starburst_chart(sp500_grouped, "MarketCap")
    st.plotly_chart(sp500_chart, use_container_width=True)


    # Create and display starburst chart for merged portfolio
    st.subheader("AI Portfolio Starburst Chart")
    merged_chart = create_starburst_chart(merged_grouped, "PortfolioPercentage")
    st.plotly_chart(merged_chart, use_container_width=True)

def ticker_compare():
    stock_dict = merged_df.set_index('Ticker')['Name'].to_dict()
    term_dict = {"1m": "1mo", "2m": "2mo", "3m": "3mo", "6m": "6mo", "1Yr": "1y", "2Yr": "2y"}
    with st.container():
        col1, col2, col3, col4 = st.columns(4)
        selected_ticker = col1.selectbox('Select a stock ticker', list(stock_dict.keys()), key='dropdown')
        time_period = col2.selectbox("Select Time Period", options=list(term_dict.keys()))
        rsi_t = col4.checkbox("Show RSI on main chart")
        ind_sec = col3.selectbox("Industry/Sector", ["Industry", "Sector"])

    # Display the selected stock name
    selected_stock_current_price = merged_df.loc[
        merged_df['Ticker'] == selected_ticker, 'CurrentPrice'
    ].values[0]
    st.title(f"OHLC Chart for {stock_dict[selected_ticker]}: [${selected_stock_current_price:.2f}] for {term_dict[time_period]}")


    stock_data = yf.download(selected_ticker, period=term_dict[time_period], progress=False)
    stock_data.ta.rsi(length=14, append=True)

    candlestick_chart = go.Candlestick(
        x=stock_data.index,
        open=stock_data['Open'],
        high=stock_data['High'],
        low=stock_data['Low'],
        close=stock_data['Close'],
        name="Candlesticks"
    )

    # Create RSI line trace
    rsi_trace = go.Scatter(
        x=stock_data.index,
        y=stock_data['RSI_14'],
        mode='lines',
        name='RSI',
        yaxis='y2'  # Assign the RSI trace to the second y-axis

    )

    # Create figure with Candlestick chart
    fig = go.Figure(data=[candlestick_chart])

    # Add RSI line trace to the figure
    if rsi_t:
        fig.add_trace(rsi_trace)

    # Add a secondary y-axis to the figure for RSI
    fig.update_layout(
        yaxis2=dict(title='RSI', overlaying='y', side='right', showgrid=False, range=[0,100]),
        margin=dict(t=40, b=40, r=40, l=40),
        title=f"{selected_ticker} Candlestick Chart with RSI (Secondary Axis)"
    )

    # Display the Candlestick chart using Plotly and Streamlit
    st.plotly_chart(fig, use_container_width=800)

    if ind_sec == "Industry":
        industry_tickers = sp500_data[sp500_data['Industry'] == sp500_data[sp500_data['Ticker'] == selected_ticker]['Industry'].iloc[0]]['Ticker']
    else:
        industry_tickers = \
        sp500_data[sp500_data['Sector'] == sp500_data[sp500_data['Ticker'] == selected_ticker]['Sector'].iloc[0]][
            'Ticker']
    industry_data = yf.download(industry_tickers.tolist(), period=term_dict[time_period])['Adj Close']

    # Calculate gains/losses for each ticker
    ticker_gains_losses = (industry_data.iloc[-1] / industry_data.iloc[0] - 1) * 100
    sorted_ticker_gains_losses = ticker_gains_losses.sort_values(ascending=False)
    sorted_ticker_gains_losses = sorted_ticker_gains_losses.reset_index()
    sorted_ticker_gains_losses.columns = ['Ticker', 'Gain']
    sorted_ticker_gains_losses.set_index('Ticker')


    # Create a bar chart
    color_discrete_map = {selected_ticker: 'yellow'}
    for ticker in sorted_ticker_gains_losses.Ticker:
        if ticker != selected_ticker:
            color_discrete_map[ticker] = 'rgb(0, 102, 204, 0.6)'



    fig = go.Figure(data=[go.Bar(
        x=sorted_ticker_gains_losses['Ticker'],  # Ticker names on x-axis
        y=sorted_ticker_gains_losses['Gain'],  # Gain values on y-axis
        marker_color=[color_discrete_map[ticker] for ticker in sorted_ticker_gains_losses['Ticker']]
        # Set colors based on map
    )])

    # Customize the layout
    fig.update_layout(
        title=f"Gains and Losses for {selected_ticker} and Industry Peers ({term_dict[time_period]})",
        xaxis_title='Ticker',
        yaxis_title='Gain/Loss (%)',
        showlegend=False  # Hide the legend
    )

    # Display the vertical column chart using Plotly and Streamlit
    selected_stock_gain = sorted_ticker_gains_losses.loc[
        sorted_ticker_gains_losses['Ticker'] == selected_ticker, 'Gain'
    ].values[0]
    st.title(f"Peer analysis vs {ind_sec} for {selected_ticker} for {term_dict[time_period]}: Gain/Loss: {selected_stock_gain:.2f}%")
    st.plotly_chart(fig, use_container_width=True)


def benchmarks():
    term_dict = {"1m": "1mo", "2m": "2mo", "3m": "3mo", "6m": "6mo", "1Yr": "1y", "2Yr": "2y"}
    with st.container():
        col1, col2, col3, col4 = st.columns(4)
        time_period = col1.selectbox("Select Time Period", options=list(term_dict.keys()))

    # Fetch the list of tickers in the portfolio
    portfolio_tickers = merged_df['Ticker'].to_list()

    # Define a list of index tickers to compare
    index_tickers = ["^FTSE", "^FTMC", "^GSPC", "^IXIC", "^STOXX", "^FCHI", "^RUT"]

    # Download historical stock data for both portfolio and index tickers
    tickers_to_download = portfolio_tickers + index_tickers
    stock_data = yf.download(tickers_to_download, period=term_dict[time_period], progress=False)['Close']
    stock_data.fillna(method='ffill', inplace=True)
    # Calculate the portfolio value over time
    portfolio_value = (stock_data[portfolio_tickers] * merged_df['Shares'].values).sum(axis=1)

    # Normalize the portfolio value to be 1 on the start date
    normalized_portfolio_value = portfolio_value / portfolio_value.iloc[0]

    # Calculate the relative performance of indices
    index_relative_performance = stock_data[index_tickers].div(stock_data[index_tickers].iloc[0], axis=1)

    # Create a line chart for the relative performance using Plotly

    index_legend_labels = {
        "^FTSE": "FTSE 100",
        "^FTMC": "FTSE 250",
        "^GSPC": "S&P 500",
        "^IXIC": "Nasdaq",
        "^STOXX": "Stoxx 600",
        "^FCHI": "CAC 40",
        "^RUT": "Russell 2000"
    }
    figure = go.Figure()

    # Add portfolio relative performance to the chart
    figure.add_trace(
        go.Scatter(x=normalized_portfolio_value.index, y=normalized_portfolio_value, mode='lines', name='Portfolio'))

    # Add index relative performances to the chart
    for index_ticker in index_tickers:
        figure.add_trace(
            go.Scatter(x=index_relative_performance.index, y=index_relative_performance[index_ticker], mode='lines',
                       name=index_legend_labels.get(index_ticker, index_ticker)))

    figure.update_layout(
        title=f'Relative Portfolio and Index Performance Over {term_dict[time_period]}',
        xaxis_title='Date',
        yaxis_title='Relative Performance',
        legend=dict(title="Indices", orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    # Calculate the relative performance of commodities
    commodity_legend_labels = {
        "GC=F": "Gold",
        "SI=F": "Silver",
        "CL=F": "Oil",
        "NG=F": "Gas",
        "ZW=F": "Wheat",
        "LE=F": "Cattle",
        "OJ=F": "Orange Juice"
    }

    commodity_tickers = list(commodity_legend_labels.keys())
    commodity_data = yf.download(commodity_tickers, period=term_dict[time_period], progress=False)['Close']
    commodity_relative_performance = commodity_data.div(commodity_data.iloc[0], axis=1)

    # ... (create the index chart as shown in your previous code) ...

    # Create a line chart for the relative performance using Plotly
    fig = go.Figure()

    # Add portfolio relative performance to the chart
    fig.add_trace(
        go.Scatter(x=normalized_portfolio_value.index, y=normalized_portfolio_value, mode='lines', name='Portfolio'))

    # Add commodity relative performances to the chart
    for commodity_ticker in commodity_tickers:
        fig.add_trace(
            go.Scatter(x=commodity_relative_performance.index, y=commodity_relative_performance[commodity_ticker],
                       mode='lines', name=commodity_legend_labels.get(commodity_ticker, commodity_ticker)))

    fig.update_layout(
        title=f'Relative Portfolio and Commodity Performance Over {term_dict[time_period]}',
        xaxis_title='Date',
        yaxis_title='Relative Performance',
        legend=dict(title="Commodities", orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    with st.container():
        col1, col2 = st.columns(2)
        col1.plotly_chart(figure, use_container_width=True)
        col2.plotly_chart(fig, use_container_width=True)

    sector_etfs = {
        "Communication Services": "XLC",
        "Consumer Discretionary": "XLY",
        "Consumer Staples": "XLP",
        "Energy": "XLE",
        "Financials": "XLF",
        "Health Care": "XLV",
        "Industrials": "XLI",
        "Materials": "XLB",
        "Real Estate": "XLRE",
        "Technology": "XLK",
        "Utilities": "XLU"
    }

    portfolio_tickers = merged_df['Ticker'].to_list()

    # Define a list of sector ETF tickers to compare
    sector_etf_tickers = list(sector_etfs.values())

    # Download historical stock data for both portfolio and sector ETF tickers
    tickers_to_download = portfolio_tickers + sector_etf_tickers
    stock_data = yf.download(tickers_to_download, period=term_dict[time_period], progress=False)['Close']
    stock_data.fillna(method='ffill', inplace=True)

    # Calculate the portfolio value over time
    portfolio_value = (stock_data[portfolio_tickers] * merged_df['Shares'].values).sum(axis=1)

    # Normalize the portfolio value to be 1 on the start date
    normalized_portfolio_value = portfolio_value / portfolio_value.iloc[0]

    # Calculate the relative performance of sector ETFs
    sector_etf_relative_performance = stock_data[sector_etf_tickers].div(stock_data[sector_etf_tickers].iloc[0], axis=1)

    # Create a line chart for the relative performance using Plotly

    sector_etf_legend_labels = {ticker: sector for sector, ticker in sector_etfs.items()}

    figure = go.Figure()

    # Add portfolio relative performance to the chart
    figure.add_trace(
        go.Scatter(x=normalized_portfolio_value.index, y=normalized_portfolio_value, mode='lines', name='Portfolio'))

    # Add sector ETF relative performances to the chart
    for sector_etf_ticker in sector_etf_tickers:
        figure.add_trace(
            go.Scatter(x=sector_etf_relative_performance.index, y=sector_etf_relative_performance[sector_etf_ticker],
                       mode='lines', name=sector_etf_legend_labels.get(sector_etf_ticker, sector_etf_ticker)))

    figure.update_layout(
        title=f'Relative Portfolio and Sector ETF Performance Over {term_dict[time_period]}',
        xaxis_title='Date',
        yaxis_title='Relative Performance',
        legend=dict(title="Sectors", orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=230, b=60, r=40, l=40), height=600  # Adjust top and bottom margins
    )

    # Define a dictionary for cryptocurrency legend labels
    crypto_legend_labels = {
        "BTC-USD": "Bitcoin",
        "ETH-USD": "Ethereum",
        "XRP-USD": "Ripple",
        "BNB-USD": "Binance Coin",
        "DOGE-USD": "Dogecoin"
    }

    crypto_tickers = list(crypto_legend_labels.keys())

    # Download historical cryptocurrency data
    crypto_data = yf.download(crypto_tickers, period=term_dict[time_period], progress=False)['Close']
    crypto_data.fillna(method='ffill', inplace=True)

    # Calculate the relative performance of cryptocurrencies
    crypto_relative_performance = crypto_data.div(crypto_data.iloc[0], axis=1)

    # Create a line chart for the relative performance of cryptocurrencies using Plotly
    fig = go.Figure()

    # Add portfolio relative performance to the chart
    fig.add_trace(
        go.Scatter(x=normalized_portfolio_value.index, y=normalized_portfolio_value, mode='lines', name='Portfolio'))

    # Add cryptocurrency relative performances to the chart
    for crypto_ticker in crypto_tickers:
        fig.add_trace(
            go.Scatter(x=crypto_relative_performance.index, y=crypto_relative_performance[crypto_ticker],
                       mode='lines', name=crypto_legend_labels.get(crypto_ticker, crypto_ticker)))

    fig.update_layout(
        title=f'Relative Portfolio and Cryptocurrency Performance Over {term_dict[time_period]}',
        xaxis_title='Date',
        yaxis_title='Relative Performance',
        legend=dict(title="Cryptocurrencies", orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=230, b=60, r=40, l=40), height=600
    )

    # ... (rest of the code for sector ETF relative performance) ...

    # Plot the relative performance charts
    with st.container():
        col1, col2 = st.columns(2)
        col1.plotly_chart(figure, use_container_width=True)
        col2.plotly_chart(fig, use_container_width=True)

if selected == page1:
    ai_chat(model)
elif selected == page2:
    sp_starburst()
elif selected == page3:
    ticker_compare()
elif selected == page4:
    benchmarks()