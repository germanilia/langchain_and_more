from bs4 import BeautifulSoup
import requests
import pandas as pd
import yfinance as yf
import time
import os
import spacy
import re

nlp = spacy.load("en_core_web_sm")



def get_ticker_info(ticker, max_attempts=5):
    for attempt in range(max_attempts):
        try:
            ticker_info = yf.Ticker(ticker).info
            return ticker_info
        except requests.exceptions.RequestException as e:
            print(f"Request to {ticker} failed, attempt {attempt + 1} of {max_attempts}")
            if attempt < max_attempts - 1:  # No need to sleep for the last attempt
                sleep_time = 2 ** attempt  # Exponential backoff
                print(f"Sleeping for {sleep_time} seconds before retrying")
                time.sleep(sleep_time)
            else:
                print(f"Failed to retrieve data for {ticker} after {max_attempts} attempts.")
    return None


def scrape_sp500():
    wiki_page = requests.get('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies').text
    sp500_list = BeautifulSoup(wiki_page, 'html.parser')
    table = sp500_list.find('table', {'class': 'wikitable sortable'})
    sector_tickers = dict()
    for row in table.findAll('tr')[1:]:
        name = row.findAll('td')[1].text.strip()
        ticker = row.findAll('td')[0].text
        ticker = ticker[:-1]
        sector = row.findAll('td')[2].text
        subindustry = row.findAll('td')[3].text
        ticker_info = get_ticker_info(ticker)
        if ticker_info is None:
            continue
        market_cap = ticker_info.get('marketCap', float('nan'))
        sector_tickers[ticker] = [ticker, name, sector, subindustry, market_cap]

        time.sleep(1)  # prevent IP from getting temporarily blocked
    df = pd.DataFrame.from_dict(sector_tickers, orient='index',
                           columns=['Ticker', 'Name', 'Sector', 'SubIndustry', 'MarketCap'])
    dir_path = "./static"
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    df.to_excel("./static/sp500 mkt cap June 27 2023.xlsx")
    return df

def get_sp500_tickers_from_text(text, sp500):
    doc = nlp(text)

    sp500_tickers = []
    sp500_list = sp500.to_list()


    for entity in doc.ents:
        if entity.label_ == "ORG":
            ticker = entity.text.upper()
            if ticker in sp500_list:
                sp500_tickers.append(ticker)

    return set(sp500_tickers)


def format_portfolio(portfolio):

    stocks = portfolio['Ticker']
    allocations = portfolio['Percentage']
    portfolio_str = "portfolio consists of "
    stock_allocations = [f"{stock} {allocation}" for stock, allocation in zip(stocks, allocations)]
    if len(stock_allocations) > 1:
        portfolio_str += ", ".join(stock_allocations[:-1]) + ", and " + stock_allocations[-1]
    else:
        portfolio_str += stock_allocations[0]


    return portfolio_str


def get_most_common_ticker(text, sp500_tickers):
    doc = nlp(text)

    # Initialize a dictionary to store the count of each stock ticker
    ticker_count = {}
    ticker_list = []

    for entity in doc.ents:
        if entity.label_ == "ORG":  # Check for organization entities
            ticker = entity.text.upper()  # Convert to uppercase for consistency
            ticker_list.append(ticker)

            # Update the count of the ticker in the dictionary
            ticker_count[ticker] = ticker_count.get(ticker, 0) + 1

    # Check if an S&P ticker with whitespace around it exists in the text
    ticker_matches = []
    for ticker in sp500_tickers:
        ticker_with_whitespace = rf"\b{re.escape(ticker)}\b"
        if re.search(ticker_with_whitespace, text, re.IGNORECASE):
            ticker_matches.append(ticker)
    print(ticker_matches)
    if ticker_matches:
        # Return the ticker with the largest number of characters
        longest_ticker = max(ticker_matches, key=len)
        print(longest_ticker)
        return [longest_ticker]

    if ticker_count:
        # Get the most commonly referenced stock ticker
        most_common_ticker = max(ticker_count, key=ticker_count.get)
        if most_common_ticker in sp500_tickers:
            return most_common_ticker

    # Check if an S&P ticker is present in the text
    for ticker in sp500_tickers:
        ticker_pattern = r"\b" + re.escape(ticker) + r"\b"
        if re.search(ticker_pattern, text, re.IGNORECASE):
            return [ticker]

    return None


def check_string(input_string):
    if "but as a text-based AI" in input_string or "Please note that these " in input_string:
        return "Here you go!"
    elif "I'm sorry, but " in input_string or "I apologize, but " in input_string:
        return "Here you go!"
    else:
        return input_string
