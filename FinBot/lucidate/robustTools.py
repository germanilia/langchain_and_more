import yfinance as yf
from typing import Dict, List
from langchain.tools import BaseTool, tool, StructuredTool
from pydantic import BaseModel, Field
from langchain.callbacks.manager import (
    CallbackManagerForToolRun,
    AsyncCallbackManagerForToolRun,
)
from langchain.tools.base import ToolException
from typing import Optional, Type
from newsapi import NewsApiClient
import os
from langchain.callbacks.base import BaseCallbackHandler
from langchain.agents.agent import AgentAction
from langchain.agents import AgentType
from langchain.callbacks import StreamlitCallbackHandler

import sys
from io import StringIO
from collections import defaultdict


class LuciMessageCollector:

    def __init__(self):
        self.messages = defaultdict(list)

    def __enter__(self):
        self.old_stdout = sys.stdout
        self.buf = StringIO()
        sys.stdout = self.buf

        return self

    def __exit__(self, type, value, traceback):
        sys.stdout = self.old_stdout

        self.parse_messages()

    def parse_messages(self):
        try:
            for line in self.buf.getvalue().splitlines():
                if '"action_input"' in line:
                    start = line.index('"action_input"') + len('"action_input"') + 3
                    end = line.rindex('"')
                    message = line[start:end]

                    self.messages["Actions"].append(message)
                elif line.startswith("Response:"):
                    self.messages["Responses"].append(line)
                elif line.startswith("Observation:"):
                    self.messages["Observations"].append(line)
                elif line.startswith("Thought:"):
                    self.messages["Thoughts"].append(line)
                elif line.startswith("Step:"):
                    self.messages["Steps"].append(line)
        except Exception as e:
            print(f"Error in parse_messages(): {e}")




def _handle_error(error: ToolException) -> str:
    return (
        "The following errors occurred during tool execution:"
        + error.args[0]
        + "Please try another tool."
    )

class StockSchema(BaseModel):
    ticker: str = Field(description="should be a valid stock ticker")

class StockTool(BaseTool):
    name = "stock_tool"
    description = "useful for when you need to get stock history and fundamentals"
    args_schema: Type[StockSchema] = StockSchema

    def _run(
        self,
        ticker: str,

        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool."""
        stock = yf.Ticker(ticker)
        history = stock.history(period="1mo")
        info = stock.info

        return {"history": history}

    async def _arun(
        self,
        ticker: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("stock_tool does not support async")


class StockInfoTool(BaseTool):
    name = "stock_tool"
    description = "useful for when you need to get stock history and fundamentals"
    args_schema: Type[StockSchema] = StockSchema

    def _run(
        self,
        ticker: str,

        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool."""
        stock = yf.Ticker(ticker)
        history = stock.history(period="1mo")
        info = stock.info
        # remove keys from the dictionary
        keys_to_remove = ['address1', 'city', 'state', 'zip', 'country', 'phone', 'fax', 'website', 'longBusinessSummary', 'companyOfficers', 'underlyingSymbol', 'firstTradeDateEpoch', 'timeZoneFullName', 'timeZoneShortName', 'messageBoardId', 'gmtOffSetMillliseconds', 'exchange', 'quoteType', 'governanceEpochDate', 'compensationAsOfEpochDate']
        for key in keys_to_remove:
            if key in info:
                del info[key]
        return {"fundamentals": info}


    async def _arun(
        self,
        ticker: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("stock_tool does not support async")


class NewsSchema(BaseModel):
    ticker: str = Field(description="should be a valid stock ticker")

class NewsTool(BaseTool):
    name = "news_tool"
    description = "useful for when you need to get the latest news about a company"
    args_schema: Type[NewsSchema] = NewsSchema

    def _run(
        self,
        ticker: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool."""
        # instantiate newsapi here

        newsapi = NewsApiClient(api_key=os.getenv("NEWSAPI_API_KEY"))
        top_headlines = newsapi.get_everything(q=ticker, language='en', page_size=10)
        return top_headlines

    async def _arun(
        self,
        ticker: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool asynchronously."""
        # newsapi async code goes here if available
        raise NotImplementedError("news_tool does not support async")



@tool
def get_company_news(ticker: str) -> dict:
    """Gets the latest news for a company given its stock ticker."""
    top_headlines = newsapi.get_everything(q=ticker, language='en')
    return {"news": top_headlines}




def calculate_score(fundamentals):
    PROFIT_WEIGHT = 2
    GROWTH_WEIGHT = 1.5
    FINANCIAL_WEIGHT = 1
    VALUATION_WEIGHT = 1.5
    SENTIMENT_WEIGHT = 0.5

    # Individual metric weights
    GROSS_MARGIN_WEIGHT = 1.2
    OP_MARGIN_WEIGHT = 1
    NET_MARGIN_WEIGHT = 0.8
    ROA_WEIGHT = 0.9
    ROE_WEIGHT = 1
    ROIC_WEIGHT = 0.9

    REVENUE_GROWTH_WEIGHT = 1.1
    EARNINGS_GROWTH_WEIGHT = 0.9
    DIVIDEND_GROWTH_WEIGHT = 0.8
    BOOK_VALUE_GROWTH_WEIGHT = 0.7

    CURRENT_RATIO_WEIGHT = 1.1
    QUICK_RATIO_WEIGHT = 1
    DEBT_TO_EQUITY_WEIGHT = 0.9
    EQUITY_MULTIPLIER_WEIGHT = 0.8
    PAYOUT_RATIO_WEIGHT = 0.8
    FCF_WEIGHT = 1
    FCF_PER_SHARE_WEIGHT = 0.9
    CASH_PER_SHARE_WEIGHT = 0.8

    PE_RATIO_WEIGHT = 1.1
    PRICE_TO_BOOK_WEIGHT = 0.9
    PRICE_TO_SALES_WEIGHT = 0.8
    ENTERPRISE_VALUE_WEIGHT = 1

    RECOMMENDATION_WEIGHT = 1.1
    NUM_ANALYSTS_WEIGHT = 0.9
    AVERAGE_VOLUME_WEIGHT = 0.8
    scores = []

    for fundamental in fundamentals:
        financials = fundamental['financialData']

        # Profitability
        grossProfitMargin_score = scale(financials['grossProfitMargin'], 0.4, 0.8, PROFIT_WEIGHT * GROSS_MARGIN_WEIGHT)

        operatingProfitMargin_score = scale(financials['operatingProfitMargin'], 0.1, 0.4,
                                            PROFIT_WEIGHT * OP_MARGIN_WEIGHT)

        netProfitMargin_score = scale(financials['netProfitMargin'], 0.05, 0.2, PROFIT_WEIGHT * NET_MARGIN_WEIGHT)

        returnOnAssets_score = scale(financials['returnOnAssets'], 0.1, 0.3, PROFIT_WEIGHT * ROA_WEIGHT)

        returnOnEquity_score = scale(financials['returnOnEquity'], 0.15, 0.4, PROFIT_WEIGHT * ROE_WEIGHT)

        returnOnInvestment_score = scale(financials['returnOnInvestment'], 0.15, 0.4, PROFIT_WEIGHT * ROIC_WEIGHT)

        # Growth
        revenueGrowth_score = scale(financials['revenueGrowth'], 0.05, 0.3, GROWTH_WEIGHT * REVENUE_GROWTH_WEIGHT)

        earningsGrowth_score = scale(financials['earningsGrowth'], 0.05, 0.3, GROWTH_WEIGHT * EARNINGS_GROWTH_WEIGHT)

        dividendGrowth_score = scale(financials['dividendGrowth'], 0.05, 0.3, GROWTH_WEIGHT * DIVIDEND_GROWTH_WEIGHT)

        bookValuePerShareGrowth_score = scale(financials['bookValuePerShareGrowth'], 0.1, 0.3,
                                              GROWTH_WEIGHT * BOOK_VALUE_GROWTH_WEIGHT)

        # Financial Health
        currentRatio_score = scale(financials['currentRatio'], 1, 3, FINANCIAL_WEIGHT * CURRENT_RATIO_WEIGHT)

        quickRatio_score = scale(financials['quickRatio'], 1, 2, FINANCIAL_WEIGHT * QUICK_RATIO_WEIGHT)

        debtToEquity_score = scale(financials['debtToEquity'], 0, 1, FINANCIAL_WEIGHT * DEBT_TO_EQUITY_WEIGHT)

        equityMultiplier_score = scale(financials['equityMultiplier'], 1, 2,
                                       FINANCIAL_WEIGHT * EQUITY_MULTIPLIER_WEIGHT)

        payoutRatio_score = scale(financials['payoutRatio'], 0, 1, FINANCIAL_WEIGHT * PAYOUT_RATIO_WEIGHT)

        freeCashFlow_score = scale(financials['freeCashFlow'], 0, None, FINANCIAL_WEIGHT * FCF_WEIGHT)

        freeCashFlowPerShare_score = scale(financials['freeCashFlowPerShare'], 0, None,
                                           FINANCIAL_WEIGHT * FCF_PER_SHARE_WEIGHT)

        cashPerShare_score = scale(financials['cashPerShare'], 0, None, FINANCIAL_WEIGHT * CASH_PER_SHARE_WEIGHT)

        # Valuation
        pERatio_score = scale(financials['pERatio'], 0, 20, VALUATION_WEIGHT * PE_RATIO_WEIGHT)

        priceToBook_score = scale(financials['priceToBook'], 0, 3, VALUATION_WEIGHT * PRICE_TO_BOOK_WEIGHT)

        priceToSalesTrailing12Months_score = scale(financials['priceToSalesTrailing12Months'], 0, 2,
                                                   VALUATION_WEIGHT * PRICE_TO_SALES_WEIGHT)

        enterpriseValue_score = scale(financials['enterpriseValue'], 0, None,
                                      VALUATION_WEIGHT * ENTERPRISE_VALUE_WEIGHT)

        # Sentiment
        analystTargetPrice_score = scale(fundamental['analystTargetPrice'], -1, 1,
                                         SENTIMENT_WEIGHT * RECOMMENDATION_WEIGHT)

        numberOfAnalystOpinions_score = scale(fundamental['numberOfAnalystOpinions'], 5, None,
                                              SENTIMENT_WEIGHT * NUM_ANALYSTS_WEIGHT)

        averageDailyVolume10Day_score = scale(fundamental['averageDailyVolume10Day'], 500000, None,
                                              SENTIMENT_WEIGHT * AVERAGE_VOLUME_WEIGHT)

        # Sum weighted scores
        total_score = sum([
            grossProfitMargin_score,
            operatingProfitMargin_score,
            netProfitMargin_score,
            returnOnAssets_score,
            returnOnEquity_score,
            returnOnInvestment_score,
            revenueGrowth_score,
            earningsGrowth_score,
            dividendGrowth_score,
            bookValuePerShareGrowth_score,
            currentRatio_score,
            quickRatio_score,
            debtToEquity_score,
            equityMultiplier_score,
            payoutRatio_score,
            freeCashFlow_score,
            freeCashFlowPerShare_score,
            cashPerShare_score,
            pERatio_score,
            priceToBook_score,
            priceToSalesTrailing12Months_score,
            enterpriseValue_score,
            analystTargetPrice_score,
            numberOfAnalystOpinions_score,
            averageDailyVolume10Day_score
        ])

        scores.append(total_score)

    return scores

def get_fundamentals(tickers: List[str]):
    '''useful for when you need to get stock history and fundamentals for a list of ticker symbols'''
    scores = {}
    for ticker in tickers:
        stock = yf.Ticker(ticker)
        info = stock.info
        score = calculate_score(info)
        print(ticker, score)
        scores[ticker] = score
    return scores


class MultiStockInput(BaseModel):
    tickers: List[str] = Field(description="should be a list of valid stock tickers")

class TickersAndFundamentals(BaseModel):
    tickers: List[str] = Field(description="should be a list of valid stock tickers")
    fundamentals: List[str] = Field(description="should be a list of valid fundamentals")

class MultiStockInfoTool(BaseTool):
    name = "multi_stock_tool"
    description = "useful for when you need to get stock fundamentals for multiple tickers"
    args_schema: Type[MultiStockInput] = MultiStockInput
    scores = {}

    def _run(
            self,
            ticker: List[str],

            run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        print(f"Ticker: {ticker}, type: {type(ticker)}")
        """Use the tool."""
        stock = yf.Ticker(ticker)
        history = stock.history(period="1mo")
        info = stock.info
        # remove keys from the dictionary
        keys_to_remove = ['address1', 'city', 'state', 'zip', 'country', 'phone', 'fax', 'website',
                          'longBusinessSummary', 'companyOfficers', 'underlyingSymbol', 'firstTradeDateEpoch',
                          'timeZoneFullName', 'timeZoneShortName', 'messageBoardId', 'gmtOffSetMillliseconds',
                          'exchange', 'quoteType', 'governanceEpochDate', 'compensationAsOfEpochDate']
        for key in keys_to_remove:
            if key in info:
                del info[key]

        for ticker in tickers:
            stock = yf.Ticker(ticker)
            info = stock.info
            score = calculate_score(info)
            print(ticker, score)
            scores[ticker] = score
        return scores
    async def _arun(
        self,
        ticker: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool asynchronously."""
        # newsapi async code goes here if available
        raise NotImplementedError("news_tool does not support async")

class MultiStockAndFundamentalsTool(BaseModel):
    name = "multi_stock_and_fundamentals_tool"
    description = "useful for when you need to get stock fundamentals for multiple tickers with a specified list of fundamentals for each stock"
    args_schema: Type[TickersAndFundamentals] = TickersAndFundamentals
    scores = {}
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION
    def _run(
            self,
            ticker: str,

            run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool."""
        stock = yf.Ticker(ticker)
        history = stock.history(period="1mo")
        info = stock.info
        # remove keys from the dictionary
        keys_to_remove = ['address1', 'city', 'state', 'zip', 'country', 'phone', 'fax', 'website',
                          'longBusinessSummary', 'companyOfficers', 'underlyingSymbol', 'firstTradeDateEpoch',
                          'timeZoneFullName', 'timeZoneShortName', 'messageBoardId', 'gmtOffSetMillliseconds',
                          'exchange', 'quoteType', 'governanceEpochDate', 'compensationAsOfEpochDate']
        for key in keys_to_remove:
            if key in info:
                del info[key]

        for ticker in tickers:
            stock = yf.Ticker(ticker)
            info = stock.info
            score = calculate_score(info)
            print(ticker, score)
            scores[ticker] = score
        return scores
    async def _arun(
        self,
        ticker: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool asynchronously."""
        # newsapi async code goes here if available
        raise NotImplementedError("news_tool does not support async")






