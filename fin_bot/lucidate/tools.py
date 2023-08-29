import yfinance as yf
from langchain.tools import BaseTool, tool
from pydantic import BaseModel, Field
from langchain.callbacks.manager import (
    CallbackManagerForToolRun,
    AsyncCallbackManagerForToolRun,
)
from typing import Optional, Type
from newsapi import NewsApiClient
import os
from langchain.tools.base import ToolException

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
