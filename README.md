# WIX Workshop: Analyzing IPOs

## Data Sources:
- [SEC](https://www.sec.gov) - For IPO filings and financial statements.
- [Alpha Vantage](https://www.alphavantage.co) - Financial data API.
- [Yahoo Finance](https://finance.yahoo.com) - Stock prices and news.

## Frameworks:

### 1. Langchain
- **Web Scraping** - Can scrape websites for news, financial data, etc.
- **Chains** - Chain prompts together for complex queries.
- **Tools** - Built in tools like summarization, translation, etc.
- **Agents (Plan & Execute)** - Can create agents to follow prompts and execute plans.

### 2. [GPT Researcher on GitHub](https://github.com/assafelovic/gpt-researcher)
- Could potentially implement web scraping functions using this tool.

## General Flow:

1. **User Interface**:
   - User selects an IPO they are interested in analyzing via a Streamlit app.
   
2. **Agent Activities**:
   - Get IPO details such as offer price, valuation, etc., from filings.
   - Research competitors in the same industry.
   - Identify and analyze IPOs in a similar sector or category that were recently launched.
   - Aggregate relevant news articles related to the company and its industry.
   - Conduct fundamental analysis using available financials.
   - Rate different aspects of the IPO based on data.
   
3. **Output**:
   - The agent compiles the analysis into a comprehensive report, presenting the results in a readable format.
   - User retrieves the report from the Streamlit app and gains insights about the selected IPO.

## Steps to Implement:

1. **Setup**:
   - Ensure all required libraries and frameworks are installed and accessible.
   - Set up access to all data sources (API keys, access permissions).

2. **Development**:
   - Design the Streamlit interface for user interactions.
   - Create functions to fetch IPO details using web scraping methods.
   - Implement the data collection, analysis, and scoring processes.
   - Use GPT Researcher or similar tools to further automate and enhance data gathering and analysis.

3. **Testing**:
   - Test the system with multiple IPOs.
   - Ensure the report is comprehensive and informative.
   - Optimize for speed and accuracy.
   
4. **Deployment**:
   - Host the Streamlit app on a suitable platform.
   - Provide necessary documentation or user guide.

## Final Thoughts:
For optimal results, it is essential to continuously update the database of IPOs, competitors, and news sources. Additionally, regular checks on the tools and frameworks will ensure that the analysis remains accurate and timely.
