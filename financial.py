from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set API key for Groq
Groq.api_key = os.getenv("GROQ_API_KEY")
if not Groq.api_key:
    raise ValueError("GROQ_API_KEY not found. Please check your .env file.")

web_search_agent=Agent(
    name="Web search Agent",
    role="Search the web for the information",
    model=Groq(id="llama3-groq-70b-8192-tool-use-preview"),
    tools=[DuckDuckGo()],
    instructions=["Always include sources"],
    show_tool_calls=True,
    markdown=True,

)

## Financial agent
finance_agent=Agent(
    name="Finance AI Agent",
    model=Groq(id="llama3-groq-70b-8192-tool-use-preview"),
    tools=[YFinanceTools(stock_price=True, analyst_recommendations=True, stock_fundamentals=True, company_news=True, key_financial_ratios=True),],
    show_tool_calls=True,
    description="You are an investment analyst that researches stock prices, analyst recommendations, and stock fundamentals.",
    instructions=["Use tables to display the data"],
    markdown=True,
)

## Multi AI agent

multi_ai_agent=Agent(
    team=[web_search_agent,finance_agent],
    model=Groq(id="llama3-groq-70b-8192-tool-use-preview"),
    instructions=["Always include sources","Use table to display the data"],
    show_tool_calls=True,
    markdown=True,
)

## Multi AI Agent output

multi_ai_agent.print_response("Summarize analyst recommendation, stock fundamentals and share the latest news for NVDA", stream=True)