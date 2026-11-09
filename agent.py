"""LangChain advisor agent and tool wiring."""

# agent.py  (the important parts — full file is in the repo)
import shutil
from pathlib import Path
import pandas as pd
from langchain.agents import create_agent
from langchain_core.tools import Tool
from langchain_google_genai import ChatGoogleGenerativeAI
from rag import get_retriever
from checker import flag_transactions
from guardrails import make_safe

llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0)
retriever = get_retriever()
UPLOAD_PATH = "data/_current_upload.csv"     # where the redacted upload is staged

def market_data(ticker: str) -> str:                    # Tool 1 — yfinance
    import importlib
    yf = importlib.import_module("yfinance")
    hist = yf.Ticker(ticker.strip().upper()).history(period="1mo")
    if hist.empty:
        return f"No market data for {ticker}."
    price, month_ago = hist["Close"].iloc[-1], hist["Close"].iloc[0]
    change = (price - month_ago) / month_ago * 100
    return f"{ticker}: price ${price:.2f}, 1-month change {change:+.1f}%."

def research(query: str) -> str:                        # Tool 2 — RAG
    hits = retriever.invoke(query)
    return "\n\n".join(f"[source] {d.page_content}" for d in hits) or "No relevant documents."

def check_uploaded(_input: str) -> str:                 # Tool 3 — transaction checker
    if not Path(UPLOAD_PATH).exists():
        return "No transactions were uploaded for this question."
    flagged = flag_transactions(pd.read_csv(UPLOAD_PATH))
    if flagged.empty:
        return "No suspicious transactions found."
    return "Suspicious transactions:\n" + flagged.to_string(index=False)

TOOLS = [
    Tool(name="MarketData", func=market_data,
         description="Get live price and 1-month trend for a stock ticker like AAPL."),
    Tool(name="Research", func=research,
         description="Look up finance documents to ground an answer about a company or topic."),
    Tool(name="CheckTransactions", func=check_uploaded,
         description="Scan the user's uploaded transactions for suspicious activity."),
]

SYSTEM_PROMPT = (
    "You are AdvisorIQ, a wealth-management assistant. Use the tools to ground your "
    "answers in real data. Be concise and never promise a specific return."
)

agent = create_agent(llm, TOOLS, system_prompt=SYSTEM_PROMPT)

def run_advisor(question: str, csv_path: str | None = None) -> str:
    """The one function app.py calls: stage upload -> run agent -> apply guardrail."""
    if csv_path:
        shutil.copyfile(csv_path, UPLOAD_PATH)
    else:
        Path(UPLOAD_PATH).unlink(missing_ok=True)   # never reuse a previous upload

    result = agent.invoke({"messages": [{"role": "user", "content": question}]})
    answer = result["messages"][-1].text

    # Grounding check: did the agent actually consult the library? Read the real
    # Research tool output from the message history instead of guessing.
    sources = [m.content for m in result["messages"]
               if getattr(m, "name", "") == "Research" and "No relevant" not in m.content]
    return make_safe(answer, sources)