import os
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

# Import your agents
from agents.intent import IntentAgent
from agents.sentiment_agent.sentiment_agent import SentimentAgent
from agents.metrics_agent.metrics_agent import MetricsAgent
from agents.document_agent.document_agent import DocumentAgent
from agents.investment_agent.investment_agent import InvestmentAgent

# Load environment variables
load_dotenv()

app = FastAPI(title="Apple Stock Analysis Multi-Agent System")

# Initialize agents
intent_agent = IntentAgent()
sentiment_agent = SentimentAgent()
metrics_agent = MetricsAgent()
document_agent = DocumentAgent()  # This is your RAG agent
investment_agent = InvestmentAgent()

# Register agents with intent_agent
intent_agent.register_agent("sentiment", sentiment_agent)
intent_agent.register_agent("metrics", metrics_agent)
intent_agent.register_agent("document", document_agent)
intent_agent.register_agent("investment", investment_agent)

class QueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = None

class QueryResponse(BaseModel):
    response: str
    sources: Optional[List[Dict[str, Any]]] = None
    debug_info: Optional[Dict[str, Any]] = None

@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    try:
        # Process the query through the intent agent
        result = intent_agent.process_query(request.query, request.session_id)
        return JSONResponse(content=result)
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)