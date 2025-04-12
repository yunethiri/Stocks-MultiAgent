# main.py

import os
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

# Import your agents
from intent import IntentAgent
from output import OutputAgent
from memory import MemoryAgent
from sentiment import SentimentAgent
from rag import RAGAgent as DocumentAgent
from visualisation import StockVisualizer

# Load environment variables
load_dotenv()

app = FastAPI(title="Apple Stock Analysis Multi-Agent System")

# Initialize core agents
intent_agent = IntentAgent()
output_agent = OutputAgent()
memory_agent = MemoryAgent()

# Set output and memory agents for intent agent
intent_agent.set_output_agent(output_agent)
intent_agent.set_memory_agent(memory_agent)

# Initialize specialized agents
sentiment_agent = SentimentAgent()
document_agent = DocumentAgent()  # This is your RAG agent
visualisation_agent = StockVisualizer()

# Register agents with intent_agent
intent_agent.register_agent("sentiment", sentiment_agent)
intent_agent.register_agent("document", document_agent)
intent_agent.register_agent("visualisation", visualisation_agent)

class QueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = None

class QueryResponse(BaseModel):
    response: str
    sources: Optional[List[Dict[str, Any]]] = None
    session_id: str
    debug_info: Optional[Dict[str, Any]] = None

@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    try:
        # Generate session ID if not provided
        session_id = request.session_id
        if not session_id:
            session_id = memory_agent.get_session_id()
            
        # Process the query through the intent agent
        result = intent_agent.process_query(request.query, session_id)
        return JSONResponse(content=result)
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "error": str(e),
                "response": "An error occurred while processing your request. Please try again.",
                "session_id": request.session_id or "error_session"
            }
        )

@app.get("/history/{session_id}")
async def get_history(session_id: str):
    try:
        history = memory_agent.get_chat_history(session_id)
        return JSONResponse(content={"session_id": session_id, "history": history})
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

@app.delete("/history/{session_id}")
async def clear_history(session_id: str):
    try:
        success = memory_agent.clear_history(session_id)
        return JSONResponse(content={"success": success})
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