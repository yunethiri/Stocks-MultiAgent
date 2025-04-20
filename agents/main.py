import os
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from openai import OpenAI
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

client = OpenAI(api_key = os.getenv("OPENAI_API_KEY"))

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
async def process_query(query: str, session_id: str = None):
    try:
        # Generate session ID if not provided
        if not session_id:
            session_id = memory_agent.get_session_id()
            
        # Process the query through the intent agent
        result = intent_agent.process_query(query, session_id)
        return JSONResponse(content=result)
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "error": str(e),
                "response": "An error occurred while processing your request. Please try again.",
                "session_id": session_id or "error_session"
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

# Response model for chat naming
class ChatNameResponse(BaseModel):
    name: str
    
@app.post("/generate_chat_name", response_model=ChatNameResponse)
async def generate_chat_name(request: str):
    """Generate a concise, descriptive name for the chat based on the first user prompt"""
    try:

        # Call OpenAI to generate a name
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system", 
                    "content": "You are a helpful assistant that creates concise, descriptive titles for conversations. "
                              "Create a short, specific title (3-5 words max) based on the user's first message. "
                              "The title should clearly indicate the topic without being generic. "
                              "Don't use quotes or other formatting. Just return the title text."
                },
                {"role": "user", "content": request}
            ],
            max_tokens=20,
            temperature=0.7,
        )
        
        # Extract and clean the title
        title = response.choices[0].message.content.strip()
        
        # Ensure it's not too long
        max_length = 30
        if len(title) > max_length:
            title = title[:max_length-3] + "..."
            
        return {"name": title}
    
    except Exception as e:
        print(e)
        # Fallback to a simpler method if the API call fails
        words = request.split()
        simple_title = " ".join(words[:3]) + ("..." if len(words) > 3 else "")
        return {"name": simple_title}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)