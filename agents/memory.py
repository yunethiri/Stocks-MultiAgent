# core/memory_agent.py

from typing import Dict, List, Any, Optional
import uuid
from datetime import datetime

class MemoryAgent:
    def __init__(self, max_history_length=20):
        self.sessions = {}  # Dictionary to store session data
        self.max_history_length = max_history_length
        
    def get_session_id(self) -> str:
        """Generate a new session ID"""
        return str(uuid.uuid4())
        
    def get_chat_history(self, session_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get chat history for a session"""
        if not session_id:
            return []
            
        if session_id not in self.sessions:
            self.sessions[session_id] = {"chat_history": []}
            
        return self.sessions[session_id]["chat_history"]
        
    def add_exchange(self, session_id: Optional[str], user_message: str, assistant_message: str) -> str:
        """Add an exchange to the chat history"""
        # Generate session ID if not provided
        if not session_id:
            session_id = self.get_session_id()
            
        # Initialize session if it doesn't exist
        if session_id not in self.sessions:
            self.sessions[session_id] = {"chat_history": []}
            
        # Add the exchange
        self.sessions[session_id]["chat_history"].append({
            "user": user_message,
            "assistant": assistant_message,
            "timestamp": datetime.now().isoformat()
        })
        
        # Trim history if it exceeds max length
        if len(self.sessions[session_id]["chat_history"]) > self.max_history_length:
            self.sessions[session_id]["chat_history"] = self.sessions[session_id]["chat_history"][-self.max_history_length:]
            
        return session_id
        
    def clear_history(self, session_id: str) -> bool:
        """Clear chat history for a session"""
        if session_id in self.sessions:
            self.sessions[session_id]["chat_history"] = []
            return True
        return False
        
    def delete_session(self, session_id: str) -> bool:
        """Delete a session entirely"""
        if session_id in self.sessions:
            del self.sessions[session_id]
            return True
        return False
        
    def get_session_summary(self, session_id: str) -> Dict[str, Any]:
        """Get a summary of session data"""
        if session_id not in self.sessions:
            return {"exists": False}
            
        history = self.sessions[session_id]["chat_history"]
        return {
            "exists": True,
            "exchanges": len(history),
            "first_timestamp": history[0]["timestamp"] if history else None,
            "last_timestamp": history[-1]["timestamp"] if history else None
        }