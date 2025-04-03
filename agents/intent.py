# core/intent_agent.py

from typing import Dict, List, Any, Optional, TypedDict
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langgraph.graph import StateGraph, END
import json
from datetime import datetime

class IntentClassification(BaseModel):
    intent: str = Field(description="The classified intent")
    confidence: float = Field(description="Confidence score between 0 and 1")
    agents_needed: List[str] = Field(description="List of agent IDs needed to fulfill this request")
    explanation: str = Field(description="Explanation of why this intent was chosen")

class IntentAgentState(TypedDict):
    query: str
    session_id: str
    chat_history: List[Dict[str, Any]]
    intent: Optional[IntentClassification]
    agent_responses: Dict[str, Any]
    final_response: Optional[str]
    error: Optional[str]
    debug_info: Dict[str, Any]

class IntentAgent:
    def __init__(self, model_name="gpt-4o"):
        self.llm = ChatOpenAI(temperature=0, model=model_name)
        self.registered_agents = {}
        self.memory_agent = None
        self.output_agent = None
        self.graph = self._build_graph()
        
    def register_agent(self, agent_id: str, agent):
        """Register an agent with the intent agent"""
        self.registered_agents[agent_id] = agent
        
    def set_memory_agent(self, memory_agent):
        """Set the memory agent"""
        self.memory_agent = memory_agent
        
    def set_output_agent(self, output_agent):
        """Set the output agent"""
        self.output_agent = output_agent
        
    def _build_graph(self):
        """Build the LangGraph for intent processing"""
        # Define the graph
        workflow = StateGraph(IntentAgentState)
        
        # Add nodes
        workflow.add_node("classify_intent", self._classify_intent)
        workflow.add_node("delegate_tasks", self._delegate_tasks)
        workflow.add_node("handle_error", self._handle_error)
        
        # Define edges
        workflow.add_edge("classify_intent", "delegate_tasks")
        workflow.add_edge("delegate_tasks", END)
        
        # Error handling edges
        workflow.add_edge_from_parent("handle_error")
        workflow.add_edge("handle_error", END)
        
        # Compile the graph
        return workflow.compile()
    
    def _classify_intent(self, state: IntentAgentState) -> IntentAgentState:
        """Classify the user's intent and determine which agents to use"""
        try:
            # Define the prompt for intent classification
            prompt = ChatPromptTemplate.from_template("""
            You are an intent classifier for a stock analysis system specialized in Apple Inc. (AAPL).
            Based on the user query, classify the intent and determine which specialized agents should handle it.
            
            Available agents:
            - sentiment: Analyzes current stock sentiment from news and social media
            - metrics: Summarizes financial metrics (revenue, cash flow, margins)
            - document: RAG agent for analyzing documents like earnings transcripts and analyst reports
            - investment: Generates investment thesis (buy/sell/hold recommendations)
            
            User Query: {query}
            
            Classify the intent into one of:
            - stock_sentiment: Questions about current market sentiment
            - financial_metrics: Questions about financial performance metrics
            - document_analysis: Questions about earnings calls, analyst opinions, or SEC filings
            - investment_recommendation: Requests for buy/sell/hold recommendations
            - general_query: General questions about Apple stock not fitting other categories
            - out_of_scope: Queries not related to Apple stock analysis
            
            Return your analysis as a JSON with these fields:
            - intent: The classified intent category
            - confidence: Confidence score between 0 and 1
            - agents_needed: List of agent IDs needed (can be multiple)
            - explanation: Brief explanation of your classification
            """)
            
            # Set up the chain
            chain = prompt | self.llm | JsonOutputParser()
            
            # Run the chain
            result = chain.invoke({"query": state["query"]})
            
            # Update state
            state["intent"] = IntentClassification(**result)
            state["debug_info"]["intent_classification"] = result
            
            return state
        except Exception as e:
            state["error"] = f"Intent classification failed: {str(e)}"
            return state
    
    def _delegate_tasks(self, state: IntentAgentState) -> IntentAgentState:
        """Delegate tasks to appropriate agents based on intent"""
        try:
            if state.get("error"):
                return state
                
            intent_data = state["intent"]
            agents_to_use = intent_data.agents_needed
            
            # Initialize responses dictionary
            state["agent_responses"] = {}
            
            # Special handling for out of scope intents
            if intent_data.intent == "out_of_scope":
                state["agent_responses"]["out_of_scope"] = {
                    "response": "I'm sorry, but your query appears to be outside the scope of my Apple stock analysis capabilities. I can help with questions about Apple's stock sentiment, financial metrics, documents analysis, or investment recommendations."
                }
                return state
            
            # Process with each required agent
            for agent_id in agents_to_use:
                if agent_id in self.registered_agents:
                    agent = self.registered_agents[agent_id]
                    
                    # Log debug info
                    state["debug_info"][f"delegating_to_{agent_id}"] = {
                        "timestamp": datetime.now().isoformat(),
                        "query": state["query"]
                    }
                    
                    # Call the agent
                    if agent_id == "document":  # Special handling for RAG agent
                        response = agent.process_query(state["query"])
                        state["agent_responses"][agent_id] = response
                    else:
                        response = agent.process(state["query"])
                        state["agent_responses"][agent_id] = response
                        
            return state
        except Exception as e:
            state["error"] = f"Task delegation failed: {str(e)}"
            return state
    
    def _handle_error(self, state: IntentAgentState) -> IntentAgentState:
        """Handle any errors that occurred during processing"""
        error_msg = state.get("error", "An unknown error occurred")
        
        # Generate a user-friendly error message
        prompt = ChatPromptTemplate.from_template("""
        You are a helpful assistant. An error occurred while processing a user's query about Apple stock.
        
        Technical error: {error}
        
        Provide a friendly, helpful response that:
        1. Acknowledges the issue
        2. Explains in simple terms what might have gone wrong
        3. Suggests how the user might reformulate their query
        
        Do not expose technical details or stack traces.
        """)
        
        chain = prompt | self.llm
        result = chain.invoke({"error": error_msg})
        
        state["final_response"] = result.content
        return state
    
    def process_query(self, query: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Process a user query through the intent agent pipeline"""
        # Get chat history from memory agent
        chat_history = []
        if self.memory_agent:
            chat_history = self.memory_agent.get_chat_history(session_id)
        
        # Initialize state
        initial_state = IntentAgentState(
            query=query,
            session_id=session_id,
            chat_history=chat_history,
            intent=None,
            agent_responses={},
            final_response=None,
            error=None,
            debug_info={
                "timestamp": datetime.now().isoformat(),
                "session_id": session_id
            }
        )
        
        # Process through the intent graph
        result = self.graph.invoke(initial_state)
        
        # Forward to output agent for response aggregation
        if self.output_agent:
            final_result = self.output_agent.aggregate_responses(result)
        else:
            # Fallback if output agent not set
            final_result = {
                "response": "Output agent not configured",
                "session_id": session_id
            }
        
        # Update memory with the exchange
        if self.memory_agent:
            self.memory_agent.add_exchange(
                session_id=session_id,
                user_message=query,
                assistant_message=final_result["response"]
            )
            
        return final_result