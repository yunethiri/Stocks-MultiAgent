from typing import Dict, List, Any, Optional, Annotated, TypedDict
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langgraph.graph import StateGraph, END
import json
import uuid
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
        self.sessions = {}  # Store session data
        self.graph = self._build_graph()
        
    def register_agent(self, agent_id: str, agent):
        """Register an agent with the intent agent"""
        self.registered_agents[agent_id] = agent
        
    def _build_graph(self):
        """Build the LangGraph for intent processing"""
        # Define the graph
        workflow = StateGraph(IntentAgentState)
        
        # Add nodes
        workflow.add_node("classify_intent", self._classify_intent)
        workflow.add_node("delegate_tasks", self._delegate_tasks)
        workflow.add_node("aggregate_responses", self._aggregate_responses)
        workflow.add_node("handle_error", self._handle_error)
        
        # Define edges
        workflow.add_edge("classify_intent", "delegate_tasks")
        workflow.add_edge("delegate_tasks", "aggregate_responses")
        workflow.add_edge("aggregate_responses", END)
        
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
    
    def _aggregate_responses(self, state: IntentAgentState) -> IntentAgentState:
        """Aggregate responses from all agents into a final response"""
        try:
            if state.get("error"):
                return state
                
            # Prepare the aggregation prompt with all agent responses
            agent_responses_text = ""
            for agent_id, response in state["agent_responses"].items():
                # Format may differ between agents, handle accordingly
                if isinstance(response, dict) and "response" in response:
                    agent_responses_text += f"\n\n{agent_id.upper()} AGENT RESPONSE:\n{response['response']}"
                elif isinstance(response, str):
                    agent_responses_text += f"\n\n{agent_id.upper()} AGENT RESPONSE:\n{response}"
                else:
                    # Handle more complex response structures
                    agent_responses_text += f"\n\n{agent_id.upper()} AGENT RESPONSE:\n{json.dumps(response, indent=2)}"
            
            # If we have an "out_of_scope" response, use it directly
            if "out_of_scope" in state["agent_responses"]:
                state["final_response"] = state["agent_responses"]["out_of_scope"]["response"]
                return state
            
            # If we have just one agent response, we might use it directly
            if len(state["agent_responses"]) == 1:
                agent_id = list(state["agent_responses"].keys())[0]
                response = state["agent_responses"][agent_id]
                
                # If it's already a well-formatted string, use it directly
                if isinstance(response, dict) and "response" in response:
                    state["final_response"] = response["response"]
                    return state
            
            # Otherwise, synthesize a response from multiple agents
            prompt = ChatPromptTemplate.from_template("""
            You are an expert financial analyst specializing in Apple (AAPL) stock.
            Synthesize the following information from different analysis agents into a coherent, comprehensive response.
            
            USER QUERY: {query}
            
            {agent_responses}
            
            Create a well-structured, professional response that directly addresses the user's query.
            Be concise but thorough, highlighting the most important insights from each agent.
            If there are conflicting perspectives, acknowledge them and provide a balanced view.
            """)
            
            # Run the aggregation chain
            chain = prompt | self.llm
            result = chain.invoke({
                "query": state["query"],
                "agent_responses": agent_responses_text
            })
            
            state["final_response"] = result.content
            
            # Update chat history
            self._update_chat_history(state)
            
            return state
        except Exception as e:
            state["error"] = f"Response aggregation failed: {str(e)}"
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
    
    def _update_chat_history(self, state: IntentAgentState):
        """Update the chat history for the session"""
        session_id = state["session_id"]
        
        if session_id not in self.sessions:
            self.sessions[session_id] = {
                "chat_history": []
            }
        
        # Add the current exchange to history
        self.sessions[session_id]["chat_history"].append({
            "user": state["query"],
            "assistant": state["final_response"],
            "timestamp": datetime.now().isoformat()
        })
        
        # Update state with full history
        state["chat_history"] = self.sessions[session_id]["chat_history"]
    
    def process_query(self, query: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Process a user query through the intent agent pipeline"""
        # Generate session ID if not provided
        if not session_id:
            session_id = str(uuid.uuid4())
        
        # Get chat history if available
        chat_history = []
        if session_id in self.sessions:
            chat_history = self.sessions[session_id]["chat_history"]
        
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
        
        # Process through the graph
        result = self.graph.invoke(initial_state)
        
        # Prepare the response
        response = {
            "response": result["final_response"],
            "session_id": session_id
        }
        
        # Add sources if available (particularly from RAG agent)
        if "document" in result.get("agent_responses", {}):
            doc_response = result["agent_responses"]["document"]
            if isinstance(doc_response, dict) and "sources" in doc_response:
                response["sources"] = doc_response["sources"]
        
        # Include debug info if in development mode
        if os.getenv("ENVIRONMENT") == "development":
            response["debug_info"] = result["debug_info"]
            
        return response