# core/intent_agent.py

from typing import Dict, List, Any, Optional, TypedDict
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langgraph.graph import StateGraph, END
from datetime import datetime
import json

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
        # ChatOpenAI automatically reads your OPENAI_API_KEY from the environment.
        self.llm = ChatOpenAI(temperature=0, model=model_name)
        self.registered_agents = {}  # Other specialized agents can be registered if needed.
        self.memory_agent = None
        self.output_agent = None
        self.graph = self._build_graph()
        
    def register_agent(self, agent_id: str, agent):
        """Register an agent with the intent agent."""
        self.registered_agents[agent_id] = agent
        
    def set_memory_agent(self, memory_agent):
        """Set the memory agent."""
        self.memory_agent = memory_agent
        
    def set_output_agent(self, output_agent):
        """Set the output agent."""
        self.output_agent = output_agent
        
    def _build_graph(self):
        """Build the LangGraph for intent processing (without a dedicated error node)."""
        workflow = StateGraph(IntentAgentState)
        
        # Add nodes for intent classification and task delegation only.
        workflow.add_node("classify_intent", self._classify_intent)
        workflow.add_node("delegate_tasks", self._delegate_tasks)
        
        # Define the graph edges.
        workflow.add_edge("classify_intent", "delegate_tasks")
        workflow.add_edge("delegate_tasks", END)
        
        return workflow.compile()
    
    def _classify_intent(self, state: IntentAgentState) -> IntentAgentState:
        """Classify the user's intent using a prompt chain and update the state."""
        try:
            prompt = ChatPromptTemplate.from_template("""
            You are an intent classifier for a stock analysis system specialized in Apple Inc. (AAPL).
            Based on the user query, classify the intent and determine which specialized agents should handle it.
            
            Available agents:
            - sentiment: Analyzes current stock sentiment from news and social media
            - document: RAG agent for analyzing documents like earnings transcripts and analyst reports
            - visualisation: Creates visualisations of the stocks price for the requested period
            
            User Query: {query}
            
            Classify the intent into one of:
            - stock_sentiment: Questions about current market sentiment
            - document_analysis: Questions about earnings calls, analyst opinions, or SEC filings
            - visualisation: Requests for visualisation of stocks price for a given period
            - web_search: Handles general questions about Apple stock not fitting other categories 
            - out_of_scope: Queries not related to Apple stock analysis
            
            Return your analysis as a JSON with these fields:
            - intent: The classified intent category
            - confidence: Confidence score between 0 and 1
            - agents_needed: List of agent IDs needed (can be multiple)
            - explanation: Brief explanation of your classification
            """)
            
            chain = prompt | self.llm | JsonOutputParser()
            result = chain.invoke({"query": state["query"]})
            
            state["intent"] = IntentClassification(**result)
            state["debug_info"]["intent_classification"] = result
            return state
        except Exception as e:
            # If classification fails, register the error and let delegation handle it.
            state["error"] = f"Intent classification failed: {str(e)}"
            return state

    def _perform_web_search(self, query: str) -> Dict[str, Any]:
        """
        Perform a web search using ChatGPT's web search tool mode.
        This simulates the behavior described in the OpenAI docs for tool mode.
        In an actual configuration, ensure your account has access to this mode.
        """
        search_prompt = ChatPromptTemplate.from_template("""
        You are now using ChatGPT's web search tool. Please perform a web search for the following query and return your findings succinctly.
        
        Query: {query}
        """)
        chain = search_prompt | self.llm | JsonOutputParser()
        result = chain.invoke({"query": query})
        return {"response": result.get("response", f"No web search results for '{query}' were found.")}
    
    def _delegate_tasks(self, state: IntentAgentState) -> IntentAgentState:
        """
        Delegate tasks based on intent. If an error occurred or if the intent is out_of_scope,
        use the web search fallback. Otherwise, record the agents needed.
        """
        try:
            # If an error occurred during classification, fall back to web search.
            if state.get("error"):
                state["agent_responses"]["web_search"] = self._perform_web_search(state["query"])
                state["debug_info"]["delegated_to"] = "web_search fallback due to classification error"
                return state
                
            intent_data = state["intent"]
            
            # If the intent is out_of_scope, perform a web search.
            if intent_data.intent == "out_of_scope":
                state["agent_responses"]["web_search"] = self._perform_web_search(state["query"])
                state["debug_info"]["delegated_to"] = "web_search (out_of_scope)"
                return state
            
            # Otherwise, record which agents are needed.
            agents_to_use = intent_data.agents_needed
            state["debug_info"]["agents_needed"] = agents_to_use
            state["agent_responses"] = {}  # Placeholder for additional delegation.
            return state
        except Exception as e:
            # In case of an error during delegation, fall back to web search.
            state["error"] = f"Task delegation failed: {str(e)}"
            state["agent_responses"]["web_search"] = self._perform_web_search(state["query"])
            state["debug_info"]["delegated_to"] = "web_search fallback due to delegation error"
            return state
    
    def process_query(self, query: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Process a user query through the entire pipeline."""
        chat_history = self.memory_agent.get_chat_history(session_id) if self.memory_agent else []
        
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
        
        # Run the intent graph (classification followed by task delegation).
        state = self.graph.invoke(initial_state)
        
        # Forward state to the output agent for response aggregation.
        if self.output_agent:
            final_result = self.output_agent.aggregate_responses(state)
        else:
            final_result = {
                "response": "Output agent not configured",
                "session_id": session_id
            }
        
        # Update memory with the new exchange.
        if self.memory_agent:
            self.memory_agent.add_exchange(
                session_id=session_id,
                user_message=query,
                assistant_message=final_result["response"]
            )
            
        return final_result

# # Example usage (with dummy memory and output agents):

# if __name__ == "__main__":
#     # Dummy Memory Agent to store chat history.
#     class DummyMemoryAgent:
#         def __init__(self):
#             self.history = {}
#         def get_chat_history(self, session_id):
#             return self.history.get(session_id, [])
#         def add_exchange(self, session_id, user_message, assistant_message):
#             if session_id not in self.history:
#                 self.history[session_id] = []
#             self.history[session_id].append({
#                 "user": user_message,
#                 "assistant": assistant_message,
#                 "timestamp": datetime.now().isoformat()
#             })
    
#     # Dummy Output Agent aggregates responses by concatenating them.
#     class DummyOutputAgent:
#         def aggregate_responses(self, state: IntentAgentState) -> Dict[str, Any]:
#             responses = []
#             for key, resp in state["agent_responses"].items():
#                 responses.append(f"{key}: {resp.get('response', '')}")
#             aggregated = "\n".join(responses) if responses else "No response generated."
#             return {"response": aggregated, "session_id": state["session_id"]}
    
#     # Instantiate and configure the agent.
#     intent_agent = IntentAgent()
#     dummy_memory = DummyMemoryAgent()
#     dummy_output = DummyOutputAgent()
#     intent_agent.set_memory_agent(dummy_memory)
#     intent_agent.set_output_agent(dummy_output)
    
#     # Example query that is likely out of scope.
#     query = "Tell me about the latest trends in quantum computing."
#     session_id = "session_123"
    
#     final_response = intent_agent.process_query(query, session_id)
#     print("Final Response:")
#     print(final_response["response"])
