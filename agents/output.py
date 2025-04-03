# core/output_agent.py

from typing import Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
import json
import os

class OutputAgent:
    def __init__(self, model_name="gpt-4o"):
        self.llm = ChatOpenAI(temperature=0, model=model_name)
        
    def aggregate_responses(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Aggregate responses from all agents into a final coherent response"""
        try:
            if state.get("error"):
                return {
                    "response": state["final_response"],
                    "session_id": state["session_id"]
                }
                
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
                return {
                    "response": state["agent_responses"]["out_of_scope"]["response"],
                    "session_id": state["session_id"]
                }
            
            # If we have just one agent response, we might use it directly
            if len(state["agent_responses"]) == 1:
                agent_id = list(state["agent_responses"].keys())[0]
                response = state["agent_responses"][agent_id]
                
                # If it's already a well-formatted string, use it directly
                if isinstance(response, dict) and "response" in response:
                    return {
                        "response": response["response"],
                        "session_id": state["session_id"],
                        "sources": response.get("sources")
                    }
            
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
            
            final_response = {
                "response": result.content,
                "session_id": state["session_id"]
            }
            
            # Add sources if available (particularly from RAG agent)
            if "document" in state.get("agent_responses", {}):
                doc_response = state["agent_responses"]["document"]
                if isinstance(doc_response, dict) and "sources" in doc_response:
                    final_response["sources"] = doc_response["sources"]
            
            # Include debug info if in development mode
            if os.getenv("ENVIRONMENT") == "development":
                final_response["debug_info"] = state["debug_info"]
                
            return final_response
            
        except Exception as e:
            # Handle errors in the output agent
            return {
                "response": f"I encountered an issue while preparing your response. Please try again with a more specific question about Apple stock.",
                "session_id": state["session_id"],
                "error": str(e)
            }