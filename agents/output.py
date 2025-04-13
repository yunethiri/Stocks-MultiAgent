# core/output_agent.py

from typing import Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
import json
import os
from matplotlib.figure import Figure  # For handling matplotlib figures

class OutputAgent:
    def __init__(self, model_name="gpt-4o"):
        self.llm = ChatOpenAI(temperature=0, model=model_name)
        
    def aggregate_responses(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Aggregate responses from all agents into a final coherent response by synthesizing them with the query."""
        try:
            # 1) If an upstream error, return immediately using the existing final_response field (if set)
            if state.get("error"):
                return {
                    "response": state.get("final_response", ""),
                    "session_id": state.get("session_id")
                }
            
            # 2) Build text from all agent responses; capture any base64 image from visualisation agent
            agent_responses_text = ""
            image_b64 = None

            for agent_id, response in state.get("agent_responses", {}).items():
                # Skip if someone accidentally stored a Figure instance.
                if isinstance(response, Figure):
                    continue

                aid = agent_id.lower()

                # Special‑case visualisation agent
                if aid in ("visualisation", "visualization") and isinstance(response, dict):
                    summary = response.get("summary", "")
                    image_b64 = response.get("image_base64")
                    agent_responses_text += f"\n\nVISUALISATION AGENT SUMMARY:\n{summary}"
                    continue

                # Plain string responses
                if isinstance(response, str):
                    agent_responses_text += f"\n\n{agent_id.upper()} AGENT RESPONSE:\n{response}"
                # Dict responses with a "response" key
                elif isinstance(response, dict) and "response" in response:
                    agent_responses_text += (
                        f"\n\n{agent_id.upper()} AGENT RESPONSE:\n"
                        f"{response['response']}"
                    )
                # Dump other types as JSON
                else:
                    agent_responses_text += (
                        f"\n\n{agent_id.upper()} AGENT RESPONSE:\n"
                        f"{json.dumps(response, indent=2)}"
                    )
            
            # 3) Always synthesize a final response using the query and aggregated agent responses
            synthesis_prompt = ChatPromptTemplate.from_template("""
            You are an expert financial analyst specializing in Apple (AAPL) stock.
            Synthesize the following information from various analysis agents into a coherent, comprehensive response.

            USER QUERY: {query}

            AGENT RESPONSES: {agent_responses}
                                                                
            Instructions:
            1. If any agent responses clearly indicate the query is **not related to finance** (e.g., queries about cooking, travel, etc.), respond with:
            "The query doesn't seem to be related to finance. It appears to be about a non-financial topic (briefly specify the topic). Please ask any finance-related questions, and I’ll be happy to help!"
                                                                
            2. If the responses suggest the query **is finance-related but not about Apple** (e.g., related to Google or Tesla stocks), say:
            "Internal data is only available for Apple Inc. (AAPL). Therefore, the answer below is generated solely based on web search results:"

            3. In the above case, also provide a **summary of the key findings** from the web search results. 
            - Embed **clickable links** (if provided) in markdown format.
            - Preserve and display any **web links or references** given by the web search agent.
            - Write a clear summary first, then list the links below it.

            4. If the query is related to Apple and financial in nature, integrate all relevant insights from the agent responses into a well-structured answer. Do not mention the source (e.g., RAG, web search, etc.)—just provide a unified expert-level analysis.

            Keep your response:
            - Professional and easy to follow
            - Balanced when differing perspectives are present
            - Free of any redundant explanation about how you arrived at the answer
            """)
            
            chain = synthesis_prompt | self.llm
            output = chain.invoke({
                "query": state.get("query", ""),
                "agent_responses": agent_responses_text
            })
            
            # Extract the final text from the LLM output
            text = output.content if hasattr(output, "content") else output

            final_response = {
                "response": text,
                "session_id": state["session_id"]
            }
            
            # 4) Attach the visualisation image if available
            if image_b64:
                final_response["image_base64"] = image_b64

            # 5) Attach document sources if present (for RAG/document agent)
            doc_resp = state["agent_responses"].get("document", {})
            if isinstance(doc_resp, dict) and "sources" in doc_resp:
                final_response["sources"] = doc_resp["sources"]

            # 6) Optionally include debug info in development mode
            if os.getenv("ENVIRONMENT") == "development":
                final_response["debug_info"] = state.get("debug_info")
                
            return final_response

        except Exception as e:
            print("OutputAgent.aggregate_responses error:", repr(e))
            return {
                "response": (
                    "I encountered an issue while preparing your response. "
                    "Please try again with a more specific question about Apple stock."
                ),
                "session_id": state.get("session_id"),
                "error": str(e)
            }
