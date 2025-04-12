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
            If the response does not relate to Apple finance at all, make sure that the response contains:
            "If the query is unrelated to finance, respond as follows:\n"
                    "Example Query: 'How do I bake a chocolate cake?'\n"
                    "Response: 'The query doesn't seem to be related to finance. It looks like it's more about cooking. Feel free to ask any finance-related questions, and I'll be happy to help!'

            You are an expert financial analyst specializing in Apple (AAPL) stock.
            Synthesize the following information from different analysis agents into a coherent, comprehensive response.

            USER QUERY: {query}

            {agent_responses}

            Please provide a well-structured, professional answer that directly addresses the user's query.
            Be concise yet thorough, and if there are differing perspectives, provide a balanced view.
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
