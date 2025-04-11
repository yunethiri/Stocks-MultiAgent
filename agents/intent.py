# core/intent_agent.py

from typing import Dict, List, Any, Optional, TypedDict
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langgraph.graph import StateGraph, END
from datetime import datetime
import json
from langchain_community.tools import BraveSearch
import os
import time
from openai import OpenAI

openai_api_key = os.getenv("OPENAI_API_KEY")  # Use OpenAI API key
client = OpenAI(api_key = openai_api_key)

# --- Chat response generator function ---
def generate_chat_response(prompt: str) -> str:
    """
    Generate a detailed answer from the LLM (e.g., OpenAI's ChatCompletion) using the provided prompt.
    This prompt is expected to incorporate search results and citations (such as title and link).
    """
    # In your actual app, you might include additional context such as stock data, etc.
    # For this example, we'll keep it simple.
    messages = [{
        "role": "user",
        "content": prompt
    }]

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=1000
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error occurred while generating answer: {str(e)}"
    
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
        # Ensure we import both START and END
        from langgraph.graph import START, END
        workflow = StateGraph(IntentAgentState)
        
        # Add nodes for intent classification and task delegation
        workflow.add_node("classify_intent", self._classify_intent)
        workflow.add_node("delegate_tasks", self._delegate_tasks)
        
        # Add the entry point edge from START to "classify_intent"
        workflow.add_edge(START, "classify_intent")
        
        # Define the rest of the workflow edges.
        workflow.add_edge("classify_intent", "delegate_tasks")
        workflow.add_edge("delegate_tasks", END)
        
        return workflow.compile()
    
    def _classify_intent(self, state: IntentAgentState) -> IntentAgentState:
        """Classify the user's intent using a prompt chain and update the state."""
        try:
            prompt = ChatPromptTemplate.from_template("""
            You are an intent classifier for a stock analysis system specialized in Apple Inc. (AAPL).
            Based on the user query, classify the intent and determine which specialized agents should handle it.
            You only have data in the year 2024. If the query falls out of this range, classify the intent to out_of_scope.
            
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
    
    def _perform_web_search(
        self,
        query: str,
        country: str = "US",
        search_lang: str = "en",
        count: int = 20,
        # offset: int = 0,
        spellcheck: bool = True,
        extra_snippets: bool = True,
        summary: bool = True,
        combine_response: bool = True
    ) -> Dict[str, Any]:
        """
        Uses the web search agent to gather information from the internet,
        formats the results (including title, link, and snippet for each hit),
        and then uses an LLM to generate a comprehensive answer that cites the sources.
        It enforces a rate limit of 1 query per second.
        """
        try:
            # Retrieve your Brave API key from the environment.
            brave_key = os.getenv("BRAVE_AI_API_KEY")
            if not brave_key:
                raise ValueError("Brave API key not found in environment variables.")

            # Build the search keyword arguments
            search_kwargs = {
                "q": query,
                "country": country,
                "search_lang": search_lang,
                "count": count,
                # "offset": offset,
                # "safesearch": safesearch,
                # "text_decorations": 1 if text_decorations else 0,
                "spellcheck": 1 if spellcheck else 0,
                "extra_snippets": 1 if extra_snippets else 0,
                "summary": 1 if summary else 0
            }

            search = BraveSearch.from_api_key(api_key=brave_key, search_kwargs=search_kwargs)
            raw_result = search.run(query)
            
            # Enforce the 1 query per second rate limit.
            time.sleep(1)
            
        except Exception as e:
            raw_result = f"Web search error: {str(e)}"
        
        final_answer = None
        # If combine_response is True, use llm to produce a comprehensive answer
        if combine_response:
            try:
                # Attempt to parse raw_result as JSON
                try:
                    parsed_results = json.loads(raw_result)
                except Exception:
                    parsed_results = raw_result
                
                # Format results into a combined template.
                formatted_results = ""
                if isinstance(parsed_results, list):
                    for item in parsed_results:
                        title = item.get("title", "No title")
                        link = item.get("link", "No link")
                        snippet = item.get("snippet", "No snippet available")
                        formatted_results += f"Title: {title}\nLink: {link}\nSnippet: {snippet}\n\n"
                else:
                    formatted_results = str(parsed_results)
                
                # Build the prompt using the formatted search results.
                prompt_template = (
                    "Using the following web search results, generate a detailed answer to the query '{query}'.\n"
                    "Present your response in a series of numbered summary, up to a maximum of 4. Ensure that each summary answers the query"
                    "and include repliable sources in the following format: [Source: <Source Name>](<URL>).\n\n"
                    "For example:\n"
                    "1. Baymax! is an animated superhero series that premiered on Disney+ on June 29, 2022. [Source: Wikipedia - Baymax!](https://en.wikipedia.org/wiki/Baymax!)\n"
                    "2. The series features returning voice actors from Big Hero 6 along with new cast members. [Source: IMDb - Baymax!](https://www.imdb.com/title/tt13622958/)\n\n"
                    "Now, using the web search results below, provide your answer:\n\n"
                    "{formatted_results}"
            )
                chat_prompt = prompt_template.format(query=query, formatted_results=formatted_results)
                # Generate the final answer from the LLM.
                final_answer = generate_chat_response(chat_prompt)
            except Exception as e:
                final_answer = f"Error generating combined answer: {str(e)}"
        
        return final_answer
            

    
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
            # Dispatch to each registered agent.
            for agent_id in agents_to_use:
                if agent_id in self.registered_agents:
                    agent = self.registered_agents[agent_id]
                    # Depending on your agent's implementation, you might call process() or process_query()
                    try:
                        # If the agent provides a process_query method (for example, a RAG agent), use that.
                        if hasattr(agent, "process_query"):
                            response = agent.process_query(state["query"])
                        else:
                            # Otherwise, use the generic process method.
                            response = agent.process(state["query"])
                        state["agent_responses"][agent_id] = response
                        state["debug_info"][f"delegated_to_{agent_id}"] = f"Called {agent_id} agent successfully."
                    except Exception as e:
                        # If an agent call fails, log that error.
                        state["agent_responses"][agent_id] = {"response": f"Error calling {agent_id}: {str(e)}"}
                        state["debug_info"][f"error_{agent_id}"] = str(e)
                else:
                    # If an agent isn't registered, log that as well.
                    state["agent_responses"][agent_id] = {"response": f"{agent_id} agent is not registered."}
                    state["debug_info"][f"error_{agent_id}"] = f"{agent_id} not found in registered agents."
                    
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


##for intent agent testing
# if __name__ == '__main__':
#     from dotenv import load_dotenv
#     load_dotenv() 

#     # Create an instance of the IntentAgent.
#     agent = IntentAgent(model_name="gpt-4o")
#     test_query = "how to survive without watre for 5 days?"
#     combined_response = agent._perform_web_search(test_query)
    
#     print("Raw Web Search Results:")
#     print(json.dumps(combined_response["raw_results"], indent=2))

#     print("\nGenerated Comprehensive Answer:")
#     print(combined_response["final_answer"])




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
