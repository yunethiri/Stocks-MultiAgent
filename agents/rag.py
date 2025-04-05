import os
from typing import Dict, List, Optional, Any, Tuple
from pydantic import BaseModel, Field
import qdrant_client
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema import Document
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolExecutor
from langgraph.checkpoint import JsonCheckpoint
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import HumanMessage, AIMessage
    
class RAGAgentState(BaseModel):
    """Represents the state of our financial RAG agent."""
    query: str = Field(description="The original user query")
    context: List[Document] = Field(default_factory=list, description="Retrieved context from vector store")
    financial_entities: List[Dict] = Field(default_factory=list, description="Extracted financial entities")
    response: Optional[str] = Field(default=None, description="The final response to the user")
    chat_history: List[Dict] = Field(default_factory=list, description="Chat history")
    intermediate_steps: List[Dict] = Field(default_factory=list, description="Intermediate steps for debugging")
    error: Optional[str] = Field(default=None, description="Error message if any")
    collection_choice: Optional[str] = Field(default=None, description="Which collection to query")

class RAGAgent:
    def __init__(self, qdrant_client, collection_names, model_name="gpt-4o"):
        """
        Initialize the Financial RAG Agent.
        
        Args:
            qdrant_client: Initialized Qdrant client
            collection_names: List of available collection names
            api_key: OpenAI API key
        """
        # Initialize LLM
        self.llm = ChatOpenAI(temperature=0, model=model_name)

        # Create vector stores for each collectio
        self.qdrant_client = qdrant_client
        self.collection_names = collection_names
        self.vector_stores = {
            name: QdrantVectorStore(client=qdrant_client, collection_name=name)
            for name in collection_names
        }

        # Initialise memory and output agents
        self.memory_agent = None
        self.output_agent = None
        self.graph = self._build_graph()
        
    def set_memory_agent(self, memory_agent):
        """Set the memory agent"""
        self.memory_agent = memory_agent
        
    def set_output_agent(self, output_agent):
        """Set the output agent"""
        self.output_agent = output_agent
    
    def build_graph(self):
        """Build the LangGraph workflow for the financial RAG agent."""
        # Define the graph
        workflow = StateGraph(RAGAgentState)
        
        # Add nodes
        workflow.add_node("collection_selector", self.select_collection)
        workflow.add_node("retriever", self.retrieve_context)
        workflow.add_node("entity_extractor", self.extract_financial_entities)
        workflow.add_node("response_generator", self.generate_response)
        
        # Define edges
        workflow.add_edge("collection_selector", "retriever")
        workflow.add_edge("retriever", "entity_extractor")
        workflow.add_edge("entity_extractor", "response_generator")
        workflow.add_edge("response_generator", END)
        
        # Set the entry point
        workflow.set_entry_point("collection_selector")
        
        # Create checkpoint for persistence
        checkpointer = JsonCheckpoint(os.path.join(os.getcwd(), "checkpoints"))
        
        # Compile the graph
        return workflow.compile(checkpointer=checkpointer)

    def select_collection(self, state: RAGAgentState) -> RAGAgentState:
        """Determine which collection(s) to query based on the user query."""
        try:
            prompt = ChatPromptTemplate.from_template(
                """You are a financial data assistant that helps determine which data source to query.
                
                Available collections:
                - financial_news: Recent financial news articles
                - aapl_10k_10q_forms: SEC filings including 10-K and 10-Q forms
                - earnings_calls: Transcripts from company earnings calls
                
                User query: {query}
                
                Based on this query, which ONE collection should I search to provide the most relevant information?
                Reply with ONLY ONE of: "financial_news", "aapl_10k_10q_forms", or "earnings_calls".
                """
            )
            
            chain = prompt | self.llm | StrOutputParser()
            collection = chain.invoke({"query": state.query})
            
            # Validate the collection name
            if collection not in self.collection_names:
                collection = self.collection_names[0]  # Default to first collection
            
            # Update state
            state.collection_choice = collection
            state.intermediate_steps.append({"action": "select_collection", "result": collection})
            
            return state
        except Exception as e:
            state["error"] = f"Collection selection failed: {str(e)}"
            return state
    
    def retrieve_context(self, state: RAGAgentState) -> RAGAgentState:
        """Retrieve relevant documents from the selected collection."""
        try:
            # Get the vector store for the selected collection
            vector_store = self.vector_stores[state.collection_choice]
            
            # Retrieve documents
            docs = vector_store.similarity_search(
                state.query,
                k=5  # Retrieve top 5 documents
            )
            
            # Update state
            state.context = docs
            state.intermediate_steps.append({
                "action": "retrieve_context", 
                "collection": state.collection_choice,
                "num_docs": len(docs)
            })
            
        except Exception as e:
            state.error = f"Error during retrieval: {str(e)}"
            
        return state