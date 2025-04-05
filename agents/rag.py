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
    collection_choice: Optional[str] = Field(default=None, description="Which collection to query")
    context: List[Document] = Field(default_factory=list, description="Retrieved context from vector store")
    financial_entities: List[Dict] = Field(default_factory=list, description="Extracted financial entities")
    response: Optional[str] = Field(default=None, description="The final response to the user")
    intermediate_steps: List[Dict] = Field(default_factory=list, description="Intermediate steps for debugging")
    error: Optional[str] = Field(default=None, description="Error message if any")
    source_documents: List[Dict] = Field(default_factory=list, description="Source documents with metadata")

class RAGAgent:
    def __init__(self, qdrant_client, collection_names, model_name="gpt-4o", api_key=None):
        """
        Initialize the Financial RAG Agent.
        
        Args:
            qdrant_client: Initialized Qdrant client
            collection_names: List of available collection names
            api_key: OpenAI API key
        """
        # Initialize LLM
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.model_name = model_name
        self.llm = ChatOpenAI(temperature=0, model=model_name, api_key=self.api_key)

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
        checkpointer = JsonCheckpoint(os.path.join(os.getcwd(), "checkpoints", "financial_rag"))
        
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
            state.source_documents = [
                {"content": doc.page_content, "metadata": doc.metadata}
                for doc in docs
            ]
            state.intermediate_steps.append({
                "action": "retrieve_context", 
                "collection": state.collection_choice,
                "num_docs": len(docs)
            })
            
            return state
        except Exception as e:
            state.error = f"Error during retrieval: {str(e)}"
            return state

    def extract_financial_entities(self, state: RAGAgentState) -> RAGAgentState:
        """Extract financial entities from the retrieved documents."""
        if not state.context:
            return state
            
        # Combine context for entity extraction
        context_text = "\n\n".join([doc.page_content for doc in state.context])
        
        prompt = ChatPromptTemplate.from_template(
            """You are a financial entity extraction specialist.
            
            Extract key financial entities from the following financial text:
            
            {context}
            
            Extract and return a JSON array of objects with the following properties:
            - entity_type: The type of entity (e.g., company, metric, stock_symbol, financial_term, person, date)
            - entity_name: The name of the entity
            - value: Any associated value or metric (if applicable)
            - sentiment: Positive, negative, or neutral (if applicable)
            
            Format your response as a valid JSON array, nothing else.
            """
        )
        
        try:
            from json import loads
            chain = prompt | self.llm | StrOutputParser()
            result = chain.invoke({"context": context_text})
            entities = loads(result)
            
            # Update state
            state.financial_entities = entities
            state.intermediate_steps.append({
                "action": "extract_entities",
                "num_entities": len(entities)
            })
            
        except Exception as e:
            state.error = f"Error during entity extraction: {str(e)}"
            state.financial_entities = []
            
        return state
    
    def generate_response(self, state: RAGAgentState) -> RAGAgentState:
        """Generate the final response using the LLM with the retrieved context."""
        # Prepare context snippets
        context_snippets = [f"Document {i+1}:\n{doc.page_content}\n" 
                           for i, doc in enumerate(state.context)]
        
        context_text = "\n".join(context_snippets)
        
        # Format entity information
        entity_info = ""
        if state.financial_entities:
            entity_info = "Key entities identified:\n"
            for entity in state.financial_entities:
                entity_info += f"- {entity.get('entity_name', 'Unknown')} ({entity.get('entity_type', 'Unknown')})"
                if entity.get('value'):
                    entity_info += f": {entity.get('value')}"
                if entity.get('sentiment'):
                    entity_info += f" [{entity.get('sentiment')}]"
                entity_info += "\n"
        
        prompt = ChatPromptTemplate.from_template(
            """You are a financial analysis assistant that provides accurate information based on the retrieved documents.
            
            User query: {query}
            
            Retrieved information:
            {context}
            
            {entity_info}
            
            Based on the retrieved information, provide a comprehensive response to the user's query.
            Be specific and cite information from the documents where appropriate.
            If the information is not sufficient to answer the query completely, acknowledge the limitations.
            
            Response:
            """
        )
        
        chain = prompt | self.llm | StrOutputParser()
        response = chain.invoke({
            "query": state.query,
            "context": context_text,
            "entity_info": entity_info
        })
        
        # Update state
        state.response = response
        
        return state
    
    def process_query(self, query: str) -> Dict:
        """Process a user query through the full graph and return the result."""
        # Create initial state
        initial_state = RAGAgentState(query=query)
        
        # Run the graph
        result = self.graph.invoke(initial_state)
        
        return {
            "response": result.response,
            "source_documents": result.source_documents[:3],
            "collection_used": result.collection_choice,
            "entities": result.financial_entities,
            "error": result.error
        }
    
        return output
    

    