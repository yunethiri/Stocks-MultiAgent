import os
import cohere
from typing import Dict, List, Optional
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain.schema import Document
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv
from openai import OpenAI 

load_dotenv()

class RAGAgentState(BaseModel):
    """Represents the state of our financial RAG agent."""

    query: str = Field(description="The original user query")
    collection_choice: Optional[str] = Field(
        default=None, description="Which collection to query"
    )
    context: List[Document] = Field(
        default_factory=list, description="Retrieved context from vector store"
    )
    financial_entities: List[Dict] = Field(
        default_factory=list, description="Extracted financial entities"
    )
    response: Optional[str] = Field(
        default=None, description="The final response to the user"
    )
    intermediate_steps: List[Dict] = Field(
        default_factory=list, description="Intermediate steps for debugging"
    )
    error: Optional[str] = Field(default=None, description="Error message if any")
    source_documents: List[Dict] = Field(
        default_factory=list, description="Source documents with metadata"
    )


class RAGAgent:
    def __init__(self, model_name="gpt-4o", api_key=None):
        """
        Initialize the Financial RAG Agent.

        Args:
            qdrant_client: Initialized Qdrant client
            collection_names: List of available collection names
            api_key: OpenAI API key
        """
        # Initialize LLM
        self.openai_api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model_name = model_name
        self.llm = ChatOpenAI(
            temperature=0, model=model_name, api_key=self.openai_api_key
        )

        # initliase db
        self.cohere_api_key = os.getenv("COHERE_API_KEY")
        self.cohere_client = cohere.ClientV2(self.cohere_api_key)
        self.cohere_model = "embed-english-v3.0"

        self.qdrant_client = QdrantClient(url="http://qdrant:6333")
        self.collection_names = [
            "financial_news",
            "earnings_calls",
            "aapl_10k_10q_forms",
        ]

        self.graph = self._build_graph()

    def _build_graph(self):
        """Build the LangGraph workflow for the financial RAG agent."""
        # Define the graph
        workflow = StateGraph(RAGAgentState)

        # Add nodes
        workflow.add_node("collection_selector", self._select_collection)
        workflow.add_node("retriever", self._retrieve_context)
        workflow.add_node("entity_extractor", self._extract_financial_entities)
        workflow.add_node("response_generator", self._generate_response)

        # Define edges
        workflow.add_edge("collection_selector", "retriever")
        workflow.add_edge("retriever", "entity_extractor")
        workflow.add_edge("entity_extractor", "response_generator")
        workflow.add_edge("response_generator", END)

        # Set the entry point
        workflow.set_entry_point("collection_selector")

        # Compile the graph
        return workflow.compile()

    def _select_collection(self, state: RAGAgentState) -> RAGAgentState:
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
            
            print(f"Collection selected: {collection}")

            # Update state
            state.collection_choice = collection
            state.intermediate_steps.append(
                {"action": "select_collection", "result": collection}
            )

            return state
        except Exception as e:
            state["error"] = f"Collection selection failed: {str(e)}"
            return state

    def _retrieve_context(self, state: RAGAgentState) -> RAGAgentState:
        """Retrieve relevant documents from the selected collection."""
        try:
            # Perform query embeddings
            query_embeddings = self.cohere_client.embed(
                texts=[state.query],
                model=self.cohere_model,
                input_type="search_query",
                embedding_types=["float"],
            )

            # Retrieve documents
            response = self.qdrant_client.query_points(
                collection_name=state.collection_choice,
                query=query_embeddings.embeddings.float_[0],
                limit=10,
                with_payload = True,
                with_vectors = False,
            ).points

            docs = []
            for point in response:
                content = point.payload.get("document", "")
                metadata = {
                    k: v for k, v in point.payload.items() if k not in ["document"]
                }
                docs.append(Document(page_content=content, metadata=metadata))

            # Update state
            state.context = docs
            state.source_documents = [
                {"content": doc.page_content, "metadata": doc.metadata} for doc in docs
            ]
            state.intermediate_steps.append(
                {
                    "action": "retrieve_context",
                    "collection": state.collection_choice,
                    "num_docs": len(docs),
                }
            )

            return state
        except Exception as e:
            state.error = f"Error during retrieval: {str(e)}"
            return state

    def _extract_financial_entities(self, state: RAGAgentState) -> RAGAgentState:
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
            
            Format your response as a valid JSON array, nothing else.
            """
        )

        try:
            chain = prompt | self.llm | StrOutputParser()
            result = chain.invoke({"context": context_text})
            output_parser = JsonOutputParser(pydantic_object=state.financial_entities)
            entities = output_parser.parse(result)

            # Update state
            state.financial_entities = entities
            state.intermediate_steps.append(
                {"action": "extract_entities", "num_entities": len(entities)}
            )

            print(f"Extracted Financial Entities: {entities}")

        except Exception as e:
            state.error = f"Error during entity extraction: {str(e)}"
            state.financial_entities = []

        return state

    def _generate_response(self, state: RAGAgentState) -> RAGAgentState:
        """Generate the final response using the LLM with the retrieved context."""
        # Prepare context snippets
        context_snippets = [
            f"File {doc.metadata['file_name']}:\n{doc.page_content}\n"
            for i, doc in enumerate(state.context)
        ]

        context_text = "\n".join(context_snippets)

        # Format entity information
        entity_info = ""
        if state.financial_entities:
            entity_info = "Key entities identified:\n"
            for entity in state.financial_entities:
                entity_info += f"- {entity.get('entity_name', 'Unknown')} ({entity.get('entity_type', 'Unknown')})"
                if entity.get("value"):
                    entity_info += f": {entity.get('value')}"
                if entity.get("sentiment"):
                    entity_info += f" [{entity.get('sentiment')}]"
                entity_info += "\n"

        prompt = ChatPromptTemplate.from_template(
            """You are a financial analysis assistant that provides accurate information based on the retrieved documents.
            
            User query: {query}

            Data source: {collection_choice}
            
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
        response = chain.invoke(
            {
                "query": state.query,
                "collection_choice": state.collection_choice,
                "context": context_text,
                "entity_info": entity_info,
            }
        )

        # Update state
        state.response = response

        return state

    def process_query(self, query: str) -> Dict:
        """Process a user query through the full graph and return the result."""
        # Create initial state
        initial_state = RAGAgentState(query=query)

        # Run the graph
        result = self.graph.invoke(initial_state)

        result_dict = dict(result)  # Convert to a standard dictionary
        output = {
            "response": result_dict.get("response"),
            "source_documents": result_dict.get("source_documents", [])[:3],
            "collection_used": result_dict.get("collection_choice"),
            "entities": result_dict.get("financial_entities"),
            "error": result_dict.get("error"),
        }


        return output
