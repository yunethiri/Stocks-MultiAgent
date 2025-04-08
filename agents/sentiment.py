import os
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from pydantic import BaseModel, Field
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.http import models
from langchain.output_parsers import PydanticOutputParser
from dotenv import load_dotenv
import json

load_dotenv()

class ArticleSentiment(BaseModel):
    """ Model for article sentiment analysis results """
    article_id: str
    title: str
    date: str
    sentiment_score: float = Field(description="Sentiment score of the article (-1 to 1)")
    confidence: float = Field(description="Confidence in the sentiment score (0 to 1)")
    summary: str = Field(description="Brief summary of the article")
    url: Optional[str] = Field(default=None, description="URL of the article")

class TimeframeSentiment(BaseModel):
    """ Model for aggregated sentiment over a timeframe """
    timeframe: str = Field(description="Description of the timeframe")
    start_date: str = Field(description="Start date of the timeframe (YYYY-MM-DD)")
    end_date: str = Field(description="End date of the timeframe (YYYY-MM-DD)")
    average_sentiment: float = Field(description="Average sentiment score for the timeframe")
    average_confidence: float = Field(description="Average confidence score for the timeframe")
    article_count: int = Field(description="Number of articles analyzed")
    major_events: List[Dict[str, Any]] = Field(default_factory=list, description="List of major events identified")
    sentiment_trend: str = Field(description="Analysis of the sentiment trend over the timeframe")
    articles: List[ArticleSentiment] = Field(default_factory=list, description="List of individual article sentiment results")

class SentimentAgentState(BaseModel):
    """Represents the state of the sentiment analysis agent."""
    query: str = Field(description="The original user query")
    timeframe_description: Optional[str] = Field(default=None, description="Description of the extracted timeframe")
    start_date: Optional[datetime] = Field(default=None, description="Start date of the analysis")
    end_date: Optional[datetime] = Field(default=None, description="End date of the analysis")
    articles_sentiment: List[ArticleSentiment] = Field(default_factory=list, description="Sentiment analysis for retrieved articles")
    aggregated_sentiment: Optional[TimeframeSentiment] = Field(default=None, description="Aggregated sentiment over the timeframe")
    response: Optional[str] = Field(default=None, description="The final response to the user")
    intermediate_steps: List[Dict] = Field(default_factory=list, description="Intermediate steps for debugging")
    error: Optional[str] = Field(default=None, description="Error message if any")

class SentimentAgent:
    """ Agent for analyzing sentiment from news articles over specific timeframes"""

    def __init__(self, model_name="gpt-4o", api_key=None):
        self.openai_api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.llm = ChatOpenAI(temperature=0, model=model_name, api_key=self.openai_api_key)
        self.embeddings = OpenAIEmbeddings(openai_api_key=self.openai_api_key)

        # Initialize Qdrant client
        self.qdrant = QdrantClient(url="http://localhost:6333")
        try:
            self.qdrant_collection_info = self.qdrant.get_collection(collection_name="financial_news")
            self.qdrant_collection_name = "financial_news"
        except Exception as e:
            print(f"Warning: Could not connect to or find collection 'financial_news' in Qdrant: {e}")
            self.qdrant_collection_name = None

        # Prompt templates for event detection and trend sentiment analysis
        self.event_detection_template = PromptTemplate(
            input_variables=["articles_data"],
            template="""
            You are an expert financial analyst. Based on the following news article summaries and their sentiment scores,
            identify the major events that occurred during this timeframe. Focus on events that significantly impacted
            the stock or company.

            Articles:
            {articles_data}

            Identify major events from these articles. For each event:
            1. Provide a concise title for the event
            2. The date it occurred
            3. A brief description of the event
            4. The sentiment impact (positive, negative, or neutral)
            5. Which articles mention this event (by ID)

            Format your response as a JSON list of objects with keys: "title", "date", "description", "sentiment_impact", "article_ids"
            """
        )

        self.trend_analysis_template = PromptTemplate(
            input_variables=["sentiment_data", "timeframe"],
            template="""
            You are an expert financial analyst. Analyze the following sentiment data for a {timeframe} timeframe:

            {sentiment_data}

            Describe the overall sentiment trend during this period. Consider:
            1. Whether sentiment improved, worsened, or remained stable
            2. Any significant fluctuations
            3. How the ending sentiment compares to the beginning

            Provide a concise 2-3 sentence analysis of the sentiment trend.
            """
        )

        self.event_detection_chain = LLMChain(llm=self.llm, prompt=self.event_detection_template)
        self.trend_analysis_chain = LLMChain(llm=self.llm, prompt=self.trend_analysis_template)
        self.article_sentiment_parser = PydanticOutputParser(pydantic_object=ArticleSentiment)

    def _extract_timeframe(self, query: str) -> Dict[str, Any]:
        """Extract timeframe from the query using LLM"""

        current_date = datetime.now().date()
        
        # Define the prompt for timeframe extraction
        timeframe_prompt = PromptTemplate(
            input_variables=["query", "current_date"],
            template="""
            You are a financial data assistant that extracts timeframe information from user queries.
            
            User query: {query}
            
            Today's date is {current_date}. Extract the timeframe mentioned in the query. The available data is ONLY for the year 2024.
            If the query mentions a timeframe that's partially or fully outside of 2024:
            1. If the timeframe is completely outside 2024, indicate that it's outside the available range in the description, return 'valid' field as false, and return start_date and end_date as None.
            2. If the timeframe partially overlaps with 2024, adjust the timeframe to only include the portion within 2024, explain this adjustment in the description, and return 'valid' as true.
            For example: If asked "data from one year ago to now" and today is March 1, 2025, then return the period within the valid timeframe which is from March 1, 2024 to December 31, 2024.
            
            For reference:
            - Q1 2024: January 1, 2024 to March 31, 2024
            - First Quarter of 2024: January 1, 2024 to March 31, 2024
            - Q2 2024: April 1, 2024 to June 30, 2024
            - Second Quarter of 2024: April 1, 2024 to June 30, 2024
            - Q3 2024: July 1, 2024 to September 30, 2024
            - Third Quarter of 2024: July 1, 2024 to September 30, 2024
            - Q4 2024: October 1, 2024 to December 31, 2024
            - Fourth Quarter of 2024: October 1, 2024 to December 31, 2024
            - First Half of 2024: January 1, 2024 to June 30, 2024
            - Second Half of 2024: July 1, 2024 to December 31, 2024
            - Whole Year of 2024: January 1, 2024 to December 31, 2024
            
            Return a JSON object with these fields:
            - valid: true if at least some portion of the timeframe is within 2024
            - start_date: start date in YYYY-MM-DD format (if valid)
            - end_date: end date in YYYY-MM-DD format (if valid)
            - description: human-readable description of the timeframe, including any adjustments made to fit within available data
            """
        )
        # Create and run the chain
        timeframe_chain = LLMChain(llm=self.llm, prompt=timeframe_prompt)
        result = timeframe_chain.run(query=query, current_date=current_date)
        
        try:
            # Parse the JSON response
            timeframe_data = json.loads(result)
            
            if not timeframe_data.get("valid", False):
                return None  # Indicate timeframe outside available data
            
            # Convert string dates to datetime objects
            start_date = datetime.strptime(timeframe_data["start_date"], "%Y-%m-%d")
            end_date = datetime.strptime(timeframe_data["end_date"], "%Y-%m-%d")
            description = timeframe_data["description"]
            
            # Validate that dates are within 2024
            min_date = datetime(2024, 1, 1)
            max_date = datetime(2024, 12, 31)
            
            if start_date <= min_date or end_date >= max_date:
                return None  # Additional check for timeframe outside available data
            
            return {
                "start_date": start_date,
                "end_date": end_date,
                "description": description
            }
            
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            # If there's an error parsing the LLM response, fall back to default timeframe
            print(f"Error parsing timeframe from LLM: {e}")
            return {
                "start_date": datetime(2024, 1, 1),
                "end_date": datetime(2024, 12, 31),
                "description": "full year 2024"
            }

    def _retrieve_articles(self, start_date: datetime, end_date: datetime) -> List[Dict]:
        """Retrieve articles from Qdrant within the specified timeframe"""
        if not self.qdrant_collection_name:
            return []

        # Convert dates to string format for filtering
        start_str = start_date.strftime("%Y-%m-%d")
        end_str = end_date.strftime("%Y-%m-%d")

        # Query Qdrant for articles in the timeframe
        filter_query = models.Filter(
            must=[
                models.FieldCondition(
                    key="publish_date",
                    range=models.Range(
                        gte=start_str,
                        lte=end_str
                    )
                )
            ]
        )

        all_articles = []
        scroll_offset = None
        while True:
            results, scroll_offset = self.qdrant.scroll(
                collection_name=self.qdrant_collection_name,
                scroll_filter=filter_query,
                limit=None,
                offset=scroll_offset,
                with_payload=True
            )
            if not results:
                break
            for hit in results:
                if hit.payload:
                    all_articles.append({
                        "id": hit.id,
                        "payload": hit.payload
                    })
            if scroll_offset is None:
                break
        return all_articles

    def process_query(self, query: str) -> Dict[str, Any]:
        """Process a query about stock sentiment over a timeframe"""
        if not self.qdrant_collection_name:
            return {
                "response": "Sentiment analysis is currently unavailable due to a connection issue with the data source.",
                "data": None,
                "timeframe": None,
                "error": "Qdrant collection 'financial_news' not found or not accessible."
            }

        # Extract timeframe from query
        timeframe_info = self._extract_timeframe(query)

        if timeframe_info is None:
            return {
                "response": "We currently only have sentiment data available for the full year 2024. Please specify a timeframe within that year.",
                "data": None,
                "timeframe": None,
                "error": "Requested timeframe outside available data."
            }

        # Retrieve articles for the specified timeframe
        articles_data = self._retrieve_articles(timeframe_info["start_date"], timeframe_info["end_date"])

        if not articles_data:
            return {
                "response": f"No articles found for the specified timeframe ({timeframe_info['description']}).",
                "data": {
                    "timeframe": timeframe_info["description"],
                    "start_date": timeframe_info["start_date"].strftime("%Y-%m-%d"),
                    "end_date": timeframe_info["end_date"].strftime("%Y-%m-%d"),
                    "article_count": 0,
                    "articles": []
                },
                "timeframe": timeframe_info["description"],
                "error": None
            }

        # Analyze sentiment of each article
        articles_sentiment: List[ArticleSentiment] = []
        for article in articles_data:
            sentiment_score = article["payload"].get("sentiment")
            confidence = article["payload"].get("confidence")
            title = article["payload"].get("title", "Unknown Title")
            date_str = article["payload"].get("publish_date", "Unknown Date")
            summary = article["payload"].get("summary", "No summary available")
            url = article["payload"].get("url")

            if sentiment_score is not None and confidence is not None and date_str:
                try:
                    article_sentiment = ArticleSentiment(
                        article_id=str(article["id"]),
                        title=title,
                        date=date_str,
                        sentiment_score=float(sentiment_score),
                        confidence=float(confidence),
                        summary=summary,
                        url=url
                    )
                    articles_sentiment.append(article_sentiment)
                except ValueError:
                    print(f"Warning: Could not parse sentiment or confidence for article {article['id']}")

        if not articles_sentiment:
            return {
                "response": f"Could not analyze sentiment for any articles in the specified timeframe ({timeframe_info['description']}).",
                "data": {
                    "timeframe": timeframe_info["description"],
                    "start_date": timeframe_info["start_date"].strftime("%Y-%m-%d"),
                    "end_date": timeframe_info["end_date"].strftime("%Y-%m-%d"),
                    "article_count": 0,
                    "articles": []
                },
                "timeframe": timeframe_info["description"],
                "error": "No articles with valid sentiment data found."
            }

        # Calculate aggregated sentiment
        avg_sentiment = sum(article.sentiment_score for article in articles_sentiment) / len(articles_sentiment)
        avg_confidence = sum(article.confidence for article in articles_sentiment) / len(articles_sentiment)

        # Detect major events
        major_events = self._detect_major_events(articles_sentiment)

        # Analyze sentiment trend
        sentiment_trend = self._analyze_sentiment_trend(articles_sentiment, timeframe_info["description"])

        # Prepare response data
        result = TimeframeSentiment(
            timeframe=timeframe_info["description"],
            start_date=timeframe_info["start_date"].strftime("%Y-%m-%d"),
            end_date=timeframe_info["end_date"].strftime("%Y-%m-%d"),
            average_sentiment=round(avg_sentiment, 2),
            average_confidence=round(avg_confidence, 2),
            article_count=len(articles_sentiment),
            major_events=major_events,
            sentiment_trend=sentiment_trend,
            articles=[{
                "article_id": article.article_id,
                "title": article.title,
                "date": article.date,
                "sentiment_score": article.sentiment_score,
                "confidence": article.confidence,
                "summary": article.summary,
                "url": article.url  } for article in articles_sentiment]
        )

        # Format response for the user
        response = self._format_response(result)

        return {
            "response": response,
            "data": result.dict(),
            "timeframe": timeframe_info["description"],
            "error": None
        }
    
# Test with different timeframes
print("\n=== Testing Different Timeframes ===")
test_queries = [
    "How did the sentiment about Apple change in the first half of 2024?",
    "What was the sentiment trend for Apple in February 2024?",
    "Analyze the sentiment for Apple stock in Q2 2024"
]

sentiment_agent = SentimentAgent(model_name="gpt-4o")
for query in test_queries:
    print(f"\n{'='*80}\nQuery: {query}\n{'='*80}")
    result = sentiment_agent.process_query(query)
    print(result["response"])
    print(f"\nTimeframe: {result['timeframe']}")
    if result["error"]:
        print(f"Error: {result['error']}")