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

            Identify 3-5 major events from these articles. For each event:
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
        """Extract timeframe information from the query"""

        min_date = datetime(2024, 1, 1)
        max_date = datetime(2024, 12, 31)
        today = min(datetime.now(), max_date)

        # Default to last year (full 2024)
        start_date = min_date
        end_date = max_date
        description = "full year 2024"

        # Check for specific timeframes in the query
        if "last year" in query.lower():
            start_date = datetime(2024, 1, 1)
            end_date = datetime(2024, 12, 31)
            description = "last year"
        elif "q1" in query.lower() or "Q1" in query.lower() or "first quarter" in query.lower():
            start_date = datetime(2024, 1, 1)
            end_date = datetime(2024, 3, 31)
            description = "Q1 2024"
        elif "q2" in query.lower() or "Q2" in query.lower() or "second quarter" in query.lower():
            start_date = datetime(2024, 4, 1)
            end_date = datetime(2024, 6, 30)
            description = "Q2 2024"
        elif "q3" in query.lower() or "Q3" in query.lower() or "third quarter" in query.lower():
            start_date = datetime(2024, 7, 1)
            end_date = datetime(2024, 9, 30)
            description = "Q3 2024"
        elif "q4" in query.lower() or "Q4" in query.lower() or "fourth quarter" in query.lower():
            start_date = datetime(2024, 10, 1)
            end_date = datetime(2024, 12, 31)
            description = "Q4 2024"
        elif "last month" in query.lower():
            today = datetime(2025, 4, 8) # Setting a fixed 'today' for consistent testing
            first_day_last_month = datetime(today.year, today.month - 1, 1)
            last_day_last_month = datetime(today.year, today.month, 1) - timedelta(days=1)
            if first_day_last_month.year == 2024:
                start_date = first_day_last_month
                end_date = last_day_last_month
                description = last_day_last_month.strftime("%B %Y")
            else:
                return None # Indicate timeframe outside available data
        elif "this month" in query.lower():
            today = datetime(2025, 4, 8) # Setting a fixed 'today' for consistent testing
            start_date = datetime(today.year, today.month, 1)
            end_date = datetime(today.year, today.month + 1, 1) - timedelta(days=1)
            if start_date.year == 2024 and end_date.year == 2024:
                description = end_date.strftime("%B %Y")
            else:
                return None # Indicate timeframe outside available data
        elif "last week" in query.lower():
            today = datetime(2025, 4, 8) # Setting a fixed 'today' for consistent testing
            start_date = today - timedelta(days=today.weekday() + 7)
            end_date = start_date + timedelta(days=6)
            if start_date.year == 2024 and end_date.year == 2024:
                description = f"week of {start_date.strftime('%Y-%m-%d')}"
            else:
                return None # Indicate timeframe outside available data
        elif "this week" in query.lower():
            today = datetime(2025, 4, 8) # Setting a fixed 'today' for consistent testing
            start_date = today - timedelta(days=today.weekday())
            end_date = start_date + timedelta(days=6)
            if start_date.year == 2024 and end_date.year == 2024:
                description = f"week of {start_date.strftime('%Y-%m-%d')}"
            else:
                return None # Indicate timeframe outside available data
        # Add more specific date range parsing if needed, ensuring they fall within 2024
        else:
            # If no specific valid timeframe is found, default to full year 2024
            start_date = datetime(2024, 1, 1)
            end_date = datetime(2024, 12, 31)
            description = "full year 2024"

        return {
            "start_date": start_date,
            "end_date": end_date,
            "description": description
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
                limit=100,
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