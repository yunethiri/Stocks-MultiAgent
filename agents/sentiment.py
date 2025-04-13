import os
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from pydantic import BaseModel, Field
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.http import models
from langchain.output_parsers import PydanticOutputParser
from dotenv import load_dotenv
import json
from openai import OpenAI  # Import the OpenAI client
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from qdrant_client.models import Filter, FieldCondition, Range
from datetime import datetime
from numpy import zeros
load_dotenv()

# Retrieve the API key from the environment variable
api_key = os.getenv("OPENAI_API_KEY")

# Initialize the OpenAI client
client = OpenAI(api_key=api_key)

class ArticleSentiment(BaseModel):
    article_id: str
    title: str
    date: str
    sentiment_score: float = Field(description="Sentiment score of the article (-1 to 1)")
    confidence: float = Field(description="Confidence in the sentiment score (0 to 1)")
    summary: str
    content: str
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
        self.qdrant = QdrantClient(url="http://qdrant:6333")
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
            You are an expert financial analyst. 
            The articles_data parameter contains a list of dictionaries (representing differet news articles) with the fields (ID, title, date, sentiment_score, confidence_score, summary, content).
            Based on the following news article information, identify the major events that occurred during this timeframe. Focus on events with a significant impact on the financial market (high magnitude of sentiment_score and confidence_score), ignoring neutral events (sentiment_score of around 0).

            Articles:
            {articles_data}

            Identify major events from these articles. For each event:
            1. Provide a concise title for the event
            2. The date it occurred
            3. A brief description of the event and how it impacts the stock (based on content)
            4. The sentiment impact (positive, negative, or neutral)
            5. The sentiment score of the article (if available)
            6. The confidence score of the article (if available)
            7. ID of the article which mention the event

            Format your response as a JSON list of objects with keys: "title", "date", "description", "sentiment_impact", "sentiment_score", "confidence_score", "article_ids"
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
        print(f"Current date for timeframe extraction: {current_date}") # Print current date

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

            Respond with ONLY a valid JSON object with these fields:
            - valid: true if at least some portion of the timeframe is within 2024
            - start_date: start date in%Y-%m-%d format (if valid)
            - end_date: end date in%Y-%m-%d format (if valid)
            - description: human-readable description of the timeframe, including any adjustments made to fit within available data
            """
        )

        # Create and run the chain
        timeframe_chain = LLMChain(llm=self.llm, prompt=timeframe_prompt)
        result = timeframe_chain.run(query=query, current_date=current_date)
        print(f"LLM response for timeframe extraction: {result}") # Print LLM response

        try:
            # Attempt to clean the result if it's not proper JSON
            cleaned_result = result.strip()
            if not cleaned_result.startswith('{'):
                import re
                json_match = re.search(r'(\{.*\})', cleaned_result, re.DOTALL)
                if json_match:
                    cleaned_result = json_match.group(1)
                else:
                    print("Warning: LLM response for timeframe is not valid JSON. Using default timeframe.")
                    return {
                        "start_date": datetime(2024, 1, 1).date(),
                        "end_date": datetime(2024, 12, 31).date(),
                        "description": "full year 2024 (default)"
                    }

            # Parse the JSON response
            timeframe_data = json.loads(cleaned_result)
            print(f"Parsed timeframe data: {timeframe_data}") # Print parsed data

            if not timeframe_data.get("valid", False):
                print("Extracted timeframe is outside the valid range.")
                return None  # Indicate timeframe outside available data

            # Convert string dates to datetime objects
            start_date_str = timeframe_data.get("start_date")
            end_date_str = timeframe_data.get("end_date")

            if start_date_str and end_date_str:
                start_date = datetime.strptime(start_date_str, "%Y-%m-%d").date()
                end_date = datetime.strptime(end_date_str, "%Y-%m-%d").date()
            else:
                print("Warning: Start or end date missing from LLM response. Using default timeframe.")
                return {
                    "start_date": datetime(2024, 1, 1).date(),
                    "end_date": datetime(2024, 12, 31).date(),
                    "description": "full year 2024 (default)"
                }
            description = timeframe_data["description"]

            # Validate that dates are within 2024
            min_date = datetime(2024, 1, 1).date()
            max_date = datetime(2024, 12, 31).date()

            if start_date < min_date:
                start_date = min_date
            if end_date > max_date:
                end_date = max_date

            print(f"Extracted Start Date: {start_date}, End Date: {end_date}, Description: {description}") # Print extracted dates
            return {
                "start_date": start_date,
                "end_date": end_date,
                "description": description
            }

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            # If there's an error parsing the LLM response, fall back to default timeframe
            print(f"Error parsing timeframe from LLM: {e}")
            print(f"Raw LLM response: {result}")  # Log the raw response for debugging
            return {
                "start_date": datetime(2024, 1, 1).date(),
                "end_date": datetime(2024, 12, 31).date(),
                "description": "full year 2024 (default)"
            }



    def _retrieve_articles(self, start_date: datetime, end_date: datetime) -> List[Dict]:
        """Retrieve articles from Qdrant within the specified timeframe"""
        if not self.qdrant_collection_name:
            return []

        print(f"Retrieving articles between {start_date} and {end_date}")  # Print retrieval timeframe

        # Convert datetime.date objects to datetime.datetime objects at midnight (00:00:00)
        start_timestamp = start_date.isoformat()
        end_timestamp = end_date.isoformat()

        # Construct the date filter for Qdrant using Range with UNIX timestamps
        date_filter = Filter(
            must=[
                FieldCondition(
                    key="publish_date",
                    range=models.DatetimeRange(
                        gte=start_timestamp,
                        lte=end_timestamp,
                    )
                )
            ]
        )

        # Create a dummy query vector with the correct dimension (1024)
        query_vector = zeros(1024)  # Adjust the vector size to match your collection's expected dimension

        # Retrieve articles directly filtered by date from Qdrant
        found_articles = self.qdrant.search(
            collection_name=self.qdrant_collection_name,
            query_vector=query_vector,  # Correct vector size
            query_filter=date_filter,
            limit=1000,  # Adjust limit as needed
            with_payload=True
        )

        articles_data = [{"id": hit.id, "payload": hit.payload} for hit in found_articles]

        print(f"Found {len(articles_data)} articles within the timeframe.")  # Print number of found articles
        return articles_data

    def _map_sentiment_string_to_weighted_score(self, sentiment_str: str) -> Optional[float]:
        """Maps a sentiment string to a weighted numerical score."""
        sentiment_lower = sentiment_str.lower()
        if sentiment_lower == 'positive':
            return 1.0
        elif sentiment_lower == 'negative':
            return -1.0
        elif sentiment_lower == 'neutral':
            return 0.0
        # Add more mappings as needed for your data
        return None

    def _format_articles_for_event_detection(self, articles: List[ArticleSentiment]) -> str:
        """Formats article data for the event detection prompt."""
        formatted_articles = []
        for article in articles:
            sentiment_category = "neutral"
            if article.sentiment_score > 0.2:
                sentiment_category = "positive"
            elif article.sentiment_score < -0.2:
                sentiment_category = "negative"
            formatted_articles.append(f"ID: {article.article_id}, Title: {article.title}, Date: {article.date}, Sentiment Score: {article.sentiment_score:.2f}, Confidence Score: {article.confidence:.2f}, Summary: {article.summary}, Content: {article.content}")
        return "\n".join(formatted_articles)

    def _format_sentiment_data_for_trend_analysis(self, articles: List[ArticleSentiment]) -> str:
        """Formats sentiment data for the trend analysis prompt."""
        sentiment_data_list = [f"Date: {article.date}, Sentiment Score: {article.sentiment_score:.2f}" for article in articles]
        return "\n".join(sentiment_data_list)

    def _format_response(self, result: TimeframeSentiment) -> str:
        """Formats the final response for the user."""
        response = f"Sentiment analysis for {result.timeframe}:\n"
        response += f"- Average Sentiment: {result.average_sentiment:.2f} (based on {result.article_count} articles)\n"
        response += f"- Average Confidence: {result.average_confidence:.2f}\n"
        response += f"- Sentiment Trend: {result.sentiment_trend}\n"
        if result.major_events:
            response += "\nMajor Events Identified:\n"
            for event in result.major_events:
                response += f"  - Title: {event['title']}\n"
                response += f"    - Date: {event['date']}\n"
                response += f"    - Description: {event['description']}\n"
                response += f"    - Sentiment Impact: {event['sentiment_impact']}\n"
                response += f"    - Articles: {event['article_ids']}\n"
        else:
            response += "\nNo major events identified during this timeframe.\n"
        return response
    
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
        sentiment_sum = 0
        total_confidence = 0
        valid_article_count = 0

        for article in articles_data:
            sentiment_raw = article["payload"].get("sentiment")
            confidence_raw = article["payload"].get("confidence")
            title = article["payload"].get("title", "Unknown Title")
            date_str = article["payload"].get("publish_date", "Unknown Date")
            summary = article["payload"].get("summary", "No summary available")
            content = article["payload"].get("content", "No content available")
            url = article["payload"].get("url", None)

            sentiment_score = None
            if isinstance(sentiment_raw, str):
                sentiment_score = self._map_sentiment_string_to_weighted_score(sentiment_raw)
                if sentiment_score is None:
                    print(f"Warning: Could not map sentiment string '{sentiment_raw}' for article {article['id']}")
                    continue # Skip this article if sentiment string cannot be mapped
            else:
                print(f"Warning: Invalid sentiment format for article {article['id']}")
                continue

            confidence = None
            if isinstance(confidence_raw, (int, float)):
                confidence = float(confidence_raw)
            else:
                print(f"Warning: Invalid confidence format for article {article['id']}")
                continue

            if sentiment_score is not None and confidence is not None and date_str:
                try:
                    article_sentiment = ArticleSentiment(
                        article_id=str(article["id"]),
                        title=title,
                        date=date_str,
                        sentiment_score=sentiment_score,
                        confidence=confidence,
                        summary=summary,
                        content=content,
                        url=url
                    )
                    articles_sentiment.append(article_sentiment)
                    sentiment_sum += sentiment_score
                    total_confidence += confidence
                    valid_article_count += 1
                except ValueError:
                    print(f"Warning: Could not create ArticleSentiment object for article {article['id']}")

        average_sentiment = 0.0
        average_confidence = 0.0

        if valid_article_count > 0:
            average_sentiment = sentiment_sum / total_confidence if total_confidence > 0 else 0.0
            average_confidence = total_confidence / valid_article_count

        result_articles = [{
            "article_id": article.article_id,
            "title": article.title,
            "date": article.date,
            "sentiment_score": article.sentiment_score,
            "confidence": article.confidence,
            "summary": article.summary,
            "content": article.content,
            "url": article.url
        } for article in articles_sentiment]

        # Format articles for event detection
        articles_for_event_detection = self._format_articles_for_event_detection(articles_sentiment)

        # Detect major events
        major_events = self.event_detection_chain.run(articles_data=articles_for_event_detection)
        try:
            # Debug info
            #print(f"Type of major_events: {type(major_events)}")
            #print(f"Raw major_events: {repr(major_events)}")
            
            # Handle different possible formats
            if isinstance(major_events, list) or isinstance(major_events, dict):
                # Already a Python object, use directly
                major_events_list = major_events
            elif isinstance(major_events, str):
                # Clean the string of any potentially problematic characters
                major_events = major_events.strip()
                # Try to handle common LLM output issues
                if major_events.startswith("```json") and major_events.endswith("```"):
                    major_events = major_events[7:-3].strip()
                # Parse JSON
                major_events_list = json.loads(major_events)
            else:
                raise TypeError(f"Unexpected type: {type(major_events)}")
                
        except Exception as e:
            print(f"Error type: {type(e).__name__}")
            print(f"Error details: {str(e)}")
            # Fallback
            major_events_list = [{"title": "Error parsing events", "date": None, "description": "Could not parse event data.", "sentiment_impact": "neutral", "article_ids": []}]
        
        # Format sentiment data for trend analysis
        sentiment_data_for_trend = self._format_sentiment_data_for_trend_analysis(articles_sentiment)

        # Analyze sentiment trend
        sentiment_trend = self.trend_analysis_chain.run(sentiment_data=sentiment_data_for_trend, timeframe=timeframe_info["description"])

        result = TimeframeSentiment(
            timeframe=timeframe_info["description"],
            start_date=timeframe_info["start_date"].strftime("%Y-%m-%d"),
            end_date=timeframe_info["end_date"].strftime("%Y-%m-%d"),
            average_sentiment=round(average_sentiment, 2),
            average_confidence=round(average_confidence, 2),
            article_count=valid_article_count,
            major_events=major_events_list,
            sentiment_trend=sentiment_trend.strip(),
            articles=result_articles
        )

        response = self._format_response(result)

        return {
            "response": response,
            "data": result.dict(),
            "timeframe": timeframe_info["description"],
            "error": None
        }

# # Test with different timeframes
# print("\n=== Testing Different Timeframes ===")
# test_queries = [
#     "How did the sentiment about Apple change in the first half of 2024?",
#     "How did the sentiment about Apple change in the second half of 2024?",
#     "What was the sentiment trend for Apple in December 2024?"
# ]

# sentiment_agent = SentimentAgent(model_name="gpt-4o")
# for query in test_queries:
#     print(f"\n{'='*80}\nQuery: {query}\n{'='*80}")
#     result = sentiment_agent.process_query(query)
#     print(result["response"])
#     print(f"\nTimeframe: {result['timeframe']}")
#     if result["error"]:
#         print(f"Error: {result['error']}")