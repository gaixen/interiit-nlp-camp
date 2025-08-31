"""
3 Web Search Agents with fallback logic.

1. SerpAPI
2. TavilySearch
3. DuckDuckGo
"""

import os
from typing import List, Dict, Any, Optional

from pydantic import BaseModel, Field
from dotenv import load_dotenv
from ddgs import DDGS

from langchain.agents import tool
from langchain_tavily import TavilySearch 
from langchain_community.utilities import SerpAPIWrapper

load_dotenv()


class SearchResult(BaseModel):
    """Pydantic model for individual search results."""
    title: str
    content: str
    source: str

class SearchResponse(BaseModel):
    """Pydantic model for search responses."""
    success: bool
    results: List[SearchResult] = Field(default_factory=list)
    error: Optional[str] = None
    agent_used: Optional[str] = None


tavily = TavilySearch(max_results=5) if os.getenv('TAVILY_API_KEY') else None
serp = SerpAPIWrapper(params={"num": 5}) if os.getenv('SERPAPI_API_KEY') else None


def run_tavily_search(query: str) -> SearchResponse:
    """Tavily search"""
    if not tavily:
        return SearchResponse(success=False, error="Tavily API key not configured")
    
    try:
        search_result = tavily.invoke(query)
        results = [
            SearchResult(
                title=r.get("title", "No title"),
                content=r.get("content", ""),
                source=r.get("url", "")
            )
            for r in search_result if r.get("url")
        ]
        return SearchResponse(success=True, results=results, agent_used="Tavily")
    except Exception as e:
        return SearchResponse(success=False, error=str(e))


def run_serp_search(query: str) -> SearchResponse:
    """SerpAPI search"""
    if not serp:
        return SearchResponse(success=False, error="SerpAPI key not configured")
    
    try:
        search_result = serp.results(query)
        results = [
            SearchResult(
                title=r.get("title", "No title"),
                content=r.get("snippet", ""),
                source=r.get("link", "")
            )
            for r in search_result if r.get("link")
        ]
        return SearchResponse(success=True, results=results, agent_used="SerpAPI")
    except Exception as e:
        return SearchResponse(success=False, error=str(e))


def run_duckduckgo_search(query: str) -> SearchResponse:
    """DuckDuckGo search"""
    try:
        with DDGS() as ddgs:
            search_result = list(ddgs.text(query, max_results=5))
        results = [
            SearchResult(
                title=r.get("title", "No title"),
                content=r.get("body", ""),
                source=r.get("href", r.get("link", ""))
            )
            for r in search_result if r.get("href") or r.get("link")
        ]
        return SearchResponse(success=True, results=results, agent_used="DuckDuckGo")
    except Exception as e:
        return SearchResponse(success=False, error=str(e))


@tool
def fallback_search(query: str) -> Dict[str, Any]:
    """
    Fallback search function that tries SerpAPI, then Tavily, then DuckDuckGo.
    Returns the first successful search result.
    """
    
    # Try SerpAPI first
    result = run_serp_search(query)
    if result.success:
        return {
            "agent": result.agent_used,
            "final_result": [r.model_dump() for r in result.results]
        }

    # Try Tavily
    result = run_tavily_search(query)
    if result.success:
        return {
            "agent": result.agent_used,
            "final_result": [r.model_dump() for r in result.results]
        }
    
    # Try DuckDuckGo
    result = run_duckduckgo_search(query)
    if result.success:
        return {
            "agent": result.agent_used,
            "final_result": [r.model_dump() for r in result.results]
        }
    
    return {"error": "All search agents failed", "final_result": []}


if __name__ == "__main__":
    test_query = "What is Agentic AI"
    result = fallback_search.invoke(test_query)
    print(result)