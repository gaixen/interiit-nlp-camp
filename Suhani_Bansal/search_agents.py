import time
import asyncio
import aiohttp
import logging
import os
from abc import ABC, abstractmethod
from urllib.parse import quote_plus
from typing import List

from data_models import SearchResult, SearchResponse

try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False

logger = logging.getLogger(__name__)

class SearchAgent(ABC):
    """Abstract base class for search agents"""
    def __init__(self, base_delay: float = 1.0):
        self.base_delay = base_delay
        self.last_request_time = 0

    @abstractmethod
    async def search(self, query: str, num_results: int = 10) -> SearchResponse:
        """Perform a search and return results"""
        pass

    async def _rate_limit(self):
        """Implement rate limiting"""
        now = time.time()
        time_since_last = now - self.last_request_time
        if time_since_last < self.base_delay:
            delay = self.base_delay - time_since_last
            logger.debug(f"Rate limiting {type(self).__name__} for {delay:.2f}s")
            await asyncio.sleep(delay)
        self.last_request_time = time.time()

class DuckDuckGoSearchAgent(SearchAgent):
    """DuckDuckGo search agent using their HTML search page."""
    async def search(self, query: str, num_results: int = 10) -> SearchResponse:
        await self._rate_limit()
        start_time = time.time()
        encoded_query = quote_plus(query)
        search_url = f"https://html.duckduckgo.com/html/?q={encoded_query}"
        logger.info(f"Searching DuckDuckGo for '{query}'...")

        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(search_url, headers=headers, timeout=15) as response:
                    if response.status == 200:
                        html = await response.text()
                        results = self._parse_duckduckgo_results(html, query)
                        logger.info(f"DuckDuckGo search successful, found {len(results)} results.")
                        return SearchResponse(
                            success=True, results=results[:num_results], source="duckduckgo",
                            total_results=len(results), response_time=time.time() - start_time
                        )
                    else:
                        logger.error(f"DuckDuckGo search failed with HTTP status {response.status}")
                        raise Exception(f"HTTP {response.status}")
        except Exception as e:
            logger.error(f"An error occurred during DuckDuckGo search: {e}", exc_info=True)
            return SearchResponse(
                success=False, results=[], source="duckduckgo", total_results=0,
                response_time=time.time() - start_time, error_message=str(e)
            )

    def _parse_duckduckgo_results(self, html: str, query: str) -> List[SearchResult]:
        if not BS4_AVAILABLE:
            logger.warning("BeautifulSoup not installed. Cannot parse DuckDuckGo results.")
            return []
        try:
            soup = BeautifulSoup(html, 'html.parser')
            results = []
            result_links = soup.find_all('a', class_='result__a')
            result_snippets = soup.find_all('a', class_='result__snippet')

            for i, link in enumerate(result_links[:10]):
                title, url = link.get_text(strip=True), link.get('href', '')
                snippet = result_snippets[i].get_text(strip=True) if i < len(result_snippets) else ""
                if title and url and not url.startswith('/'):
                    results.append(SearchResult(title=title, url=url, snippet=snippet, source="duckduckgo", metadata={"query": query}))
            
            logger.debug(f"Successfully parsed {len(results)} DuckDuckGo results.")
            return results
        except Exception as e:
            logger.warning(f"Failed to parse DuckDuckGo HTML: {e}", exc_info=True)
            return []

class WikipediaSearchAgent(SearchAgent):
    """Wikipedia search using their API"""
    async def search(self, query: str, num_results: int = 10) -> SearchResponse:
        await self._rate_limit()
        start_time = time.time()
        logger.info(f"Searching Wikipedia for '{query}'...")
        search_url = "https://en.wikipedia.org/w/api.php"
        params = {
            'action': 'query', 'format': 'json', 'list': 'search',
            'srsearch': query, 'srlimit': min(num_results, 10),
            'srprop': 'snippet|titlesnippet|size'
        }
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(search_url, params=params, timeout=10) as response:
                    if response.status == 200:
                        data, results = await response.json(), []
                        if 'query' in data and 'search' in data['query']:
                            for item in data['query']['search']:
                                title = item.get('title', '')
                                snippet = item.get('snippet', '').replace('<span class="searchmatch">', '').replace('</span>', '')
                                url = f"https://en.wikipedia.org/wiki/{quote_plus(title.replace(' ', '_'))}"
                                results.append(SearchResult(
                                    title=f"Wikipedia: {title}", url=url, snippet=snippet, source="wikipedia",
                                    metadata={"size": item.get('size', 0)}))
                        logger.info(f"Wikipedia search successful, found {len(results)} results.")
                        return SearchResponse(
                            success=True, results=results, source="wikipedia",
                            total_results=len(results), response_time=time.time() - start_time)
                    else:
                        logger.error(f"Wikipedia search failed with HTTP status {response.status}")
                        raise Exception(f"HTTP {response.status}")
        except Exception as e:
            logger.error(f"An error occurred during Wikipedia search: {e}", exc_info=True)
            return SearchResponse(
                success=False, results=[], source="wikipedia", total_results=0,
                response_time=time.time() - start_time, error_message=str(e))

class SerpApiSearchAgent(SearchAgent):
    """Google Search using the SerpApi service"""

    def __init__(self, api_key: str = None):
        # Checks for the SERPAPI_API_KEY environment variable
        self.api_key = api_key or os.environ.get("SERPAPI_API_KEY")
        if not self.api_key:
            raise ValueError("SerpApi API key not found. Please set the SERPAPI_API_KEY environment variable.")
        super().__init__()

    async def search(self, query: str, num_results: int = 10) -> SearchResponse:
        await self._rate_limit()
        start_time = time.time()
        logger.info(f"Searching Google via SerpApi for '{query}'...")
        
        search_url = "https://serpapi.com/search.json"
        params = {
            "q": query,
            "engine": "google",
            "num": num_results,
            "api_key": self.api_key,
            "location": "Kanpur, Uttar Pradesh, India",
            "hl": "en",
            "gl": "in"
        }
        headers = {'User-Agent': 'python:web-search-rag:v1.0'}

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(search_url, headers=headers, params=params, timeout=20) as response:
                    if response.status == 200:
                        data, results = await response.json(), []
                        for item in data.get("organic_results", []):
                            title, url, snippet = item.get('title', ''), item.get('link', ''), item.get('snippet', '')
                            if title and url:
                                results.append(SearchResult(
                                    title=f"Google (SerpApi): {title}", url=url, snippet=snippet, source="serpapi_google",
                                    metadata={"position": item.get('position', 0)}))
                        
                        logger.info(f"SerpApi search successful, found {len(results)} results.")
                        return SearchResponse(
                            success=True, results=results, source="serpapi_google",
                            total_results=len(results), response_time=time.time() - start_time)
                    else:
                        logger.error(f"SerpApi search failed with HTTP status {response.status}. Response: {await response.text()}")
                        raise Exception(f"HTTP {response.status}")
        except Exception as e:
            logger.error(f"An error occurred during SerpApi search: {e}", exc_info=True)
            return SearchResponse(
                success=False, results=[], source="serpapi_google", total_results=0,
                response_time=time.time() - start_time, error_message=str(e))

class MultiSearchAgent:
    """Agent that combines multiple search agents"""
    def __init__(self, search_agents: List[SearchAgent], max_concurrent: int = 3):
        self.search_agents = search_agents
        self.max_concurrent = max_concurrent
        logger.info(f"MultiSearchAgent initialized with {len(search_agents)} agents and max concurrency {max_concurrent}.")

    async def search(self, query: str, num_results: int = 10) -> SearchResponse:
        start_time = time.time()
        logger.info(f"Starting multi-agent search for '{query}'...")
        semaphore = asyncio.Semaphore(self.max_concurrent)

        async def search_with_semaphore(agent):
            async with semaphore:
                try:
                    return await agent.search(query, num_results // len(self.search_agents) + 2)
                except Exception as e:
                    logger.warning(f"Search agent {type(agent).__name__} failed: {e}")
                    return SearchResponse(False, [], type(agent).__name__, 0, 0, str(e))

        search_tasks = [search_with_semaphore(agent) for agent in self.search_agents]
        responses = await asyncio.gather(*search_tasks, return_exceptions=True)

        all_results, successful_sources, seen_urls = [], [], set()
        for response in responses:
            if isinstance(response, SearchResponse) and response.success:
                for result in response.results:
                    if result.url not in seen_urls:
                        seen_urls.add(result.url)
                        all_results.append(result)
                successful_sources.append(response.source)

        logger.info(f"Combined {len(all_results)} unique results from {len(successful_sources)} sources.")
        all_results.sort(key=lambda x: (len(x.snippet), 1 if x.source == "wikipedia" else 0), reverse=True)
        
        return SearchResponse(
            success=len(all_results) > 0, results=all_results[:num_results], source="+".join(successful_sources),
            total_results=len(all_results), response_time=time.time() - start_time)