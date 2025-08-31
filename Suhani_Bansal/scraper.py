import asyncio
import aiohttp
import re
import time
import logging
from datetime import datetime
from typing import List

from data_models import ScrapedContent

try:
    from bs4 import BeautifulSoup
    import trafilatura
    SCRAPING_AVAILABLE = True
except ImportError:
    SCRAPING_AVAILABLE = False

logger = logging.getLogger(__name__)

class WebScraperAgent:
    """web scraper with content extraction"""
    def __init__(self, max_retries: int = 3, timeout: int = 30, rate_limit_delay: float = 1.0):
        self.max_retries = max_retries
        self.timeout = timeout
        self.rate_limit_delay = rate_limit_delay
        self.session = None
        self.last_request_time = 0
        logger.info(f"WebScraperAgent initialized with max_retries={max_retries}, timeout={timeout}s")

    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.timeout),
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'})
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def _rate_limit(self):
        now = time.time()
        time_since_last = now - self.last_request_time
        if time_since_last < self.rate_limit_delay:
            delay = self.rate_limit_delay - time_since_last
            logger.debug(f"Rate limiting scraper for {delay:.2f}s")
            await asyncio.sleep(delay)
        self.last_request_time = time.time()

    def _clean_text(self, text: str) -> str:
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s\.,!?;:()\-"]', '', text)
        return re.sub(r' +', ' ', text).strip()

    async def scrape_url(self, url: str) -> ScrapedContent:
        await self._rate_limit()
        logger.info(f"Scraping URL: {url}")
        error_msg = ""
        for attempt in range(self.max_retries):
            try:
                async with self.session.get(url, allow_redirects=True) as response:
                    if response.status == 200:
                        html = await response.text()
                        if SCRAPING_AVAILABLE:
                            logger.debug(f"Attempting extraction with Trafilatura for {url}")
                            content = trafilatura.extract(html, include_comments=False, include_tables=True)
                            if content:
                                soup = BeautifulSoup(html, 'html.parser')
                                title = soup.title.string.strip() if soup.title else "No Title"
                                cleaned_content = self._clean_text(content)
                                logger.info(f"Successfully extracted content from {url} using Trafilatura.")
                                return ScrapedContent(
                                    url=url, title=title, content=cleaned_content, text_length=len(cleaned_content),
                                    scrape_timestamp=datetime.now(), success=True, metadata={"method": "trafilatura"})
                        
                        logger.debug(f"Falling back to BeautifulSoup for {url}")
                        soup = BeautifulSoup(html, 'html.parser')
                        title = soup.title.string.strip() if soup.title else "No Title"
                        for element in soup(['script', 'style', 'nav', 'header', 'footer', 'aside']): element.decompose()
                        main_content = soup.find('main') or soup.find('article') or soup.body
                        text_content = main_content.get_text() if main_content else ""
                        cleaned_content = self._clean_text(text_content)
                        logger.info(f"Successfully extracted content from {url} using BeautifulSoup.")
                        return ScrapedContent(
                            url=url, title=title, content=cleaned_content, text_length=len(cleaned_content),
                            scrape_timestamp=datetime.now(), success=True, metadata={"method": "beautifulsoup"})
                    else:
                        error_msg = f"HTTP {response.status}"
            except Exception as e:
                error_msg = str(e)
                logger.warning(f"Attempt {attempt + 1}/{self.max_retries} failed for {url}: {e}")
                if attempt < self.max_retries - 1: await asyncio.sleep(2 ** attempt)

        logger.error(f"All {self.max_retries} attempts failed for URL: {url}. Last error: {error_msg}")
        return ScrapedContent(url=url, title="", content="", text_length=0, scrape_timestamp=datetime.now(), success=False, error_message=error_msg)

    async def scrape_multiple_urls(self, urls: List[str], max_concurrent: int = 5) -> List[ScrapedContent]:
        logger.info(f"Starting concurrent scrape for {len(urls)} URLs with max concurrency of {max_concurrent}.")
        semaphore = asyncio.Semaphore(max_concurrent)
        async def scrape_with_semaphore(url):
            async with semaphore: return await self.scrape_url(url)
        tasks = [scrape_with_semaphore(url) for url in urls]
        return await asyncio.gather(*tasks)