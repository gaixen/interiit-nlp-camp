import time
import logging
from urllib.parse import urlparse
import pickle
import numpy as np

from search_agents import DuckDuckGoSearchAgent, WikipediaSearchAgent, MultiSearchAgent
from scraper import WebScraperAgent
from rag_agent import ImprovedRAGAgent, EMBEDDINGS_AVAILABLE
from data_models import RAGResult
import config

logger = logging.getLogger(__name__)

class RobustWebSearchRAGSystem:
    """Main system that orchestrates search, scraping, and RAG."""
    def __init__(self, search_agents: list, use_embeddings: bool = True, scraper_config: dict = None, rag_config: dict = None):
        logger.info("Initializing Robust Web Search RAG System...")
        self.scraper_config = scraper_config or {}
        self.rag_config = rag_config or {}
        
        # Now accepts a list of agents and passes it to the MultiSearchAgent
        self.search_agents = search_agents
        self.multi_search = MultiSearchAgent(self.search_agents, max_concurrent=config.MULTI_SEARCH_CONCURRENCY)
        
        self.rag_agent = ImprovedRAGAgent(
            use_embeddings=use_embeddings, **self.rag_config)
    
    async def query_with_rag(self, query: str, search_first: bool = True, num_search_results: int = 8) -> dict:
        logger.info(f"--- Starting new RAG pipeline for query: '{query}' ---")
        start_time = time.time()
        try:
            if not query.strip(): return {"error": "Empty query provided"}
            
            scraped_contents = []
            if search_first:
                logger.info("[STEP 1/4] Performing multi-agent search.")
                search_response = await self.multi_search.search(query, num_search_results)
                if not search_response.success or not search_response.results:
                    return {"error": "Search failed or returned no results"}
                
                valid_urls = [r.url for r in search_response.results if urlparse(r.url).scheme in ['http', 'https']]
                logger.info(f"[STEP 2/4] Scraping {len(valid_urls)} valid URLs.")
                if not valid_urls: return {"error": "No valid URLs found to scrape"}
                
                async with WebScraperAgent(**self.scraper_config) as scraper:
                    scraped_contents = await scraper.scrape_multiple_urls(valid_urls, config.SCRAPER_MAX_CONCURRENT)
                
                successful_contents = [c for c in scraped_contents if c.success and len(c.content) > 100]
                logger.info(f"[STEP 3/4] Indexing content from {len(successful_contents)} successfully scraped pages.")
                if successful_contents: await self.rag_agent.index_documents(successful_contents)
            
            logger.info("[STEP 4/4] Querying RAG index.")
            rag_result = await self.rag_agent.query(query, top_k=num_search_results)
            
            total_time = time.time() - start_time
            logger.info(f"--- RAG pipeline completed in {total_time:.2f}s ---")
            return {
                "success": True, "rag_result": rag_result, "scraped_contents": scraped_contents,
                "total_processing_time": total_time, "statistics": self._get_statistics()}
        except Exception as e:
            logger.critical(f"A critical error occurred in the RAG pipeline: {e}", exc_info=True)
            return {"error": f"System error: {str(e)}"}

    def _get_statistics(self) -> dict:
        doc_chunks = self.rag_agent.document_chunks
        return {
            "total_chunks": len(doc_chunks),
            "unique_sources": len(set(c.source_url for c in doc_chunks)),
            "embeddings_enabled": self.rag_agent.use_embeddings,
            "avg_chunk_size": np.mean([c.metadata.get('word_count', 0) for c in doc_chunks]) if doc_chunks else 0
        }

    def save_index(self, filepath: str):
        logger.info(f"Saving index to {filepath}...")
        with open(filepath, 'wb') as f:
            pickle.dump({
                "document_chunks": self.rag_agent.document_chunks,
                "embeddings": getattr(self.rag_agent, 'embeddings', None)
            }, f)
        logger.info("Index saved successfully.")

    def load_index(self, filepath: str):
        logger.info(f"Loading index from {filepath}...")
        with open(filepath, 'rb') as f: data = pickle.load(f)
        
        self.rag_agent.document_chunks = data["document_chunks"]
        if self.rag_agent.use_embeddings and data.get("embeddings") is not None:
            self.rag_agent.embeddings = data["embeddings"]
            if EMBEDDINGS_AVAILABLE:
                # ... (FAISS index creation logic from rag_agent.py) ...
                logger.info("FAISS index successfully loaded and rebuilt.")
        logger.info(f"Index loaded successfully with {len(self.rag_agent.document_chunks)} chunks.")