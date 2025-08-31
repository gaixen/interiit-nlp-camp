import asyncio
import logging
from urllib.parse import urlparse
import os

import config
from system import RobustWebSearchRAGSystem
from search_agents import DuckDuckGoSearchAgent, WikipediaSearchAgent, SerpApiSearchAgent, MultiSearchAgent
from data_models import SearchResult, SearchResponse

# Setup logging configuration from the config file
logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT)
logger = logging.getLogger(__name__)

async def test_robust_system():
    """Testing the system with real queries"""
    print("\n" + "="*50)
    print("      Testing Web Search RAG System")
    print("="*50)

    # UPDATED: Check for SerpApi key before initializing the agent
    if "SERPAPI_API_KEY" not in os.environ:
        logger.error("SERPAPI_API_KEY environment variable is not set. SerpApi search will fail.")
        return

    # Initialize system using settings from config.py
    system = RobustWebSearchRAGSystem(
        search_agents=[
            DuckDuckGoSearchAgent(base_delay=config.DUCKDUCKGO_DELAY),
            WikipediaSearchAgent(base_delay=config.WIKIPEDIA_DELAY),
            SerpApiSearchAgent(api_key=os.environ["SERPAPI_API_KEY"]),
            # If you add the Reddit agent, uncomment this line:
            # RedditSearchAgent(base_delay=config.REDDIT_DELAY)
        ],
        use_embeddings=config.RAG_USE_EMBEDDINGS,
        scraper_config={
            'max_retries': config.SCRAPER_MAX_RETRIES,
            'timeout': config.SCRAPER_TIMEOUT,
            'rate_limit_delay': config.SCRAPER_RATE_LIMIT_DELAY
        },
        rag_config={
            'chunk_size': config.RAG_CHUNK_SIZE,
            'chunk_overlap': config.RAG_CHUNK_OVERLAP
        }
    )

    while True:
        user_query = input("\nEnter your query (or type 'exit' to quit): ")
        if user_query.lower().strip() in ['exit', 'quit']:
            print("Exiting.")
            break
        
        if not user_query.strip():
            print("Query cannot be empty. Please try again.")
            continue

        print(f"\n▶  Processing query: '{user_query}'")
        print("-" * 40)

        result = await system.query_with_rag(query=user_query, num_search_results=6)
        
        if result.get('error'):
            print(f" Error: {result['error']}")
            continue

        rag_result = result['rag_result']
        scraped_contents = result['scraped_contents']

        print(f" Pipeline completed in: {result['total_processing_time']:.2f}s")
        print(f"   Scraped sites: {len([c for c in scraped_contents if c.success])}/{len(scraped_contents)}")
        print(f"   Retrieved chunks: {len(rag_result.relevant_chunks)}")
        print(f"   Confidence score: {rag_result.confidence_score:.3f}")
        
        sources = rag_result.sources
        print(f"   Sources ({len(sources)}):")
        for src in sources[:3]:
            print(f"     - {src}")
        
        if rag_result.relevant_chunks:
            print(f"   Top similarity scores:")
            for i, chunk in enumerate(rag_result.relevant_chunks[:3]):
                score = chunk.metadata.get('similarity_score', 0)
                source = urlparse(chunk.source_url).netloc
                print(f"     {i+1}. {score:.3f} from {source}")
        
        print(f"\n   Response Preview: {rag_result.generated_response[:300]}...\n")

    stats = system._get_statistics()
    print("\n" + "="*50)
    print("   Final System Statistics:")
    print(f"   Total chunks in index: {stats['total_chunks']}")
    print(f"   Unique sources indexed: {stats['unique_sources']}")
    print(f"   Embeddings enabled: {stats['embeddings_enabled']}")
    print(f"   Avg chunk size: {stats['avg_chunk_size']:.0f} words")
    print("="*50)

if __name__ == "__main__":
    try:
        asyncio.run(test_robust_system())
    except KeyboardInterrupt:
        logger.info("Test interrupted by user.")