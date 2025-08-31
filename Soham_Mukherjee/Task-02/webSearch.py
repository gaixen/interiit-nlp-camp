### requirements: duckduckgo-search

import logging
from duckduckgo_search import DDGS

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def web_search(query, num_results=5):
    logging.info(f"Starting web search for query: '{query}'")
    try:
        with DDGS() as ddgs:
            results = [r for r in ddgs.text(query, max_results=num_results)]
        
        if not results:
            logging.warning("no results found for the query.")
            return []

        urls = [result['href'] for result in results]
        logging.info(f"found {len(urls)} URLs.")
        return urls
    except Exception as e:
        logging.error(f"{str(e)}")
        return []

