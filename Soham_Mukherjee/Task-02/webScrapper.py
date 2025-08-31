### requirements: langchain; beautifulsoup4; requests


import logging
import requests
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def scrape_and_chunk(urls):
    logging.info(f"scrapping {len(urls)} urls...")
    documents = []
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200,length_function=len)

    for url in urls:
        try:
            logging.info(f"scrapping: {url}")
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            for tag in ['nav', 'footer', 'header', 'script', 'style', 'aside']:
                for s in soup.select(tag):
                    s.decompose()

            main_content = soup.find('body')
            if main_content:
                text = main_content.get_text(separator=' ', strip=True)
            else:
                text = soup.get_text(separator=' ', strip=True)

            if not text:
                logging.warning(f"no text found at {url}")
                continue

            chunks = text_splitter.split_text(text)
            for chunk in chunks:
                documents.append({'text': chunk, 'source': url})
            
            logging.info(f"chunking {url}; into {len(chunks)} pieces")

        except requests.RequestException as e:
            logging.error(f"couldn't fetch {url}: {e}")
        except Exception as e:
            logging.error(f"problem with {url}: {e}")
            
    logging.info(f"chunking done for {len(documents)} docs")
    return documents
